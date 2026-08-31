"""Build a searchable index, and search it.

    python -m app.rag.index --strategy structure
    python -m app.rag.index --all                     # both chunkers, one embedder
    python -m app.rag.index --search "default retry_backoff_ms"

The pipeline: ingest -> chunk -> embed -> store. One embedder across every
strategy, always — that is the only way the two hit-rates mean anything.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any

from app.rag.chunkers import CHUNKERS
from app.rag.embeddings import get_embedder
from app.rag.loader import IngestError, load_documents
from app.rag.models import Chunk
from app.rag.store import SearchHit, collection_name, get_store

DEFAULT_DOCS = Path("docs")
DEFAULT_INDEX = Path(".rag_index")


def build_index(
    docs_root: Path = DEFAULT_DOCS,
    strategy: str = "structure",
    embedder_name: str = "bge-m3",
    persist_dir: Path | None = DEFAULT_INDEX,
    sdk_version: str | None = None,
) -> dict[str, Any]:
    """Ingest, chunk, embed and store. Returns stats about the run."""
    report = load_documents(docs_root, sdk_version)
    if not report.documents:
        raise IngestError(f"no documents to index in {docs_root}")

    chunker = CHUNKERS[strategy]()
    chunks: list[Chunk] = [c for d in report.documents for c in chunker.chunk(d)]

    embedder = get_embedder(embedder_name)
    store = get_store(strategy, embedder.name, persist_dir)

    # Always start clean. upsert updates and inserts but never deletes, so
    # re-indexing after a chunker change would leave orphaned chunks behind to
    # pollute every later search.
    store.reset()

    started = time.perf_counter()
    vectors = embedder.embed([c.text for c in chunks])
    embed_seconds = time.perf_counter() - started

    store.upsert(chunks, vectors)

    return {
        "strategy": strategy,
        "embedder": embedder.name,
        "collection": collection_name(strategy, embedder.name),
        "documents": report.loaded_count,
        "quarantined": report.failed_count,
        "chunks": len(chunks),
        "stored": store.count(),
        "embed_seconds": round(embed_seconds, 2),
    }


def search(
    query: str,
    strategy: str = "structure",
    embedder_name: str = "bge-m3",
    k: int = 5,
    persist_dir: Path | None = DEFAULT_INDEX,
    where: dict[str, Any] | None = None,
    hybrid: bool = False,
    stats: dict[str, float] | None = None,
) -> list[SearchHit]:
    """Dense vector search, or BM25 + RRF fusion when `hybrid` is set.

    The Week 4 retrieval change is this one keyword and the branch below it. The
    dense path's final line is unchanged, so every caller - the chat route, the
    Week 3 evaluator, the CLI - keeps its existing behaviour by default. The
    import is function-local, matching how chromadb and sentence-transformers
    are already deferred in this package, so the module load graph is unmoved.
    """
    embedder = get_embedder(embedder_name)
    store = get_store(strategy, embedder.name, persist_dir)
    if hybrid:
        from app.rag.retrieval import hybrid_search

        return hybrid_search(
            query,
            embedder=embedder,
            store=store,
            strategy=strategy,
            k=k,
            where=where,
            stats=stats,
        )
    return store.search(embedder.embed_query(query), k=k, where=where)


def print_stats(stats: dict[str, Any]) -> None:
    print(f"\n=== {stats['collection']} ===")
    for key in ("documents", "quarantined", "chunks", "stored", "embed_seconds"):
        print(f"  {key:<14} {stats[key]}")


def print_hits(hits: list[SearchHit], query: str, hybrid: bool = False) -> None:
    print(f'\nquery: "{query}"')
    if not hits:
        print("  no results")
        return
    if hybrid:
        # Fusion diagnostics. Printed only for --hybrid, so the dense table
        # below stays byte-identical to what it printed before this change.
        print(f"  {'#':<3} {'score':<8} {'rrf':<9} {'d':<4} {'b':<4} chunk_id")
        print("  " + "-" * 76)
        for rank, hit in enumerate(hits, start=1):
            meta = hit.metadata
            d = meta.get("dense_rank")
            b = meta.get("bm25_rank")
            print(
                f"  {rank:<3} {hit.score:<8.4f} "
                f"{meta.get('rrf_score', 0.0):<9.6f} "
                f"{'-' if d is None else d:<4} {'-' if b is None else b:<4} "
                f"{hit.chunk_id}"
            )
        return
    print(f"  {'#':<3} {'score':<8} {'target':<34} chunk_id")
    print("  " + "-" * 76)
    for rank, hit in enumerate(hits, start=1):
        print(f"  {rank:<3} {hit.score:<8.4f} {hit.target:<34} {hit.chunk_id}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build and query the vector index.")
    parser.add_argument("--docs", type=Path, default=DEFAULT_DOCS)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--strategy", choices=sorted(CHUNKERS), default="structure")
    parser.add_argument("--embedder", default="bge-m3")
    parser.add_argument("--version", default=None, help="index one sdk_version only")
    parser.add_argument(
        "--all", action="store_true", help="index every chunking strategy"
    )
    parser.add_argument("--search", default=None, help="query instead of indexing")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument(
        "--filter-version", default=None, help="restrict a search to one sdk_version"
    )
    parser.add_argument(
        "--hybrid", action="store_true", help="BM25 + RRF fusion instead of dense only"
    )
    args = parser.parse_args(argv)

    if args.search:
        where = {"sdk_version": args.filter_version} if args.filter_version else None
        try:
            hits = search(
                args.search,
                strategy=args.strategy,
                embedder_name=args.embedder,
                k=args.k,
                persist_dir=args.index,
                where=where,
                hybrid=args.hybrid,
            )
        except Exception as exc:
            print(f"search failed: {exc}", file=sys.stderr)
            return 1
        print_hits(hits, args.search, hybrid=args.hybrid)
        return 0

    strategies = sorted(CHUNKERS) if args.all else [args.strategy]
    print(f"embedder: {args.embedder}   (frozen across every strategy)")
    for strategy in strategies:
        try:
            stats = build_index(
                docs_root=args.docs,
                strategy=strategy,
                embedder_name=args.embedder,
                persist_dir=args.index,
                sdk_version=args.version,
            )
        except IngestError as exc:
            print(f"indexing failed: {exc}", file=sys.stderr)
            return 1
        print_stats(stats)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
