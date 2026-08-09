"""Chunking CLI.

    python -m app.rag.chunk --strategy fixed
    python -m app.rag.chunk --strategy structure --show
    python -m app.rag.chunk --compare

Takes the Documents that ingestion produced and splits them into Chunks.
Embedding, storing and retrieval are later phases and are not wired in here.
"""

import argparse
import sys
from pathlib import Path

from app.rag.chunkers import CHUNKERS
from app.rag.chunkers.base import has_balanced_fences, has_table_header, has_table_row
from app.rag.loader import IngestError, load_documents
from app.rag.models import Chunk, Document

DEFAULT_DOCS = Path("docs")


def build_chunks(documents: list[Document], strategy: str) -> list[Chunk]:
    chunker = CHUNKERS[strategy]()
    return [chunk for doc in documents for chunk in chunker.chunk(doc)]


def damage_report(chunks: list[Chunk]) -> dict[str, list[str]]:
    """Find structural damage.

    Reported, never raised — the baseline chunker is *expected* to fail these,
    and those failures are precisely the measurement.
    """
    return {
        "split_code_fences": [
            c.chunk_id for c in chunks if not has_balanced_fences(c.text)
        ],
        "orphaned_table_rows": [
            c.chunk_id
            for c in chunks
            if has_table_row(c.text) and not has_table_header(c.text)
        ],
    }


def print_summary(chunks: list[Chunk], strategy: str) -> None:
    sizes = sorted(len(c.text) for c in chunks)
    damage = damage_report(chunks)

    print(f"\n=== strategy: {strategy} ===")
    print(f"chunks            : {len(chunks)}")
    print(f"pages             : {len({(c.sdk_version, c.page_id) for c in chunks})}")
    print(f"size min/med/max  : {sizes[0]} / {sizes[len(sizes) // 2]} / {sizes[-1]}")
    print(f"holding a table   : {sum(c.has_table_row for c in chunks)}")
    print(f"holding code      : {sum(c.has_code_fence for c in chunks)}")

    print("-- structural damage --")
    for label, ids in damage.items():
        status = "none" if not ids else f"{len(ids)} -> {', '.join(ids)}"
        print(f"{label:20}: {status}")


def print_chunks(chunks: list[Chunk]) -> None:
    for chunk in chunks:
        print("\n" + "-" * 72)
        print(f"[{chunk.chunk_id}]   chars {chunk.char_start}:{chunk.char_end}")
        print(f"path: {chunk.heading_path or '(none — this chunker has no idea)'}")
        print("-" * 72)
        print(chunk.text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chunk an ingested corpus.")
    parser.add_argument("--docs", type=Path, default=DEFAULT_DOCS)
    parser.add_argument("--version", default=None, help="one sdk_version only")
    parser.add_argument(
        "--strategy", choices=sorted(CHUNKERS), default=sorted(CHUNKERS)[-1]
    )
    parser.add_argument("--show", action="store_true", help="print every chunk")
    parser.add_argument(
        "--compare", action="store_true", help="run both strategies side by side"
    )
    args = parser.parse_args(argv)

    try:
        report = load_documents(args.docs, args.version)
    except IngestError as exc:
        print(f"ingest failed: {exc}", file=sys.stderr)
        return 1

    if not report.documents:
        print("no documents to chunk", file=sys.stderr)
        return 1

    print(f"chunking {report.loaded_count} document(s) from {args.docs}")
    if report.failures:
        print(f"({report.failed_count} file(s) were quarantined at ingest)")

    strategies = sorted(CHUNKERS) if args.compare else [args.strategy]
    for strategy in strategies:
        chunks = build_chunks(report.documents, strategy)
        if args.show and not args.compare:
            print_chunks(chunks)
        print_summary(chunks, strategy)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
