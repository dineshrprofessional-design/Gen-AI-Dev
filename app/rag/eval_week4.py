"""hit-rate@3 over the Week 4 golden set, before and after one retrieval change.

    python -m app.rag.eval_week4 --out report/w4/before.json
    python -m app.rag.eval_week4 --hybrid --out report/w4/after.json
    python -m app.rag.eval_week4 --compare report/w4/before.json report/w4/after.json
    python -m app.rag.eval_week4 --detail

## Why this is not `evaluate.py`

The Week 3 evaluator scores a hit by character-range overlap with a gold
*section*, always runs both chunking strategies, and defaults to k=5. All three
are wrong here: Week 4 needs chunk_id equality, exactly one strategy (a second
one is a second variable, and the fixed chunker emits no anchors so its ids are
not comparable at all), and k=3 as the headline. Bending `evaluate.py` to do
both jobs would put Week 3's already-reported numbers at risk, so the result
shapes are copied rather than shared.

## Retrieval runs once per question, three metrics come out

`k=5` is retrieved and hit@1/@3/@5 are read off the same list. That is sound
because the candidate pool (`FETCH_K`) is fixed independently of `k`, so `k`
only truncates an ordering that was already decided - asking for 3 or for 5
cannot change which chunk is third.

## Latency

Timed around the public `search()` call, by the same wrapper in both runs, so
the comparison is symmetric by construction. One warm-up query is discarded:
the first call pays the ~2.2 GB model load, the first HNSW query materialises
the graph, and the first BM25 call builds its index. None are per-query costs,
and counting them would make whichever run went first look slower.

`p50` is reported two ways - the median of the twelve per-question medians, and
the pooled median over every sample - because "p50" is otherwise ambiguous
between them.
"""

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

from pydantic import BaseModel, Field

from app.rag.fingerprint import IndexFingerprint, assert_same, fingerprint
from app.rag.golden_set import DEFAULT_GOLDEN_JSONL, GoldenRow, load_jsonl
from app.rag.index import DEFAULT_DOCS, DEFAULT_INDEX, search

WARMUP_QUERY = "how do I configure the client"  # identical in both runs


class RunConfig(BaseModel):
    hybrid: bool = False
    strategy: str = "structure"
    embedder: str = "bge-m3"
    k: int = 5
    hit_at: int = 3
    reps: int = 5
    docs_root: str = str(DEFAULT_DOCS)
    index_dir: str = str(DEFAULT_INDEX)
    golden_path: str = str(DEFAULT_GOLDEN_JSONL)


class Week4Hit(BaseModel):
    rank: int
    chunk_id: str
    score: float
    rrf_score: float | None = None
    dense_rank: int | None = None
    bm25_rank: int | None = None
    is_gold: bool = False
    contains_answer: bool = False


class Week4Outcome(BaseModel):
    question_id: str
    question: str
    depends_on: str
    exact_token: str | None
    gold_chunk_id: str
    hit_rank: int | None = None
    hits: list[Week4Hit] = Field(default_factory=list)
    latencies_ms: list[float] = Field(default_factory=list)
    embed_only_ms: float = 0.0
    component_ms: dict[str, float] = Field(default_factory=dict)

    def hit_at(self, n: int) -> bool:
        return self.hit_rank is not None and self.hit_rank <= n

    @property
    def p50_ms(self) -> float:
        return statistics.median(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def answerable_from_top3(self) -> bool:
        """Did any top-3 chunk contain the answer, gold or not?

        True on a metric miss means retrieval failed the *tag* but not the
        reader. Worth separating: it is the difference between a hit-rate delta
        and a user-visible improvement.
        """
        return any(h.contains_answer for h in self.hits[:3])


class Week4Run(BaseModel):
    config: RunConfig
    fingerprint: IndexFingerprint
    environment: dict[str, str]
    outcomes: list[Week4Outcome] = Field(default_factory=list)

    def hits(self, n: int) -> int:
        return sum(o.hit_at(n) for o in self.outcomes)

    def hit_rate(self, n: int) -> str:
        return f"{self.hits(n)}/{len(self.outcomes)}"

    @property
    def per_question_p50_ms(self) -> float:
        return statistics.median([o.p50_ms for o in self.outcomes])

    @property
    def pooled_p50_ms(self) -> float:
        return statistics.median([x for o in self.outcomes for x in o.latencies_ms])

    @property
    def pooled_p95_ms(self) -> float:
        samples = sorted(x for o in self.outcomes for x in o.latencies_ms)
        index = min(len(samples) - 1, int(round(0.95 * (len(samples) - 1))))
        return samples[index]

    @property
    def pooled_min_ms(self) -> float:
        return min(x for o in self.outcomes for x in o.latencies_ms)

    @property
    def embed_share_ms(self) -> float:
        return statistics.median([o.embed_only_ms for o in self.outcomes])


def _environment() -> dict[str, str]:
    import chromadb

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "chromadb": getattr(chromadb, "__version__", "unknown"),
    }


def run_one(row: GoldenRow, cfg: RunConfig) -> Week4Outcome:
    from app.rag.embeddings import get_embedder

    index_dir = Path(cfg.index_dir)
    latencies: list[float] = []
    hits = []
    component: dict[str, float] = {}

    for _ in range(cfg.reps):
        stats: dict[str, float] = {}
        started = time.perf_counter()
        hits = search(
            row.question,
            strategy=cfg.strategy,
            embedder_name=cfg.embedder,
            k=cfg.k,
            persist_dir=index_dir,
            hybrid=cfg.hybrid,
            stats=stats,
        )
        latencies.append((time.perf_counter() - started) * 1000)
        if stats:
            component = {key: round(value, 3) for key, value in stats.items()}

    # The one component measurement that is symmetric across both runs.
    embedder = get_embedder(cfg.embedder)
    started = time.perf_counter()
    embedder.embed_query(row.question)
    embed_only_ms = (time.perf_counter() - started) * 1000

    records: list[Week4Hit] = []
    hit_rank: int | None = None
    needles = [m.lower() for m in row.must_contain]
    for rank, hit in enumerate(hits, start=1):
        is_gold = hit.chunk_id == row.gold_chunk_id
        if is_gold and hit_rank is None:
            hit_rank = rank
        body = hit.text.lower()
        records.append(
            Week4Hit(
                rank=rank,
                chunk_id=hit.chunk_id,
                score=round(hit.score, 4),
                rrf_score=hit.metadata.get("rrf_score"),
                dense_rank=hit.metadata.get("dense_rank"),
                bm25_rank=hit.metadata.get("bm25_rank"),
                is_gold=is_gold,
                contains_answer=all(n in body for n in needles) if needles else False,
            )
        )

    return Week4Outcome(
        question_id=row.id,
        question=row.question,
        depends_on=row.depends_on,
        exact_token=row.exact_token,
        gold_chunk_id=row.gold_chunk_id,
        hit_rank=hit_rank,
        hits=records,
        latencies_ms=[round(x, 3) for x in latencies],
        embed_only_ms=round(embed_only_ms, 3),
        component_ms=component,
    )


def run(cfg: RunConfig) -> Week4Run:
    rows = load_jsonl(Path(cfg.golden_path))
    mismatched = [r.id for r in rows if r.strategy != cfg.strategy]
    if mismatched:
        raise ValueError(
            f"the golden set is tagged for strategy {rows[0].strategy!r} but this "
            f"run uses {cfg.strategy!r}; chunk_ids are not comparable across "
            f"chunkers ({len(mismatched)} rows affected)"
        )

    fp = fingerprint(
        cfg.strategy,
        cfg.embedder,
        Path(cfg.docs_root),
        Path(cfg.index_dir),
        Path(cfg.golden_path),
    )
    if not fp.index_matches_corpus:
        raise ValueError(
            "the index does not match the corpus - chunk ids differ from a fresh "
            "chunking of the docs root. Re-index before measuring anything."
        )

    # Warm-up, discarded. Pays the model load, the HNSW graph and the BM25 build.
    search(
        WARMUP_QUERY,
        strategy=cfg.strategy,
        embedder_name=cfg.embedder,
        k=cfg.k,
        persist_dir=Path(cfg.index_dir),
        hybrid=cfg.hybrid,
    )

    return Week4Run(
        config=cfg,
        fingerprint=fp,
        environment=_environment(),
        outcomes=[run_one(row, cfg) for row in rows],
    )


def print_table(result: Week4Run) -> None:
    cfg = result.config
    mode = "hybrid (BM25 + RRF)" if cfg.hybrid else "dense only"
    print(f"mode      : {mode}")
    print(f"strategy  : {cfg.strategy}   embedder: {cfg.embedder}   k={cfg.k}")
    print(f"questions : {len(result.outcomes)}   reps={cfg.reps}")
    print()
    header = f"{'id':5} {'tok':6} {'gold_chunk_id':34} {'@3':6} {'rank':5} {'p50 ms':>8}"
    print(header)
    print("-" * len(header))
    for o in result.outcomes:
        cell = "HIT" if o.hit_at(cfg.hit_at) else "MISS"
        rank = str(o.hit_rank) if o.hit_rank else "-"
        flag = "exact" if o.exact_token else ""
        print(
            f"{o.question_id:5} {flag:6} {o.gold_chunk_id:34} "
            f"{cell:6} {rank:5} {o.p50_ms:8.1f}"
        )
    print("-" * len(header))
    print(f"hit-rate@1 : {result.hit_rate(1)}")
    print(f"hit-rate@3 : {result.hit_rate(3)}      <- the metric")
    print(f"hit-rate@5 : {result.hit_rate(5)}")
    print()
    print(f"p50, median of per-question medians : {result.per_question_p50_ms:8.1f} ms")
    print(f"p50, pooled over all samples        : {result.pooled_p50_ms:8.1f} ms")
    print(f"p95, pooled                         : {result.pooled_p95_ms:8.1f} ms")
    print(f"min, pooled                         : {result.pooled_min_ms:8.1f} ms")
    print(f"of which query embedding, median    : {result.embed_share_ms:8.1f} ms")

    if cfg.hybrid:
        parts: dict[str, list[float]] = {}
        for o in result.outcomes:
            for key, value in o.component_ms.items():
                parts.setdefault(key, []).append(value)
        if parts:
            print()
            print("component split (hybrid-only diagnostic, not a like-for-like split):")
            for key in ("embed_ms", "dense_ms", "bm25_ms", "fuse_ms"):
                if key in parts:
                    print(f"  {key:<10} median {statistics.median(parts[key]):8.3f} ms")


def print_detail(result: Week4Run) -> None:
    mode = "hybrid" if result.config.hybrid else "dense"
    print()
    print(f"=== inspection dump: {mode} (top {result.config.k}) ===")
    print("    * = the gold chunk    A = this chunk contains the answer strings")
    for o in result.outcomes:
        flag = f"HIT at rank {o.hit_rank}" if o.hit_rank else "MISS"
        outside = "" if o.hit_at(3) else "   <- outside top-3"
        print()
        print(f"{o.question_id}  {o.question!r}")
        print(f"  gold={o.gold_chunk_id}  -> {flag}{outside}")
        for h in o.hits:
            mark = "*" if h.is_gold else " "
            ans = "A" if h.contains_answer else " "
            extra = ""
            if h.rrf_score is not None:
                d = "-" if h.dense_rank is None else h.dense_rank
                b = "-" if h.bm25_rank is None else h.bm25_rank
                extra = f"  rrf={h.rrf_score:.6f} d={d} b={b}"
            print(f"  {mark}{ans} {h.rank}. {h.score:<8.4f} {h.chunk_id:34}{extra}")


def verdict_for(before: Week4Outcome, after: Week4Outcome, n: int = 3) -> str:
    """Five outcomes, deliberately distinct.

    "untouched" and "unfixed" are different claims: one was never broken, the
    other is still broken. The rubric asks which failures the change fixed AND
    which it left untouched, so collapsing them loses the mark.
    """
    was, now = before.hit_at(n), after.hit_at(n)
    if not was and now:
        return "fixed"
    if not was and not now:
        return "unfixed (still R)"
    if was and now:
        return "untouched (never failed)"
    return "REGRESSED"


def print_compare(before: Week4Run, after: Week4Run) -> int:
    moved = assert_same(before.fingerprint, after.fingerprint)
    changed = [
        f
        for f in before.config.model_fields
        if getattr(before.config, f) != getattr(after.config, f)
    ]
    ok = not moved and changed == ["hybrid"]

    print("## one-variable check")
    print()
    print(f"fingerprint fields that differ : {moved if moved else 'none'}")
    print(f"config keys that differ        : {changed}")
    print(f"verdict                        : {'PASS' if ok else 'FAIL'}")
    print()

    total = len(before.outcomes)
    print("## hit-rate and latency")
    print()
    print("| metric | before (dense) | after (hybrid) | delta |")
    print("| --- | --- | --- | --- |")
    for n in (1, 3, 5):
        b, a = before.hits(n), after.hits(n)
        note = " **(the metric)**" if n == 3 else ""
        print(f"| hit-rate@{n}{note} | {b}/{total} | {a}/{total} | {a - b:+d} |")
    pb, pa = before.per_question_p50_ms, after.per_question_p50_ms
    pct = (pa - pb) / pb * 100 if pb else 0.0
    print(
        f"| p50 ms, median of per-question medians | {pb:.1f} | {pa:.1f} "
        f"| {pa - pb:+.1f} ({pct:+.1f}%) |"
    )
    print(
        f"| p50 ms, pooled | {before.pooled_p50_ms:.1f} | {after.pooled_p50_ms:.1f} "
        f"| {after.pooled_p50_ms - before.pooled_p50_ms:+.1f} |"
    )
    print(
        f"| p95 ms, pooled | {before.pooled_p95_ms:.1f} | {after.pooled_p95_ms:.1f} "
        f"| {after.pooled_p95_ms - before.pooled_p95_ms:+.1f} |"
    )
    print(
        f"| query embedding, median ms | {before.embed_share_ms:.1f} "
        f"| {after.embed_share_ms:.1f} | - |"
    )
    print()

    print("## per question")
    print()
    print("| id | exact token | before | after | verdict |")
    print("| --- | --- | --- | --- | --- |")
    by_id = {o.question_id: o for o in after.outcomes}
    for b in before.outcomes:
        a = by_id[b.question_id]
        brank = b.hit_rank or "-"
        arank = a.hit_rank or "-"
        verdict = verdict_for(b, a)
        emphasis = verdict in ("fixed", "REGRESSED")
        shown = f"**{verdict}**" if emphasis else verdict
        print(
            f"| {b.question_id} | {b.exact_token or '-'} "
            f"| {'HIT' if b.hit_at(3) else 'MISS'} @{brank} "
            f"| {'HIT' if a.hit_at(3) else 'MISS'} @{arank} | {shown} |"
        )
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Week 4 hit-rate@3, before and after.")
    parser.add_argument("--docs", type=Path, default=DEFAULT_DOCS)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN_JSONL)
    parser.add_argument("--embedder", default="bge-m3")
    parser.add_argument("--strategy", default="structure")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--hit-at", type=int, default=3)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--hybrid", action="store_true", help="the one retrieval change")
    parser.add_argument("--detail", action="store_true", help="full inspection dump")
    parser.add_argument("--out", type=Path, default=None, help="write the run as JSON")
    parser.add_argument(
        "--compare", nargs=2, type=Path, default=None, metavar=("BEFORE", "AFTER")
    )
    args = parser.parse_args(argv)

    if args.compare:
        before = Week4Run(**json.loads(args.compare[0].read_text(encoding="utf-8")))
        after = Week4Run(**json.loads(args.compare[1].read_text(encoding="utf-8")))
        return print_compare(before, after)

    cfg = RunConfig(
        hybrid=args.hybrid,
        strategy=args.strategy,
        embedder=args.embedder,
        k=args.k,
        hit_at=args.hit_at,
        reps=args.reps,
        docs_root=str(args.docs),
        index_dir=str(args.index),
        golden_path=str(args.golden),
    )

    try:
        result = run(cfg)
    except (FileNotFoundError, ValueError) as exc:
        print(f"evaluation failed: {exc}", file=sys.stderr)
        return 1

    print_table(result)
    if args.detail:
        print_detail(result)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result.model_dump(), indent=2), encoding="utf-8")
        print()
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
