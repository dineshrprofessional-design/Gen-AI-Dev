"""Replay a trace from its own fields, and draw the seeded random samples.

    python -m app.rag.replay --pick --seed 20260831      # choose 1 trace_id, replay it
    python -m app.rag.replay --replay <trace_id>          # replay a specific trace
    python -m app.rag.replay --sample 20 --seed 20260831  # the Week 5 sample

## What "replayable" means here

A trace must contain enough to re-execute the request without consulting
anything else the app once knew: the question, the retrieval configuration
(strategy / hybrid / k / version filter), the model and its parameters, and the
prompt version. Replay re-runs retrieval from those fields and compares the
returned chunk_ids and scores against the recorded ones, then re-runs
generation over the RECORDED chunk ids (their text is resolved from the corpus,
which is versioned by the trace's git_sha) with the recorded model, and prints
both outputs side by side.

## What cannot be reconstructed, stated up front

- Wall-clock latency and the timestamp - properties of the original moment.
- Token-for-token LLM output. temperature=0 makes it usually stable, but the
  provider does not guarantee determinism; the replay shows both texts so the
  reader judges the difference instead of being told it is zero.
- If the corpus or prompt changed since (different git_sha / prompt_version),
  the replay reports that instead of pretending the comparison is clean.
"""

import argparse
import random
import sys
from pathlib import Path

from app.rag import trace as tracelib
from app.rag.index import DEFAULT_DOCS, DEFAULT_INDEX, search


def _chunk_texts(docs_root: Path) -> dict[str, str]:
    from app.rag.chunkers import CHUNKERS
    from app.rag.loader import load_documents

    documents = load_documents(docs_root).documents
    texts: dict[str, str] = {}
    for name in CHUNKERS:
        chunker = CHUNKERS[name]()
        for doc in documents:
            for chunk in chunker.chunk(doc):
                texts[chunk.chunk_id] = chunk.text
    return texts


def organic_ids(manifest_path: Path = Path("traces/pool_manifest.json")) -> list[str]:
    import json

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [row["trace_id"] for row in manifest["organic"] if row.get("trace_id")]


def replay(trace_id: str, docs_root: Path, index_dir: Path) -> int:
    record = tracelib.by_id(trace_id)
    cfg = record["retrieval"]

    print(f"trace {trace_id}  ts={record['ts']}  git={record['git_sha']}  "
          f"prompt={record['prompt_version']}")
    print(f"question : {record['question']!r}")
    print(f"config   : strategy={cfg['strategy']} hybrid={cfg['hybrid']} "
          f"k={cfg['k']} filter={cfg['sdk_version_filter']}")

    drift = []
    if record["git_sha"] != tracelib.git_sha():
        drift.append(f"git_sha (trace {record['git_sha']}, now {tracelib.git_sha()})")
    if record["prompt_version"] != tracelib.prompt_version():
        drift.append("prompt_version")
    if drift:
        print(f"NOTE: environment drifted since the trace: {', '.join(drift)}")

    where = (
        {"sdk_version": cfg["sdk_version_filter"]} if cfg["sdk_version_filter"] else None
    )
    hits = search(
        record["question"],
        strategy=cfg["strategy"],
        k=cfg["k"],
        persist_dir=index_dir,
        where=where,
        hybrid=cfg["hybrid"],
    )

    print()
    print("retrieval, original vs replayed:")
    print(f"  {'#':<3} {'recorded chunk_id':<36} {'score':<8} "
          f"{'replayed chunk_id':<36} {'score':<8} match")
    identical = True
    for index in range(max(len(record["retrieved"]), len(hits))):
        old = record["retrieved"][index] if index < len(record["retrieved"]) else None
        new = hits[index] if index < len(hits) else None
        old_id = old["chunk_id"] if old else "-"
        old_score = f"{old['score']:.4f}" if old else "-"
        new_id = new.chunk_id if new else "-"
        new_score = f"{new.score:.4f}" if new else "-"
        same = "yes" if (old and new and old_id == new_id) else "NO"
        identical = identical and same == "yes"
        print(f"  {index + 1:<3} {old_id:<36} {old_score:<8} {new_id:<36} {new_score:<8} {same}")
    print(f"  -> retrieval replay {'IDENTICAL' if identical else 'DIFFERS'}")

    if not record["generation_ran"]:
        reason = record["refusal_reason"] or record["error"] or "generation was off"
        print()
        print(f"generation did not run originally ({reason}); nothing to replay there.")
        return 0

    from app.rag.generate import answer as generate_answer
    from app.rag.store import SearchHit

    texts = _chunk_texts(docs_root)
    missing = [r["chunk_id"] for r in record["retrieved"] if r["chunk_id"] not in texts]
    if missing:
        print(f"cannot reconstruct chunk text for: {missing}")
        return 1

    recorded_hits = [
        SearchHit(r["chunk_id"], r["score"], texts[r["chunk_id"]], {})
        for r in record["retrieved"][:3]
    ]
    result = generate_answer(record["question"], recorded_hits, model=record["model"])

    print()
    print(f"model    : {record['model']}  params={record['params']}")
    print(f"original : {record['raw_output']!r}")
    print(f"replayed : {result.text!r}")
    print(f"  -> outputs {'IDENTICAL' if result.text == record['raw_output'] else 'DIFFER'}"
          "  (temperature=0; provider does not guarantee token determinism)")
    print()
    print("not reconstructable from the trace: wall-clock latency, timestamp.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay traces; draw seeded samples.")
    parser.add_argument("--docs", type=Path, default=DEFAULT_DOCS)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--replay", metavar="TRACE_ID", default=None)
    parser.add_argument("--pick", action="store_true", help="seeded random pick, then replay")
    parser.add_argument("--sample", type=int, default=None, metavar="N")
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args(argv)

    if args.sample:
        ids = organic_ids()
        chosen = random.Random(args.seed).sample(ids, args.sample)
        print(f"seed={args.seed}  population={len(ids)} organic traces  sample={args.sample}")
        for trace_id in chosen:
            record = tracelib.by_id(trace_id)
            print(f"  {trace_id}  {record['question']!r}")
        return 0

    if args.pick:
        ids = organic_ids()
        trace_id = random.Random(args.seed).choice(ids)
        print(f"seed={args.seed}  picked {trace_id} of {len(ids)} organic traces")
        print()
        return replay(trace_id, args.docs, args.index)

    if args.replay:
        return replay(args.replay, args.docs, args.index)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
