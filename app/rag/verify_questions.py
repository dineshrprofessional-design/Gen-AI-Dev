"""Verify the gold question set against the corpus.

    python -m app.rag.verify_questions
    python -m app.rag.verify_questions --show
    python -m app.rag.verify_questions --week4     # the 12-question Week 4 set

Exits 1 if any question points at a section that doesn't exist, or at a section
that doesn't contain its own answer. Run this before trusting any hit-rate
computed from these questions.

`--week4` checks the Week 4 golden set instead, and asks three things the Week 3
check does not: that every derived `gold_chunk_id` resolves, that it is actually
**present in the live index** (a stale `.rag_index` otherwise burns a whole
measurement run producing plausible wrong numbers), and that the answer strings
appear in that *chunk* rather than merely somewhere in the section.
"""

import argparse
import sys
from pathlib import Path

from app.rag.loader import IngestError, load_documents
from app.rag.questions import DEFAULT_QUESTIONS, load_questions, verify

MIN_STRUCTURAL = 3  # the task requires at least three table/code questions


MIN_EXACT_TOKEN = 4  # Week 4 requires at least four exact-token questions
CROWDING_WARN = 4  # questions sharing one gold chunk before it's worth saying so


def verify_week4(docs_root, index_dir, yaml_path, jsonl_path) -> int:
    """Check the Week 4 golden set. Returns a process exit code."""
    from app.rag.chunkers.structure import StructureAwareChunker
    from app.rag.golden_set import derive, load_jsonl, load_yaml
    from app.rag.store import get_store

    try:
        question_set = load_yaml(yaml_path)
        documents = load_documents(docs_root).documents
        rows = derive(question_set, documents)
    except (FileNotFoundError, IngestError, ValueError) as exc:
        print(f"cannot verify: {exc}", file=sys.stderr)
        return 1

    # The committed jsonl must agree with a fresh derivation.
    drift = ""
    try:
        if load_jsonl(jsonl_path) != rows:
            drift = f"{jsonl_path} is stale - re-run `python -m app.rag.golden_set`"
    except FileNotFoundError:
        drift = f"{jsonl_path} does not exist - run `python -m app.rag.golden_set`"

    # Chunk text by id, from the same chunker the index was built with.
    chunker = StructureAwareChunker()
    text_by_id = {c.chunk_id: c.text for d in documents for c in chunker.chunk(d)}

    try:
        indexed = set(get_store("structure", "bge-m3", index_dir).ids())
    except Exception as exc:  # noqa: BLE001 - report, don't crash the check
        print(f"cannot read the index at {index_dir}: {exc}", file=sys.stderr)
        return 1

    print(
        f"{len(rows)} questions against {len(documents)} documents, "
        f"index {index_dir} ({len(indexed)} chunks)"
    )
    print()
    header = f"{'id':5} {'tok':5} {'gold_chunk_id':34} status"
    print(header)
    print("-" * len(header))

    failures: list[str] = []
    for row in rows:
        problems: list[str] = []
        text = text_by_id.get(row.gold_chunk_id)
        if text is None:
            problems.append("chunk not produced by the chunker")
        if row.gold_chunk_id not in indexed:
            problems.append("NOT IN THE LIVE INDEX")
        if text is not None:
            haystack = text.lower()
            missing = [m for m in row.must_contain if m.lower() not in haystack]
            if missing:
                problems.append(f"chunk is missing: {', '.join(missing)}")

        status = "ok" if not problems else "FAIL - " + "; ".join(problems)
        if problems:
            failures.append(row.id)
        flag = "exact" if row.exact_token else "     "
        print(f"{row.id:5} {flag} {row.gold_chunk_id:34} {status}")

    exact = question_set.exact_token_count
    crowding: dict[str, int] = {}
    for row in rows:
        crowding[row.gold_chunk_id] = crowding.get(row.gold_chunk_id, 0) + 1
    worst_id, worst = max(crowding.items(), key=lambda kv: kv[1])

    print()
    print(f"exact-token         : {exact}/{len(rows)}  (need at least {MIN_EXACT_TOKEN})")
    print(f"distinct gold chunks: {len(crowding)}/{len(rows)}")
    print(f"verified            : {len(rows) - len(failures)}/{len(rows)}")
    if worst >= CROWDING_WARN:
        print()
        print(f"warning: {worst} questions share {worst_id} - that measures "
              "one retrieval event several times over")
    if drift:
        print(drift, file=sys.stderr)
        return 1
    if exact < MIN_EXACT_TOKEN:
        print(f"too few exact-token questions: {exact} < {MIN_EXACT_TOKEN}",
              file=sys.stderr)
        return 1
    if failures:
        print(f"{len(failures)} question(s) failed: {', '.join(failures)}",
              file=sys.stderr)
        return 1

    print()
    print("every gold chunk_id resolves, is in the index, and contains its answer")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify the gold question set.")
    parser.add_argument("--docs", type=Path, default=Path("docs"))
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--show", action="store_true", help="print each question")
    parser.add_argument("--index", type=Path, default=Path(".rag_index"))
    parser.add_argument(
        "--week4", action="store_true", help="verify the 12-question Week 4 golden set"
    )
    parser.add_argument("--golden-yaml", type=Path, default=Path("eval/golden_set.yaml"))
    parser.add_argument(
        "--golden-jsonl", type=Path, default=Path("eval/golden_set.jsonl")
    )
    args = parser.parse_args(argv)

    if args.week4:
        return verify_week4(args.docs, args.index, args.golden_yaml, args.golden_jsonl)

    try:
        question_set = load_questions(args.questions)
        documents = load_documents(args.docs).documents
    except (FileNotFoundError, IngestError) as exc:
        print(f"cannot verify: {exc}", file=sys.stderr)
        return 1

    results = verify(question_set, documents)
    by_id = {r.question_id: r for r in results}

    print(f"{len(question_set.questions)} questions against {len(documents)} documents\n")
    header = f"{'id':4} {'depends on':11} {'gold target':34} status"
    print(header)
    print("-" * len(header))
    for question in question_set.questions:
        result = by_id[question.id]
        status = "ok" if result.ok else f"FAIL — {result.reason}"
        print(
            f"{question.id:4} {question.depends_on.value:11} "
            f"{str(question.gold):34} {status}"
        )
        if args.show:
            print(f"       Q: {question.question}")
            print(f"       A: {question.answer}\n")

    structural = question_set.structural_count()
    failures = [r for r in results if not r.ok]

    print(f"\nstructure-dependent : {structural}/{len(question_set.questions)}"
          f"  (need at least {MIN_STRUCTURAL})")
    print(f"verified            : {len(results) - len(failures)}/{len(results)}")

    if structural < MIN_STRUCTURAL:
        print(
            f"\ntoo few structure-dependent questions: {structural} < {MIN_STRUCTURAL}",
            file=sys.stderr,
        )
        return 1
    if failures:
        print(f"\n{len(failures)} question(s) failed verification", file=sys.stderr)
        return 1

    print("\nevery gold target resolves and contains its own answer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
