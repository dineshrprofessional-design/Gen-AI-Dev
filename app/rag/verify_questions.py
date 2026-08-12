"""Verify the gold question set against the corpus.

    python -m app.rag.verify_questions
    python -m app.rag.verify_questions --show

Exits 1 if any question points at a section that doesn't exist, or at a section
that doesn't contain its own answer. Run this before trusting any hit-rate
computed from these questions.
"""

import argparse
import sys
from pathlib import Path

from app.rag.loader import IngestError, load_documents
from app.rag.questions import DEFAULT_QUESTIONS, load_questions, verify

MIN_STRUCTURAL = 3  # the task requires at least three table/code questions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify the gold question set.")
    parser.add_argument("--docs", type=Path, default=Path("docs"))
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--show", action="store_true", help="print each question")
    args = parser.parse_args(argv)

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
