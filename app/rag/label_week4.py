"""Label every Week 4 failure R / G / Not-In-Corpus, with generation in the loop.

    python -m app.rag.label_week4 --out report/w4/labels.json

hit-rate@3 is a *retrieval* metric; G is a *generation* failure. They do not
share a denominator, so this pass builds two populations rather than one:

- **Population A** - the gold chunk is not in the top-3. These are exactly the
  questions hit-rate@3 counts as misses. Each is R or Not-In-Corpus.
- **Population B** - the gold chunk *was* in the top-3 and the answer is still
  wrong. Invisible to hit-rate@3 by construction, and unreachable by any
  retrieval change. These are G.

The decision order matters and is the guard against the task's fourth listed
mistake - never label a failure R because the answer was wrong without first
checking whether the parameter table was sitting in the top-3 all along:

    1. answer strings absent from the whole corpus  -> Not-In-Corpus
    2. gold chunk not in top-3                      -> R
    3. gold chunk in top-3 but the answer is wrong  -> G

Answer correctness is judged by the recorded transcript, not by this script
guessing: it records the question, the top-3 window, the generated text, the
citations and the refusal reason, and marks `answer_contains_expected` as a
mechanical hint. The final call belongs to whoever reads the evidence, which is
what "run every miss through the inspection view" means.
"""

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

from pydantic import BaseModel, Field

from app.rag.golden_set import DEFAULT_GOLDEN_JSONL, load_jsonl
from app.rag.index import DEFAULT_DOCS, DEFAULT_INDEX, search


class LabelRecord(BaseModel):
    question_id: str
    question: str
    gold_chunk_id: str
    exact_token: str | None
    expected_answer: str
    must_contain: list[str]

    top3: list[dict] = Field(default_factory=list)
    gold_rank: int | None = None
    gold_in_top3: bool = False
    answerable_from_top3: bool = False
    in_corpus: bool = True

    generated: str = ""
    refused: bool = False
    refusal_reason: str = ""
    citations: list[str] = Field(default_factory=list)
    invalid_citations: list[str] = Field(default_factory=list)
    top_score: float = 0.0
    answer_contains_expected: bool = False

    population: str = ""  # A (retrieval miss) | B (answer failure) | pass
    proposed_label: str = ""  # R | G | Not-In-Corpus | pass


def normalise(text: str) -> str:
    """Fold the ways a model can write the same literal and still be right.

    The first pass of this check produced three false G labels: a correct
    "8,388,608 bytes" (thousands separators) and two correct answers that used
    a narrow no-break space before the number. A labelling bug that invents
    generation failures is worse than no check at all, because the tally is the
    evidence. So: NFKC-fold the unicode spaces, then drop separators between
    digits.
    """
    folded = unicodedata.normalize("NFKC", text).lower()
    folded = folded.replace(" ", " ").replace(" ", " ")
    return re.sub(r"(?<=\d)[,_ ](?=\d)", "", folded)


def corpus_contains(needles: list[str], docs_root: Path) -> bool:
    """Is the answer anywhere in the corpus at all? Decides Not-In-Corpus."""
    from app.rag.loader import load_documents

    blob = "\n".join(d.text for d in load_documents(docs_root).documents).lower()
    return all(n.lower() in blob for n in needles)


def label_all(
    golden_path: Path,
    docs_root: Path,
    index_dir: Path,
    hybrid: bool = False,
    generate: bool = True,
    model: str | None = None,
) -> list[LabelRecord]:
    from app.rag import generate as gen

    rows = load_jsonl(golden_path)
    records: list[LabelRecord] = []

    for row in rows:
        hits = search(
            row.question,
            strategy=row.strategy,
            k=5,
            persist_dir=index_dir,
            hybrid=hybrid,
        )
        top3 = hits[:3]
        needles = [m.lower() for m in row.must_contain]

        gold_rank = next(
            (r for r, h in enumerate(hits, start=1) if h.chunk_id == row.gold_chunk_id),
            None,
        )
        record = LabelRecord(
            question_id=row.id,
            question=row.question,
            gold_chunk_id=row.gold_chunk_id,
            exact_token=row.exact_token,
            expected_answer=row.answer,
            must_contain=row.must_contain,
            top3=[
                {
                    "rank": r,
                    "chunk_id": h.chunk_id,
                    "score": round(h.score, 4),
                    "contains_answer": all(n in h.text.lower() for n in needles),
                }
                for r, h in enumerate(top3, start=1)
            ],
            gold_rank=gold_rank,
            gold_in_top3=gold_rank is not None and gold_rank <= 3,
            answerable_from_top3=any(
                all(n in h.text.lower() for n in needles) for h in top3
            ),
            in_corpus=corpus_contains(row.must_contain, docs_root),
        )

        if generate and gen.is_configured():
            answer = gen.answer(row.question, top3, model=model)
            record.generated = answer.text
            record.refused = answer.refused
            record.refusal_reason = answer.reason
            record.citations = answer.citations
            record.invalid_citations = answer.invalid_citations
            record.top_score = round(answer.top_score, 4)
            # A mechanical hint only. The expected answer is prose, so this
            # looks for the answer's distinguishing literal tokens, normalised
            # so a correctly-formatted number is not mistaken for a wrong one.
            body = normalise(answer.text)
            tokens = [t for t in needles if any(ch.isdigit() for ch in t)] or needles
            record.answer_contains_expected = any(
                normalise(t) in body for t in tokens
            )

        floor_refusal = record.refused and "below floor" in record.refusal_reason

        if not record.in_corpus:
            record.population, record.proposed_label = "A", "Not-In-Corpus"
        elif not record.gold_in_top3:
            record.population, record.proposed_label = "A", "R"
        elif floor_refusal:
            # Retrieval put the answer in the window and the score floor threw
            # it away before the model was ever called. Not R (the gold was
            # retrieved) and not G (no model misused anything) - a pipeline
            # gate, reported separately so it cannot be laundered into either.
            record.population = "gate"
            record.proposed_label = "score-floor refusal"
        elif generate and not record.answer_contains_expected:
            record.population, record.proposed_label = "B", "G (confirm by reading)"
        else:
            record.population, record.proposed_label = "pass", "pass"

        records.append(record)

    return records


def print_report(records: list[LabelRecord]) -> None:
    total = len(records)
    pop_a = [r for r in records if r.population == "A"]
    pop_b = [r for r in records if r.population == "B"]
    r_count = sum(r.proposed_label == "R" for r in records)
    nic = sum(r.proposed_label == "Not-In-Corpus" for r in records)

    print(f"{total} questions")
    print()
    header = f"{'id':5} {'pop':4} {'label':22} {'gold rank':9} {'ans?':5} evidence"
    print(header)
    print("-" * 100)
    for r in records:
        rank = str(r.gold_rank) if r.gold_rank else "-"
        ans = "yes" if r.answer_contains_expected else "no"
        top1 = r.top3[0] if r.top3 else {}
        evidence = f"top1={top1.get('chunk_id', '-')} ({top1.get('score', 0)})"
        if r.answerable_from_top3 and not r.gold_in_top3:
            evidence += "  [answer WAS reachable from top-3]"
        print(f"{r.question_id:5} {r.population:4} {r.proposed_label:22} {rank:9} {ans:5} {evidence}")

    print("-" * 100)
    print()
    print(f"Population A - retrieval misses (counted by hit-rate@3) : {len(pop_a)}/{total}")
    print(f"    R                                                  : {r_count}")
    print(f"    Not-In-Corpus                                      : {nic}")
    print(f"Population B - wrong answer despite a top-3 hit (NOT in hit-rate@3) : {len(pop_b)}")
    print(f"    G                                                  : {len(pop_b)}")
    gate = [r for r in records if r.population == "gate"]
    print(f"Score-floor refusals - gold retrieved, gate refused it : {len(gate)}"
          f"  {[r.question_id for r in gate]}")
    print()
    print(f"addressable ceiling for one retrieval change = R/{total} = {r_count}/{total}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Label Week 4 failures R/G/Not-In-Corpus.")
    parser.add_argument("--docs", type=Path, default=DEFAULT_DOCS)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN_JSONL)
    parser.add_argument("--hybrid", action="store_true")
    parser.add_argument("--no-generate", action="store_true", help="retrieval only")
    parser.add_argument(
        "--model",
        default=None,
        help="Groq model id; overrides GROQ_MODEL, which may name a retired model",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        records = label_all(
            args.golden,
            args.docs,
            args.index,
            hybrid=args.hybrid,
            generate=not args.no_generate,
            model=args.model,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"labelling failed: {exc}", file=sys.stderr)
        return 1

    print_report(records)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps([r.model_dump() for r in records], indent=2), encoding="utf-8"
        )
        print()
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
