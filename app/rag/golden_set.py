"""The Week 4 golden set: 12 questions, each tagged with a known-correct chunk_id.

    python -m app.rag.golden_set --out eval/golden_set.jsonl
    python -m app.rag.golden_set --check

`eval/golden_set.yaml` is the hand-authored source of truth. `golden_set.jsonl`
is a *generated* artifact — the deliverable the task checklist names — and the
`gold_chunk_id` on every row is **derived by re-running the chunker**, never
typed by hand.

That matters more than it looks. A golden set tagged with a chunk_id that does
not exist scores 0/12 and the run looks like a retrieval catastrophe instead of
a typo. Deriving the id from the (sdk_version, page_id, anchor) triple the
question already carries makes that failure mode unrepresentable.

The derivation is only valid while one section maps to exactly one chunk. That
is true of this corpus today — 38 headings, 38 chunks — but the structure-aware
chunker will flush a section into two chunks if it exceeds `max_chars`, so
`derive` raises rather than silently taking the first.
"""

import argparse
import json
import sys
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from app.rag.chunkers.structure import StructureAwareChunker
from app.rag.models import Document
from app.rag.questions import DependsOn, GoldTarget

DEFAULT_GOLDEN_YAML = Path("eval/golden_set.yaml")
DEFAULT_GOLDEN_JSONL = Path("eval/golden_set.jsonl")


class Week4Question(BaseModel):
    """One golden question. Mirrors `questions.Question` plus Week 4 fields."""

    id: str
    question: str
    answer: str
    depends_on: DependsOn
    gold: GoldTarget
    must_contain: list[str] = Field(default_factory=list)
    exact_token: str | None = None
    note: str = ""

    @property
    def is_exact_token(self) -> bool:
        return bool(self.exact_token)


class Week4Set(BaseModel):
    questions: list[Week4Question]

    @property
    def exact_token_count(self) -> int:
        return sum(q.is_exact_token for q in self.questions)


class GoldenRow(BaseModel):
    """A jsonl row: the question plus its derived chunk_id."""

    id: str
    question: str
    gold_chunk_id: str
    gold_target: str
    answer: str
    depends_on: str
    exact_token: str | None
    must_contain: list[str]
    strategy: str


def load_yaml(path: Path = DEFAULT_GOLDEN_YAML) -> Week4Set:
    if not path.is_file():
        raise FileNotFoundError(f"golden set not found: {path}")
    return Week4Set(**yaml.safe_load(path.read_text(encoding="utf-8")))


def chunk_ids_by_target(
    documents: list[Document], strategy: str = "structure"
) -> dict[str, list[str]]:
    """Map `version/page#anchor` to every chunk_id that section produced.

    A list, not a single id, so an over-long section that got split shows up as
    a length-2 value and fails loudly in `derive` instead of quietly resolving
    to whichever chunk happened to come first.
    """
    if strategy != "structure":
        raise ValueError(
            f"chunk_ids are strategy-scoped and only the structure chunker emits "
            f"anchors; got {strategy!r}"
        )
    chunker = StructureAwareChunker()
    index: dict[str, list[str]] = {}
    for doc in documents:
        for chunk in chunker.chunk(doc):
            if chunk.anchor:
                key = f"{doc.sdk_version}/{doc.page_id}#{chunk.anchor}"
                index.setdefault(key, []).append(chunk.chunk_id)
    return index


def derive(
    question_set: Week4Set, documents: list[Document], strategy: str = "structure"
) -> list[GoldenRow]:
    by_target = chunk_ids_by_target(documents, strategy)
    rows: list[GoldenRow] = []
    for question in question_set.questions:
        target = str(question.gold)
        ids = by_target.get(target, [])
        if not ids:
            raise ValueError(f"{question.id}: gold target {target} is not in the corpus")
        if len(ids) > 1:
            raise ValueError(
                f"{question.id}: target {target} maps to {len(ids)} chunks "
                f"({', '.join(ids)}). Pin one explicitly — the section outgrew "
                "max_chars and the one-section-one-chunk assumption no longer holds."
            )
        rows.append(
            GoldenRow(
                id=question.id,
                question=question.question,
                gold_chunk_id=ids[0],
                gold_target=target,
                answer=question.answer,
                depends_on=question.depends_on.value,
                exact_token=question.exact_token,
                must_contain=question.must_contain,
                strategy=strategy,
            )
        )
    return rows


def dump_jsonl(rows: list[GoldenRow]) -> str:
    return "".join(json.dumps(r.model_dump(), ensure_ascii=False) + "\n" for r in rows)


def load_jsonl(path: Path = DEFAULT_GOLDEN_JSONL) -> list[GoldenRow]:
    if not path.is_file():
        raise FileNotFoundError(f"golden set not found: {path}")
    text = path.read_text(encoding="utf-8")
    return [GoldenRow(**json.loads(line)) for line in text.splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    from app.rag.index import DEFAULT_DOCS
    from app.rag.loader import IngestError, load_documents

    parser = argparse.ArgumentParser(description="Derive the Week 4 golden set.")
    parser.add_argument("--docs", type=Path, default=DEFAULT_DOCS)
    parser.add_argument("--yaml", type=Path, default=DEFAULT_GOLDEN_YAML)
    parser.add_argument("--out", type=Path, default=DEFAULT_GOLDEN_JSONL)
    parser.add_argument(
        "--check",
        action="store_true",
        help="re-derive and diff against the committed jsonl instead of writing",
    )
    args = parser.parse_args(argv)

    try:
        question_set = load_yaml(args.yaml)
        documents = load_documents(args.docs).documents
        rows = derive(question_set, documents)
    except (FileNotFoundError, IngestError, ValueError) as exc:
        print(f"golden set failed: {exc}", file=sys.stderr)
        return 1

    rendered = dump_jsonl(rows)

    if args.check:
        if not args.out.is_file():
            print(f"{args.out} does not exist; run without --check", file=sys.stderr)
            return 1
        current = args.out.read_text(encoding="utf-8")
        if current != rendered:
            print(
                f"{args.out} is stale — a fresh derivation from {args.yaml} differs. "
                "Re-run without --check.",
                file=sys.stderr,
            )
            return 1
        print(f"{args.out} matches a fresh derivation ({len(rows)} questions)")
        return 0

    args.out.write_text(rendered, encoding="utf-8")
    print(f"wrote {args.out}  ({len(rows)} questions)")
    print(f"exact-token questions: {question_set.exact_token_count}/{len(rows)}")
    for row in rows:
        flag = "exact" if row.exact_token else "     "
        print(f"  {row.id}  {flag}  {row.gold_chunk_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
