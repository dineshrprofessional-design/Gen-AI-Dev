"""The gold question set, and the check that keeps it honest.

Eight questions whose answers are already known and can be pointed at by page
and section. Later phases will run these against retrieval and count how often
the right section comes back in the top 5.

That number is only meaningful if every question genuinely points at the right
place, so `verify()` resolves each gold target against the real corpus and
confirms the answer text is actually there. A question that points at the wrong
section would silently deflate the hit-rate and there would be no way to tell.
"""

from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from app.rag.chunkers.structure import StructureAwareChunker
from app.rag.models import Chunk, Document

DEFAULT_QUESTIONS = Path("eval/questions.yaml")


class DependsOn(str, Enum):
    """What kind of structure the answer sits in.

    The task requires at least three questions to depend on a table row or a
    code fence — those are exactly the ones a structure-blind chunker breaks.
    """

    table_row = "table_row"
    code_fence = "code_fence"
    prose = "prose"

    @property
    def is_structural(self) -> bool:
        return self in (DependsOn.table_row, DependsOn.code_fence)


class GoldTarget(BaseModel):
    """Where the answer lives. This is what retrieval has to find."""

    sdk_version: str = Field(min_length=1)
    page_id: str = Field(min_length=1)
    anchor: str = Field(min_length=1)

    def __str__(self) -> str:
        return f"{self.sdk_version}/{self.page_id}#{self.anchor}"


class Question(BaseModel):
    id: str
    question: str
    answer: str
    depends_on: DependsOn
    gold: GoldTarget
    must_contain: list[str] = Field(default_factory=list)


class QuestionSet(BaseModel):
    questions: list[Question]

    def structural_count(self) -> int:
        return sum(q.depends_on.is_structural for q in self.questions)


class VerifyResult(BaseModel):
    question_id: str
    gold: str
    found_section: bool
    missing_strings: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.found_section and not self.missing_strings

    @property
    def reason(self) -> str:
        if not self.found_section:
            return "gold section does not exist in the corpus"
        if self.missing_strings:
            return f"section is missing: {', '.join(self.missing_strings)}"
        return ""


def load_questions(path: Path = DEFAULT_QUESTIONS) -> QuestionSet:
    if not path.is_file():
        raise FileNotFoundError(f"question set not found: {path}")
    return QuestionSet(**yaml.safe_load(path.read_text(encoding="utf-8")))


def sections_by_anchor(documents: list[Document]) -> dict[str, Chunk]:
    """Index the corpus by `version/page#anchor`.

    Sections come from the structure-aware chunker rather than a separate
    parser, so a gold target is expressed in exactly the units retrieval will
    later return.
    """
    chunker = StructureAwareChunker()
    index: dict[str, Chunk] = {}
    for doc in documents:
        for chunk in chunker.chunk(doc):
            if chunk.anchor:
                index[f"{doc.sdk_version}/{doc.page_id}#{chunk.anchor}"] = chunk
    return index


def verify(question_set: QuestionSet, documents: list[Document]) -> list[VerifyResult]:
    """Check every gold target resolves and actually contains its answer."""
    index = sections_by_anchor(documents)
    results: list[VerifyResult] = []

    for question in question_set.questions:
        key = str(question.gold)
        section = index.get(key)
        if section is None:
            results.append(
                VerifyResult(question_id=question.id, gold=key, found_section=False)
            )
            continue

        haystack = section.text.lower()
        missing = [s for s in question.must_contain if s.lower() not in haystack]
        results.append(
            VerifyResult(
                question_id=question.id,
                gold=key,
                found_section=True,
                missing_strings=missing,
            )
        )

    return results
