import re
from typing import Protocol

from app.rag.models import Chunk, Document

# Detection patterns for the damage report. These are deliberately local rather
# than imported from the markdown parser: *measuring* whether a chunk got cut
# badly is a different job from *parsing* document structure, and the two must
# stay independent so a structure-blind chunker is still measurable.
_FENCE_LINE = re.compile(r"^\s{0,3}(?:`{3,}|~{3,})")
_TABLE_SEPARATOR = re.compile(r"^\s*\|?[\s:|-]*-[\s:|-]*\|?\s*$")


class Chunker(Protocol):
    """Swap one for another and nothing else in the pipeline changes."""

    name: str

    def chunk(self, doc: Document) -> list[Chunk]: ...


def count_fences(text: str) -> int:
    return sum(1 for line in text.splitlines() if _FENCE_LINE.match(line))


def has_balanced_fences(text: str) -> bool:
    """An odd fence count means a code block was cut in half."""
    return count_fences(text) % 2 == 0


def has_table_row(text: str) -> bool:
    """True if any line looks like a table data row (not the separator)."""
    return any(
        line.count("|") >= 2 and not _TABLE_SEPARATOR.match(line)
        for line in text.splitlines()
    )


def has_table_header(text: str) -> bool:
    """True if the `|---|---|` separator is present — i.e. a header came along."""
    return any(_TABLE_SEPARATOR.match(line) for line in text.splitlines())


def damage_flags(text: str) -> list[str]:
    """Human-readable structural damage in one chunk, for reports and the UI."""
    flags = []
    if not has_balanced_fences(text):
        flags.append("split code fence")
    if has_table_row(text) and not has_table_header(text):
        flags.append("orphaned table row")
    return flags


def make_chunk(
    doc: Document,
    *,
    text: str,
    char_start: int,
    char_end: int,
    ord_: int,
    strategy: str,
    heading_path: str = "",
    anchor: str = "",
) -> Chunk:
    """Build a Chunk, deriving its id and structure flags from the text.

    The id is readable on purpose: `v3/client-send#parameters::2` tells you the
    version, page, section and position without a lookup.
    """
    suffix = f"#{anchor}" if anchor else ""
    return Chunk(
        chunk_id=f"{doc.sdk_version}/{doc.page_id}{suffix}::{ord_}",
        text=text,
        source_file=doc.source_file,
        page_id=doc.page_id,
        sdk_version=doc.sdk_version,
        page_type=doc.page_type,
        heading_path=heading_path,
        anchor=anchor,
        char_start=char_start,
        char_end=char_end,
        strategy=strategy,
        chunk_ord=ord_,
        has_table_row=has_table_row(text),
        has_code_fence=count_fences(text) > 0,
    )
