"""Parse markdown into typed blocks.

This is the load-bearing file of strategy B.

Structure-awareness is a **parsing** problem, not a chunking problem. Once a
table and a code fence are each a single indivisible block, a chunker cannot
split one by accident — it can only decide where to put whole blocks.
"""

import re
from dataclasses import dataclass
from enum import Enum

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_FENCE = re.compile(r"^(\s{0,3})(`{3,}|~{3,})(.*)$")
_TABLE_SEPARATOR = re.compile(r"^\s*\|?[\s:|-]*-[\s:|-]*\|?\s*$")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_SLUG_STRIP = re.compile(r"[^a-z0-9\s-]")


class BlockKind(str, Enum):
    heading = "heading"
    paragraph = "paragraph"
    table = "table"
    code_fence = "code_fence"
    list = "list"


@dataclass
class Block:
    kind: BlockKind
    text: str
    start: int
    end: int
    level: int = 0  # heading level; 0 for everything else

    @property
    def is_atomic(self) -> bool:
        """Blocks that must never be split across chunks."""
        return self.kind in (BlockKind.table, BlockKind.code_fence)


def slugify(text: str) -> str:
    """GitHub-style anchor slug, so a chunk resolves to page + anchor."""
    slug = _SLUG_STRIP.sub("", text.strip().lower())
    return re.sub(r"[\s-]+", "-", slug).strip("-")


def _line_offsets(text: str) -> list[int]:
    offsets, position = [], 0
    for line in text.splitlines(keepends=True):
        offsets.append(position)
        position += len(line)
    offsets.append(position)
    return offsets


def _is_table_start(lines: list[str], i: int) -> bool:
    """A table needs a header row AND a separator row directly beneath it."""
    return (
        "|" in lines[i]
        and i + 1 < len(lines)
        and "|" in lines[i + 1]
        and _TABLE_SEPARATOR.match(lines[i + 1]) is not None
    )


def parse_blocks(text: str, base_offset: int = 0) -> list[Block]:
    """Split text into blocks. Offsets are absolute when base_offset is given."""
    lines = text.splitlines()
    offsets = _line_offsets(text)
    blocks: list[Block] = []
    i = 0

    def span(first: int, last: int) -> tuple[str, int, int]:
        start, end = offsets[first], offsets[last]
        return text[start:end].rstrip("\n"), base_offset + start, base_offset + end

    while i < len(lines):
        line = lines[i]

        if not line.strip():
            i += 1
            continue

        # Code fence — atomic. Swallows everything up to the closing fence,
        # including any '#' or '|' inside it, which is the entire point.
        fence = _FENCE.match(line)
        if fence:
            marker = fence.group(2)
            closing = re.compile(rf"^\s{{0,3}}{marker[0]}{{{len(marker)},}}\s*$")
            j = i + 1
            while j < len(lines) and not closing.match(lines[j]):
                j += 1
            j = min(j + 1, len(lines))  # include the closing fence if present
            body, start, end = span(i, j)
            blocks.append(Block(BlockKind.code_fence, body, start, end))
            i = j
            continue

        heading = _HEADING.match(line)
        if heading:
            body, start, end = span(i, i + 1)
            blocks.append(
                Block(BlockKind.heading, body, start, end, level=len(heading.group(1)))
            )
            i += 1
            continue

        # Table — atomic. Header row + separator + every data row stay together,
        # so a data row is never orphaned from the header naming its columns.
        if _is_table_start(lines, i):
            j = i + 2
            while j < len(lines) and "|" in lines[j] and lines[j].strip():
                j += 1
            body, start, end = span(i, j)
            blocks.append(Block(BlockKind.table, body, start, end))
            i = j
            continue

        kind = BlockKind.list if _LIST_ITEM.match(line) else BlockKind.paragraph
        j = i
        while j < len(lines) and lines[j].strip():
            if j > i and (
                _HEADING.match(lines[j])
                or _FENCE.match(lines[j])
                or _is_table_start(lines, j)
            ):
                break
            j += 1
        body, start, end = span(i, j)
        blocks.append(Block(kind, body, start, end))
        i = j

    return blocks
