"""Strategy B — structure-aware, no overlap.

Splits on markdown headings, and never splits a parameter row from its header
row or a code fence across chunks. It gets that almost for free: the block
parser already made tables and code fences indivisible, so this chunker only
ever places *whole blocks*.

**Why no overlap.** Overlap is a patch for not knowing where the boundaries
are — a fixed-size cutter repeats text so a severed idea survives on one side.
B cuts at headings, which are real boundaries, so overlap would only duplicate
content and let one section's words pollute another's match. B carries the
heading breadcrumb instead: cheaper than repetition, and more precise, because
a chunk holding nothing but a parameter table still says which method it
belongs to.
"""

from app.rag.chunkers.base import make_chunk
from app.rag.markdown import Block, BlockKind, parse_blocks, slugify
from app.rag.models import Chunk, Document

SPLIT_LEVEL = 3  # start a new chunk at #, ## or ###


class StructureAwareChunker:
    name = "structure"

    def __init__(self, max_chars: int = 1200) -> None:
        self.max_chars = max_chars

    def chunk(self, doc: Document) -> list[Chunk]:
        blocks = parse_blocks(doc.text)
        chunks: list[Chunk] = []
        ord_ = 0

        heading_stack: list[str] = []
        current: list[Block] = []
        # Captured when the chunk opens, so flushing mid-section doesn't label
        # a chunk with a heading it never contained.
        current_path: list[str] = []

        def flush() -> None:
            nonlocal ord_, current
            if not current:
                return
            start, end = current[0].start, current[-1].end
            body = doc.text[start:end].strip()
            if body:
                breadcrumb = " > ".join(current_path)
                text = f"{breadcrumb}\n\n{body}" if breadcrumb else body
                chunks.append(
                    make_chunk(
                        doc,
                        text=text,
                        char_start=start,
                        char_end=end,
                        ord_=ord_,
                        strategy=self.name,
                        heading_path=breadcrumb,
                        anchor=slugify(current_path[-1]) if current_path else "",
                    )
                )
                ord_ += 1
            current = []

        for block in blocks:
            if block.kind is BlockKind.heading and block.level <= SPLIT_LEVEL:
                flush()
                title = block.text.lstrip("#").strip()
                del heading_stack[block.level - 1 :]
                heading_stack.append(title)
                current_path = list(heading_stack)
                current = [block]
                continue

            projected = (block.end - current[0].start) if current else len(block.text)
            # A heading must never be flushed alone: that emits a chunk with no
            # content which still matches on the heading's words, and it will
            # outrank the chunk that actually holds the answer.
            has_content = any(b.kind is not BlockKind.heading for b in current)

            if current and has_content and projected > self.max_chars:
                # Over budget. Flush and start fresh — an atomic block that is
                # itself oversized becomes its own chunk rather than being cut.
                # Correctness beats the size limit.
                flush()
                current_path = list(heading_stack)

            current.append(block)

        flush()
        return chunks
