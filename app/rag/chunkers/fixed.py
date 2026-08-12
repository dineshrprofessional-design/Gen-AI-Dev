"""Strategy A — the baseline: fixed-size windows with separator backoff.

This is the naive strategy everything else is measured against. It knows nothing
about markdown structure: it counts characters and backs off to the nearest
separator. That is exactly how a table row gets orphaned from its header and a
code fence gets cut in half.

**Do not improve it.** Its failures are the measurement.
"""

from app.rag.chunkers.base import make_chunk
from app.rag.models import Chunk, Document

SEPARATORS = ["\n\n", "\n", " ", ""]


class FixedSizeChunker:
    name = "fixed"

    def __init__(self, chunk_size: int = 1000, overlap: int = 150) -> None:
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def _window_end(self, text: str, start: int) -> int:
        end = min(start + self.chunk_size, len(text))
        if end == len(text):
            return end

        # Back off to the last separator in the second half of the window.
        # Searching only the second half stops us emitting tiny chunks.
        floor = start + self.chunk_size // 2
        for separator in SEPARATORS:
            if not separator:
                break
            index = text.rfind(separator, floor, end)
            if index != -1:
                return index + len(separator)
        return end

    def chunk(self, doc: Document) -> list[Chunk]:
        text = doc.text
        chunks: list[Chunk] = []
        position, ord_ = 0, 0

        while position < len(text):
            end = self._window_end(text, position)
            body = text[position:end]

            if body.strip():
                chunks.append(
                    make_chunk(
                        doc,
                        text=body.strip(),
                        char_start=position,
                        char_end=end,
                        ord_=ord_,
                        strategy=self.name,
                    )
                )
                ord_ += 1

            if end >= len(text):
                break
            # Step forward by the window minus overlap, never backwards.
            position = max(end - self.overlap, position + 1)

        return chunks
