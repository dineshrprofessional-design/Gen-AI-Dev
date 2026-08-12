"""HTML extraction using the standard library only.

A dependency-free parser keeps the core install small. It handles the shape of
a rendered documentation page — headings, paragraphs, tables — and drops the
parts that are markup rather than content.
"""

from html.parser import HTMLParser
from pathlib import Path

from app.rag.extractors.base import Extracted

_DROP = {"script", "style", "noscript", "head"}
_HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_BLOCK = {
    "p", "div", "section", "article", "br", "li", "tr", "pre", "table",
} | _HEADINGS


class _Reader(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title: str | None = None
        self._first_heading: str | None = None
        self._suppress = 0
        self._capture: str | None = None
        self._buffer: list[str] = []
        # Track header rows so we can emit a markdown separator beneath them.
        self._row_cells = 0
        self._row_is_header = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in _DROP:
            self._suppress += 1
            return
        if tag in ("title", "h1") and self._capture is None:
            self._capture = tag
            self._buffer = []
        if tag == "tr":
            self._row_cells = 0
            self._row_is_header = False
        if tag in _BLOCK:
            self.parts.append("\n")
        # Emit headings as markdown so the structure-aware chunker can see
        # them. An <h2> that arrives as plain text is a section boundary the
        # chunker will never find — the structure was in the source and
        # ingestion is where it gets lost.
        if tag in _HEADINGS:
            self.parts.append("#" * int(tag[1]) + " ")
        if tag in ("td", "th"):
            self._row_cells += 1
            if tag == "th":
                self._row_is_header = True
            self.parts.append(" | ")

    def handle_endtag(self, tag: str) -> None:
        if tag in _DROP:
            self._suppress = max(0, self._suppress - 1)
            return
        if self._capture == tag:
            text = "".join(self._buffer).strip()
            if tag == "title":
                self.title = text
            elif self._first_heading is None:
                self._first_heading = text
            self._capture = None
        if tag in _BLOCK:
            self.parts.append("\n")
        # A `<th>` row becomes a markdown header row, so follow it with the
        # `|---|---|` separator that makes the table valid markdown.
        if tag == "tr" and self._row_is_header and self._row_cells:
            self.parts.append("\n" + " | ".join("---" for _ in range(self._row_cells)))
            self.parts.append("\n")
            self._row_is_header = False

    def handle_data(self, data: str) -> None:
        if self._suppress:
            return
        if self._capture is not None:
            self._buffer.append(data)
        self.parts.append(data)

    def text(self) -> str:
        joined = "".join(self.parts)
        lines = [" ".join(line.split()) for line in joined.splitlines()]
        return "\n".join(line for line in lines if line)

    def best_title(self) -> str | None:
        return self.title or self._first_heading


class HtmlExtractor:
    name = "html"
    extensions = (".html", ".htm")

    def extract(self, path: Path) -> Extracted:
        reader = _Reader()
        reader.feed(path.read_text(encoding="utf-8"))
        reader.close()
        return Extracted(text=reader.text(), title_hint=reader.best_title())
