"""HTML extraction using the standard library only.

A dependency-free parser keeps the core install small. It handles the shape of
a rendered documentation page — headings, paragraphs, tables — and drops the
parts that are markup rather than content.
"""

from html.parser import HTMLParser
from pathlib import Path

from app.rag.extractors.base import Extracted

_DROP = {"script", "style", "noscript", "head"}
_BLOCK = {
    "p", "div", "section", "article", "br", "li", "tr",
    "h1", "h2", "h3", "h4", "h5", "h6", "pre", "table",
}


class _Reader(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title: str | None = None
        self._first_heading: str | None = None
        self._suppress = 0
        self._capture: str | None = None
        self._buffer: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in _DROP:
            self._suppress += 1
            return
        if tag in ("title", "h1") and self._capture is None:
            self._capture = tag
            self._buffer = []
        if tag in _BLOCK:
            self.parts.append("\n")
        if tag in ("td", "th"):
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
