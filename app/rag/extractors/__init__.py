"""Extractor registry, keyed by file extension.

Adding a format is a new module plus one line here — never a refactor.
"""

from pathlib import Path

from app.rag.extractors.base import Extracted, ExtractError, Extractor
from app.rag.extractors.docx import DocxExtractor
from app.rag.extractors.html import HtmlExtractor
from app.rag.extractors.markdown import MarkdownExtractor
from app.rag.extractors.pdf import PdfExtractor
from app.rag.extractors.text import TextExtractor

EXTRACTORS: tuple[Extractor, ...] = (
    MarkdownExtractor(),
    TextExtractor(),
    HtmlExtractor(),
    PdfExtractor(),
    DocxExtractor(),
)

_BY_EXTENSION: dict[str, Extractor] = {
    extension: extractor
    for extractor in EXTRACTORS
    for extension in extractor.extensions
}

SUPPORTED_EXTENSIONS: tuple[str, ...] = tuple(sorted(_BY_EXTENSION))


def get_extractor(path: Path) -> Extractor | None:
    """The extractor for this file, or None if the format is unsupported.

    None means "record it as skipped" — never "pretend the file wasn't there".
    """
    return _BY_EXTENSION.get(path.suffix.lower())


__all__ = [
    "EXTRACTORS",
    "SUPPORTED_EXTENSIONS",
    "Extracted",
    "ExtractError",
    "Extractor",
    "get_extractor",
]
