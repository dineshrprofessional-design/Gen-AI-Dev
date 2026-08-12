"""PDF extraction.

Note what this costs: PDF stores glyph positions, not structure. A parameter
table comes back as a run of words with the row and column boundaries gone, so
nothing downstream can put them back. Ingesting a PDF is lossy in a way that
markdown and HTML are not — worth knowing before relying on one.
"""

import logging
from pathlib import Path

from app.rag.extractors.base import Extracted, ExtractError

# pypdf logs malformed-file complaints at error level. We surface those as a
# quarantine reason instead, so silence the library's own channel.
logging.getLogger("pypdf").setLevel(logging.CRITICAL)


class PdfExtractor:
    name = "pdf"
    extensions = (".pdf",)

    def extract(self, path: Path) -> Extracted:
        # Lazy import: a missing optional package should quarantine this one
        # file, not break the whole ingest at import time.
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ExtractError("pypdf is not installed — cannot read PDF") from exc

        try:
            reader = PdfReader(str(path))
            pages = [page.extract_text() or "" for page in reader.pages]
        except Exception as exc:
            raise ExtractError(f"unreadable PDF: {exc}") from exc

        text = "\n\n".join(p.strip() for p in pages if p.strip())
        if not text.strip():
            raise ExtractError("PDF contained no extractable text (scanned image?)")

        first = next((ln.strip() for ln in text.splitlines() if ln.strip()), None)
        return Extracted(text=text, title_hint=first)
