"""DOCX extraction.

Better than PDF: paragraphs and table cells are real objects, so rows can be
rebuilt as pipe-delimited lines instead of guessed from coordinates.
"""

from pathlib import Path

from app.rag.extractors.base import Extracted, ExtractError


class DocxExtractor:
    name = "docx"
    extensions = (".docx",)

    def extract(self, path: Path) -> Extracted:
        try:
            import docx
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ExtractError("python-docx is not installed") from exc

        try:
            document = docx.Document(str(path))
        except Exception as exc:
            raise ExtractError(f"unreadable DOCX: {exc}") from exc

        parts = [p.text.strip() for p in document.paragraphs if p.text.strip()]

        # Tables survive here, unlike in PDF — keep the row structure.
        for table in document.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells]
                if any(cells):
                    parts.append("| " + " | ".join(cells) + " |")

        text = "\n\n".join(parts)
        if not text.strip():
            raise ExtractError("DOCX contained no text")

        heading = next(
            (
                p.text.strip()
                for p in document.paragraphs
                if p.style.name.startswith("Heading") and p.text.strip()
            ),
            None,
        )
        return Extracted(text=text, title_hint=heading or (parts[0] if parts else None))
