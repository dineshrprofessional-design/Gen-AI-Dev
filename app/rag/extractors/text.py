from pathlib import Path

from app.rag.extractors.base import Extracted


class TextExtractor:
    name = "text"
    extensions = (".txt", ".rst")

    def extract(self, path: Path) -> Extracted:
        text = path.read_text(encoding="utf-8")
        # A plain text file declares nothing about itself; the first non-empty
        # line is the only title evidence available.
        first = next((ln.strip() for ln in text.splitlines() if ln.strip()), None)
        return Extracted(text=text, title_hint=first)
