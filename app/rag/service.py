"""Holds the most recent ingest report for the API to serve.

In-process and deliberately simple — ingestion state is not persisted in this
phase. A restart means "run it again", which is the honest behaviour while
there is no index to be stale.
"""

from pathlib import Path

from app.rag.loader import load_documents
from app.rag.models import Document, IngestReport


class IngestService:
    def __init__(self) -> None:
        self._report: IngestReport | None = None

    def run(self, docs_root: Path, sdk_version: str | None = None) -> IngestReport:
        self._report = load_documents(docs_root, sdk_version)
        return self._report

    @property
    def report(self) -> IngestReport | None:
        return self._report

    def documents(
        self, sdk_version: str | None = None, page_type: str | None = None
    ) -> list[Document]:
        if self._report is None:
            return []
        docs = self._report.documents
        if sdk_version:
            docs = [d for d in docs if d.sdk_version == sdk_version]
        if page_type:
            docs = [d for d in docs if d.page_type.value == page_type]
        return docs

    def find(self, sdk_version: str, page_id: str) -> Document | None:
        return next(
            (
                d
                for d in self.documents()
                if d.sdk_version == sdk_version and d.page_id == page_id
            ),
            None,
        )

    def reset(self) -> None:
        self._report = None


_service = IngestService()


def get_ingest_service() -> IngestService:
    """FastAPI dependency. Single instance for the process lifetime."""
    return _service
