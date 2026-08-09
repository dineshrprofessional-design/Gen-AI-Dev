from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app.core.config import Settings, get_settings
from app.rag.loader import IngestError
from app.rag.models import Document, IngestReport
from app.rag.service import IngestService, get_ingest_service

router = APIRouter(prefix="/ingest", tags=["ingest"])


class IngestRequest(BaseModel):
    docs_root: str | None = None
    sdk_version: str | None = None


class ReportSummary(BaseModel):
    report: IngestReport
    counts: dict[str, int]


def _summary(report: IngestReport) -> ReportSummary:
    return ReportSummary(report=report, counts=report.counts())


@router.post("/run", response_model=ReportSummary)
def run_ingest(
    payload: IngestRequest | None = None,
    settings: Settings = Depends(get_settings),
    service: IngestService = Depends(get_ingest_service),
) -> ReportSummary:
    payload = payload or IngestRequest()
    root = Path(payload.docs_root or settings.docs_root)
    try:
        report = service.run(root, payload.sdk_version)
    except IngestError as exc:
        # Fatal only — a corpus that cannot be opened at all. Per-file problems
        # come back inside the report as failures.
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    return _summary(report)


@router.get("/report", response_model=ReportSummary)
def get_report(
    service: IngestService = Depends(get_ingest_service),
) -> ReportSummary:
    if service.report is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no ingest has been run yet")
    return _summary(service.report)


@router.get("/documents", response_model=list[Document])
def list_documents(
    sdk_version: str | None = None,
    page_type: str | None = None,
    service: IngestService = Depends(get_ingest_service),
) -> list[Document]:
    return service.documents(sdk_version, page_type)


@router.get("/documents/{sdk_version}/{page_id}", response_model=Document)
def get_document(
    sdk_version: str,
    page_id: str,
    service: IngestService = Depends(get_ingest_service),
) -> Document:
    document = service.find(sdk_version, page_id)
    if document is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"no document {sdk_version}/{page_id}"
        )
    return document
