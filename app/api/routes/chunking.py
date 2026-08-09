"""Chunk-preview API.

Upload one document, get back the chunks each strategy produces. This is a
preview endpoint — nothing is stored, nothing is indexed. It exists so you can
*see* where a chunker cuts.
"""

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import BaseModel

from app.rag.chunkers import CHUNKERS
from app.rag.chunkers.base import damage_flags
from app.rag.extractors import SUPPORTED_EXTENSIONS, ExtractError
from app.rag.loader import load_document
from app.rag.metadata import MetadataError
from app.rag.models import Chunk

router = APIRouter(prefix="/chunk", tags=["chunk"])

MAX_UPLOAD_BYTES = 5 * 1024 * 1024


class ChunkView(BaseModel):
    """A Chunk plus the damage the UI needs to highlight."""

    chunk_id: str
    text: str
    heading_path: str
    char_start: int
    char_end: int
    size: int
    has_table_row: bool
    has_code_fence: bool
    damage: list[str]

    @classmethod
    def of(cls, chunk: Chunk) -> "ChunkView":
        return cls(
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            heading_path=chunk.heading_path,
            char_start=chunk.char_start,
            char_end=chunk.char_end,
            size=len(chunk.text),
            has_table_row=chunk.has_table_row,
            has_code_fence=chunk.has_code_fence,
            damage=damage_flags(chunk.text),
        )


class StrategyResult(BaseModel):
    strategy: str
    chunks: list[ChunkView]
    count: int
    sizes: dict[str, int]
    damaged: int


class PreviewResponse(BaseModel):
    filename: str
    source_format: str
    title: str
    doc_chars: int
    results: list[StrategyResult]


@router.get("/strategies", response_model=list[str])
def list_strategies() -> list[str]:
    return sorted(CHUNKERS)


def _run(strategy: str, doc) -> StrategyResult:
    chunks = CHUNKERS[strategy]().chunk(doc)
    views = [ChunkView.of(c) for c in chunks]
    sizes = sorted(v.size for v in views) or [0]
    return StrategyResult(
        strategy=strategy,
        chunks=views,
        count=len(views),
        sizes={
            "min": sizes[0],
            "median": sizes[len(sizes) // 2],
            "max": sizes[-1],
        },
        damaged=sum(1 for v in views if v.damage),
    )


@router.post("/preview", response_model=PreviewResponse)
async def preview(file: UploadFile = File(...)) -> PreviewResponse:
    """Chunk an uploaded document with every registered strategy."""
    raw = await file.read()
    if not raw:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "empty file")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"file exceeds {MAX_UPLOAD_BYTES // 1024 // 1024} MB",
        )

    name = Path(file.filename or "upload.md").name
    if Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"unsupported file type. Supported: {', '.join(SUPPORTED_EXTENSIONS)}",
        )

    # Write to a temp dir so the existing loader can be reused unchanged —
    # metadata resolution reads the *path*, so the file needs to exist on disk.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        target = root / name
        target.write_bytes(raw)
        try:
            doc = load_document(target, root)
        except ExtractError as exc:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY, f"could not read: {exc}"
            ) from exc
        except MetadataError as exc:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY, f"bad metadata: {exc}"
            ) from exc

        return PreviewResponse(
            filename=name,
            source_format=doc.source_format,
            title=doc.title,
            doc_chars=len(doc.text),
            results=[_run(s, doc) for s in sorted(CHUNKERS)],
        )
