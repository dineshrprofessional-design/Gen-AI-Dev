"""Ask-a-question API.

Retrieval always works. Generation works when XAI_API_KEY is set, and reports
itself unconfigured otherwise rather than failing obscurely.
"""

from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.rag.chunkers import CHUNKERS
from app.rag.generate import (
    DEFAULT_MODEL,
    GenerationUnavailable,
    answer,
    is_configured,
    list_models,
)
from app.rag.index import DEFAULT_INDEX, search

router = APIRouter(prefix="/chat", tags=["chat"])


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    strategy: str = "structure"
    k: int = Field(default=5, ge=1, le=20)
    sdk_version: str | None = None
    generate: bool = True
    model: str = DEFAULT_MODEL


class RetrievedChunk(BaseModel):
    rank: int
    chunk_id: str
    score: float
    target: str
    heading_path: str
    sdk_version: str
    text: str


class AskResponse(BaseModel):
    question: str
    strategy: str
    sdk_version_filter: str | None
    chunks: list[RetrievedChunk]
    generation_available: bool
    answer: str | None = None
    refused: bool = False
    refusal_reason: str = ""
    citations: list[str] = Field(default_factory=list)
    invalid_citations: list[str] = Field(default_factory=list)
    model: str = ""
    error: str = ""


class StatusResponse(BaseModel):
    generation_available: bool
    default_model: str
    strategies: list[str]
    models: list[str] = Field(default_factory=list)
    note: str = ""


@router.get("/status", response_model=StatusResponse)
def chat_status() -> StatusResponse:
    """What this deployment can do right now."""
    available = is_configured()
    models: list[str] = []
    note = ""
    if available:
        try:
            models = list_models()
        except Exception as exc:
            note = f"key is set but listing models failed: {exc}"
    else:
        note = "XAI_API_KEY not set — retrieval works, generation is disabled"
    return StatusResponse(
        generation_available=available,
        default_model=DEFAULT_MODEL,
        strategies=sorted(CHUNKERS),
        models=models,
        note=note,
    )


@router.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    if payload.strategy not in CHUNKERS:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"unknown strategy {payload.strategy!r}. "
            f"Available: {', '.join(sorted(CHUNKERS))}",
        )

    where = {"sdk_version": payload.sdk_version} if payload.sdk_version else None
    try:
        hits = search(
            payload.question,
            strategy=payload.strategy,
            k=payload.k,
            persist_dir=Path(DEFAULT_INDEX),
            where=where,
        )
    except Exception as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"search failed — has the index been built? "
            f"Run `python -m app.rag.index --all`. ({exc})",
        ) from exc

    chunks = [
        RetrievedChunk(
            rank=rank,
            chunk_id=hit.chunk_id,
            score=round(hit.score, 4),
            target=hit.target,
            heading_path=str(hit.metadata.get("heading_path") or ""),
            sdk_version=str(hit.metadata.get("sdk_version") or ""),
            text=hit.text,
        )
        for rank, hit in enumerate(hits, start=1)
    ]

    response = AskResponse(
        question=payload.question,
        strategy=payload.strategy,
        sdk_version_filter=payload.sdk_version,
        chunks=chunks,
        generation_available=is_configured(),
    )

    if not payload.generate:
        return response

    try:
        result = answer(payload.question, hits, model=payload.model)
    except GenerationUnavailable as exc:
        response.error = str(exc)
        return response
    except Exception as exc:
        response.error = f"generation failed: {exc}"
        return response

    response.answer = result.text
    response.refused = result.refused
    response.refusal_reason = result.reason
    response.citations = result.citations
    response.invalid_citations = result.invalid_citations
    response.model = result.model
    return response
