"""One JSON line per request: the trace file Week 5 reads.

Every /api/v1/chat/ask request appends a record to `traces/traces.jsonl`. The
schema exists to make a trace REPLAYABLE from its own fields alone, which is
Week 5's first requirement — so it carries everything a replay needs and
nothing that only looks useful:

- the question exactly as asked, and the full retrieval config
  (strategy / hybrid / k / version filter)
- the retrieved chunk_ids WITH scores and, under fusion, the per-arm ranks
- the model id, its parameters, and the prompt version
- the raw output, refusal state, and citations
- timing, and the git SHA the app was running at

`prompt_version` is a content hash of generate.SYSTEM_PROMPT rather than a
hand-maintained number: a version constant would drift the first time someone
edited the prompt and forgot to bump it, and a drifted version field is worse
than none because replays would trust it.

Tracing is append-only and must never break the request that produced it: a
failure to write a trace is swallowed after a stderr note, because losing one
line of telemetry is better than failing a user's question.
"""

import hashlib
import json
import subprocess
import sys
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

TRACE_DIR = Path("traces")
TRACE_FILE = TRACE_DIR / "traces.jsonl"


@lru_cache(maxsize=1)
def prompt_version() -> str:
    from app.rag.generate import SYSTEM_PROMPT

    return hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:12]


@lru_cache(maxsize=1)
def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001 - tracing must never break a request
        return "unknown"


def new_trace_id() -> str:
    return uuid.uuid4().hex[:12]


def build_trace(
    *,
    trace_id: str,
    question: str,
    strategy: str,
    hybrid: bool,
    k: int,
    sdk_version_filter: str | None,
    hits: list,
    model: str,
    raw_output: str | None,
    refused: bool,
    refusal_reason: str,
    citations: list[str],
    invalid_citations: list[str],
    retrieval_ms: float,
    generation_ran: bool,
    error: str = "",
) -> dict[str, Any]:
    return {
        "trace_id": trace_id,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "question": question,
        "prompt_version": prompt_version(),
        "git_sha": git_sha(),
        "retrieval": {
            "strategy": strategy,
            "hybrid": hybrid,
            "k": k,
            "sdk_version_filter": sdk_version_filter,
        },
        "retrieved": [
            {
                "rank": rank,
                "chunk_id": hit.chunk_id,
                "score": round(hit.score, 4),
                "rrf_score": hit.metadata.get("rrf_score"),
                "dense_rank": hit.metadata.get("dense_rank"),
                "bm25_rank": hit.metadata.get("bm25_rank"),
            }
            for rank, hit in enumerate(hits, start=1)
        ],
        "model": model,
        "params": {"temperature": 0},
        "generation_ran": generation_ran,
        "raw_output": raw_output,
        "refused": refused,
        "refusal_reason": refusal_reason,
        "citations": citations,
        "invalid_citations": invalid_citations,
        "retrieval_ms": round(retrieval_ms, 1),
        "error": error,
    }


def append(record: dict[str, Any], path: Path = TRACE_FILE) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:  # noqa: BLE001 - see module docstring
        print(f"trace write failed: {exc}", file=sys.stderr)


def load_all(path: Path = TRACE_FILE) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def by_id(trace_id: str, path: Path = TRACE_FILE) -> dict[str, Any]:
    for record in load_all(path):
        if record["trace_id"] == trace_id:
            return record
    raise KeyError(f"trace {trace_id} not found in {path}")
