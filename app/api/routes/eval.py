"""Serve the recorded Week 4 measurement runs to the UI.

    GET /api/v1/eval/summary

Read-only. This route reports numbers that were measured offline and written to
`report/w4/`; it never re-runs an evaluation. That is deliberate on two counts:
a live run is 120 queries and about a minute of CPU, and — more importantly —
the graded before/after comparison is a recorded artifact. A button that
recomputes it would quietly produce a *different* pair of numbers from the ones
in results.md.

The replication runs are included on purpose. The headline p50 fell by 10.8 ms
after the change, and two runs of the *same* configuration differ by more than
that, so the UI needs both pairs to show the delta is noise rather than a
speed-up.
"""

import json
import statistics
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.rag.eval_week4 import Week4Run, verdict_for
from app.rag.fingerprint import assert_same

router = APIRouter(prefix="/eval", tags=["eval"])

REPORT_DIR = Path("report/w4")
STAGE_LABELS = {
    "embed_ms": "query embedding (bge-m3)",
    "dense_ms": "dense vector search",
    "bm25_ms": "BM25",
    "fuse_ms": "RRF fusion",
}
# The two stages the Week 4 change actually added. Everything else was already
# being paid before it.
ADDED_BY_CHANGE = ("bm25_ms", "fuse_ms")


class AccuracyRow(BaseModel):
    k: int
    dense: int
    hybrid: int
    total: int
    is_metric: bool = False


class LatencyStats(BaseModel):
    p50_ms: float
    p95_ms: float
    min_ms: float
    samples: int


class StageRow(BaseModel):
    stage: str
    ms: float
    added_by_change: bool


class ReplicationRow(BaseModel):
    run: str
    hybrid: bool
    hit_at_3: str
    p50_ms: float


class QuestionRow(BaseModel):
    id: str
    question: str
    exact_token: str | None
    before_rank: int | None
    after_rank: int | None
    verdict: str


class OneVariableCheck(BaseModel):
    fingerprint_diff: list[str]
    config_diff: list[str]
    passed: bool


class EvalSummary(BaseModel):
    available: bool
    reason: str = ""
    questions: int = 0
    accuracy: list[AccuracyRow] = Field(default_factory=list)
    latency: dict[str, LatencyStats] = Field(default_factory=dict)
    stages: list[StageRow] = Field(default_factory=list)
    added_ms: float = 0.0
    embed_share_pct: float = 0.0
    replication: list[ReplicationRow] = Field(default_factory=list)
    replication_spread_ms: float = 0.0
    one_variable: OneVariableCheck | None = None
    per_question: list[QuestionRow] = Field(default_factory=list)
    fingerprint: dict = Field(default_factory=dict)


def _load(name: str) -> Week4Run | None:
    path = REPORT_DIR / f"{name}.json"
    if not path.is_file():
        return None
    return Week4Run(**json.loads(path.read_text(encoding="utf-8")))


def build_summary() -> EvalSummary:
    before, after = _load("before"), _load("after")
    if before is None or after is None:
        return EvalSummary(
            available=False,
            reason=(
                "no recorded runs in report/w4. Produce them with "
                "`python -m app.rag.eval_week4 --out report/w4/before.json` and "
                "`--hybrid --out report/w4/after.json`."
            ),
        )

    total = len(before.outcomes)

    accuracy = [
        AccuracyRow(
            k=k, dense=before.hits(k), hybrid=after.hits(k), total=total, is_metric=k == 3
        )
        for k in (1, 3, 5)
    ]

    latency = {
        "dense": LatencyStats(
            p50_ms=round(before.per_question_p50_ms, 1),
            p95_ms=round(before.pooled_p95_ms, 1),
            min_ms=round(before.pooled_min_ms, 1),
            samples=sum(len(o.latencies_ms) for o in before.outcomes),
        ),
        "hybrid": LatencyStats(
            p50_ms=round(after.per_question_p50_ms, 1),
            p95_ms=round(after.pooled_p95_ms, 1),
            min_ms=round(after.pooled_min_ms, 1),
            samples=sum(len(o.latencies_ms) for o in after.outcomes),
        ),
    }

    # Stage split comes from the hybrid run: the dense path carries no
    # instrumentation, so this is a one-sided diagnostic, not a like-for-like
    # comparison. Labelled as such in the UI.
    parts: dict[str, list[float]] = {}
    for outcome in after.outcomes:
        for key, value in outcome.component_ms.items():
            parts.setdefault(key, []).append(value)

    stages = [
        StageRow(
            stage=STAGE_LABELS.get(key, key),
            ms=round(statistics.median(values), 3),
            added_by_change=key in ADDED_BY_CHANGE,
        )
        for key, values in parts.items()
        if key in STAGE_LABELS
    ]
    stages.sort(key=lambda s: -s.ms)
    total_stage_ms = sum(s.ms for s in stages) or 1.0
    added_ms = sum(s.ms for s in stages if s.added_by_change)
    embed_share = next((s.ms for s in stages if s.stage.startswith("query embedding")), 0.0)

    replication: list[ReplicationRow] = []
    same_config_p50: list[float] = []
    for name in ("before", "after", "before2", "after2"):
        run = _load(name)
        if run is None:
            continue
        replication.append(
            ReplicationRow(
                run=name,
                hybrid=run.config.hybrid,
                hit_at_3=run.hit_rate(3),
                p50_ms=round(run.per_question_p50_ms, 1),
            )
        )
        if not run.config.hybrid:
            same_config_p50.append(run.per_question_p50_ms)

    moved = assert_same(before.fingerprint, after.fingerprint)
    config_diff = [
        f
        for f in before.config.model_fields
        if getattr(before.config, f) != getattr(after.config, f)
    ]

    after_by_id = {o.question_id: o for o in after.outcomes}
    per_question = [
        QuestionRow(
            id=b.question_id,
            question=b.question,
            exact_token=b.exact_token,
            before_rank=b.hit_rank,
            after_rank=after_by_id[b.question_id].hit_rank,
            verdict=verdict_for(b, after_by_id[b.question_id]),
        )
        for b in before.outcomes
    ]

    return EvalSummary(
        available=True,
        questions=total,
        accuracy=accuracy,
        latency=latency,
        stages=stages,
        added_ms=round(added_ms, 3),
        embed_share_pct=round(embed_share / total_stage_ms * 100, 1),
        replication=replication,
        replication_spread_ms=(
            round(max(same_config_p50) - min(same_config_p50), 1)
            if len(same_config_p50) > 1
            else 0.0
        ),
        one_variable=OneVariableCheck(
            fingerprint_diff=moved,
            config_diff=config_diff,
            passed=not moved and config_diff == ["hybrid"],
        ),
        per_question=per_question,
        fingerprint=before.fingerprint.model_dump(),
    )


@router.get("/summary", response_model=EvalSummary)
def summary() -> EvalSummary:
    """Accuracy and latency for the recorded dense vs hybrid runs."""
    return build_summary()
