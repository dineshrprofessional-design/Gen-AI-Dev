"""The shared execution contract for the Week 7 race.

    python -m app.rag.week7_contract --prompts     # every prompt, verbatim
    python -m app.rag.week7_contract --price       # the rate table and its provenance
    python -m app.rag.week7_contract --parse '...' # exercise the output parser
    python -m app.rag.week7_contract --validate    # self-checks, no network

The Week 7 rubric awards 20 points for the fixed workflow "genuinely doing the
same task: same inputs, same output contract, same tools, no loop hiding inside
it". That is won by making the two arms *structurally unable* to differ, not by
asserting they don't. So everything they share lives here and nowhere else:

    the question objects        load_questions()      -> the same objects, same file
    the model and temperature   ArmConfig             -> one frozen config
    the output contract         ANSWER_CONTRACT       -> byte-identical in both prompts
    the LLM call                chat_once()           -> the ONLY OpenAI call site
    tool dispatch               run_tool()            -> wraps week7_tools.dispatch
    parsing                     parse_task_output()   -> one parser
    scoring                     score_output()        -> one scorer, arm-blind
    token accounting            Meter                 -> per-call, cumulative

`week7_agent.py` and `week7_workflow.py` must import `chat_once` and `run_tool`
from here and must NOT import `openai`, `app.rag.generate`, or `app.rag.index`
directly. That restriction is mechanically checkable — see `imports_are_clean()`.

## The one variable that cannot be held still

One arm must be told it has tools; the other must be told its context is already
assembled. Their preambles therefore cannot be identical. `ANSWER_CONTRACT` — the
output format, the citation rule, the refusal rule — is byte-identical and its
sha is recorded, and both preambles are printed verbatim by `--prompts`. Claiming
zero prompt difference would be false; the honest move is to bound it and show it.

## What is deliberately NOT here

Budgets. Iteration caps, token ceilings, cost ceilings and wall-clock limits
belong to the agent loop and arrive in the next step. `Meter` exposes everything a
budget will need to read (per-call and cumulative tokens and cost), so budgets can
be layered on without reopening this file.
"""

import argparse
import hashlib
import inspect
import json
import re
import sys
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Literal

import yaml
from pydantic import BaseModel, Field

from app.rag.generate import REFUSAL_TOKEN
from app.rag.week6_assertions import chunk_texts, loose, normalise, prose_only, scan_citations
from app.rag.week7_questions import (
    DEFAULT_QUESTIONS_JSONL,
    Week7Question,
    load_questions,
)
from app.rag.week7_tools import (
    TOOL_NAMES,
    ToolCallRecord,
    ToolConfig,
    ToolContext,
    dispatch,
    tool_schemas,
)

DEFAULT_PRICES = Path("eval/week7_prices.yaml")

VERSION_TOKEN = re.compile(r"\bv([23])\b|\bversion\s*([23])\b", re.I)

__all__ = [
    "ArmConfig", "TaskOutput", "parse_task_output", "ANSWER_CONTRACT",
    "AGENT_PREAMBLE", "WORKFLOW_PREAMBLE", "system_prompt", "user_prompt",
    "ChatResult", "ToolCall", "chat_once", "run_tool", "new_tool_context",
    "Price", "load_price", "LapUsage", "Meter",
    "CheckResult", "Score", "score_output", "contract_shas",
    "classify_provider_error", "retry_after_seconds",
    "ERROR_RATE_LIMIT_TPM", "ERROR_QUOTA_TPD", "ERROR_TRANSIENT", "ERROR_OTHER",
]


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


class ArmConfig(BaseModel):
    """Everything that must be identical across both arms.

    Passed to both; `sha()` goes into each arm's fairness record, so "same
    model, same temperature, same retrieval" is provable rather than asserted.
    """

    model: str = ""              # "" resolves to GROQ_MODEL at call time
    temperature: float = 0.0
    tools: ToolConfig = Field(default_factory=ToolConfig)
    retries: int = 5
    max_tool_result_chars: int = 4000

    def resolved_model(self) -> str:
        if self.model:
            return self.model
        from app.rag.generate import _model

        return _model()

    def sha(self) -> str:
        payload = self.model_dump(mode="json")
        payload["model"] = self.resolved_model()
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()[:16]


# --------------------------------------------------------------------------
# the output contract
# --------------------------------------------------------------------------

ANSWER_CONTRACT = f"""\
You answer developer questions about an SDK and the HTTP API behind it, using \
ONLY what the tools return. You have no other knowledge of this SDK.

Reply in exactly this format, and nothing else:

ANSWER: <your answer, one short paragraph>
VERSION: <v2, v3, both, or none>
CITATIONS: <comma-separated chunk ids or operationIds you used, or none>

Rules:

1. Every factual claim must come from a tool result. Never state a parameter \
name, a default value, a path or a version you have not seen a tool return.

2. If the tools do not contain the answer, reply with exactly:
ANSWER: {REFUSAL_TOKEN}
VERSION: none
CITATIONS: none

3. Defaults differ between v2 and v3. When you state a value, say which version \
it belongs to. VERSION must name the version your ANSWER describes.

4. Cite what you actually used: chunk ids look like v3/client-send#parameters::1 \
and operationIds look like sendMessageV3. Do not cite anything a tool did not \
return to you.

5. Be brief. Answer the question asked, then stop.
"""

AGENT_PREAMBLE = """\

You have tools available. Call them as needed to gather what you need before \
answering. Some questions cannot be answered from a single tool call: if a \
result points you at something else — a replacement operation, another version — \
look that up too before you answer.
"""

WORKFLOW_PREAMBLE = """\

The tool results you need have already been gathered for you and appear below. \
Do not ask for more; answer from what is provided, or refuse if it is not there.
"""


def system_prompt(arm: Literal["agent", "workflow"]) -> str:
    """The shared contract plus the one deliberate per-arm difference."""
    if arm == "agent":
        return ANSWER_CONTRACT + AGENT_PREAMBLE
    if arm == "workflow":
        return ANSWER_CONTRACT + WORKFLOW_PREAMBLE
    raise ValueError(f"unknown arm {arm!r}")


def user_prompt(question: Week7Question, context: str = "") -> str:
    """The question verbatim, plus pre-assembled context for the workflow arm."""
    if context:
        return f"Question: {question.question}\n\nTool results:\n\n{context}\n"
    return f"Question: {question.question}\n"


class TaskOutput(BaseModel):
    answer: str = ""
    sdk_version: str = "none"
    citations: list[str] = Field(default_factory=list)
    refused: bool = False
    raw: str = ""
    parse_error: str = ""

    @property
    def parsed(self) -> bool:
        return not self.parse_error


_ANSWER_LINE = re.compile(r"^\s*ANSWER:\s*(.*)$", re.I | re.M)
_VERSION_LINE = re.compile(r"^\s*VERSION:\s*(\S+)", re.I | re.M)
_CITATIONS_LINE = re.compile(r"^\s*CITATIONS:\s*(.*)$", re.I | re.M)
_VALID_VERSIONS = {"v2", "v3", "both", "none"}


def parse_task_output(text: str) -> TaskOutput:
    """One parser, used by both arms. A parse failure is terminal for both.

    No retry, ever — a parse-retry in one arm and not the other is exactly the
    asymmetry the "same task" criterion is looking for.
    """
    raw = text or ""
    answer_match = _ANSWER_LINE.search(raw)
    if not answer_match:
        return TaskOutput(raw=raw, parse_error="no ANSWER: line")

    # ANSWER runs to the VERSION line, so a multi-line answer survives.
    start = answer_match.start(1)
    version_match = _VERSION_LINE.search(raw)
    end = version_match.start() if version_match else len(raw)
    answer = raw[start:end].strip() if end > start else answer_match.group(1).strip()

    version = (version_match.group(1).strip().lower() if version_match else "none")
    if version not in _VALID_VERSIONS:
        version = "none"

    citations: list[str] = []
    citation_match = _CITATIONS_LINE.search(raw)
    if citation_match:
        blob = citation_match.group(1).strip()
        if blob.lower() not in ("none", "n/a", ""):
            citations = [
                part.strip().strip("[]`")
                for part in re.split(r"[,\s]+", blob)
                if part.strip().strip("[]`")
            ]
    # Anything the model bracketed inline counts too — tolerant on purpose, for
    # the same reason week6_assertions.scan_citations is: Week 5 mode 1 was a
    # correct citation the pipeline could not see.
    for found in scan_citations(raw):
        if found not in citations:
            citations.append(found)

    refused = REFUSAL_TOKEN in normalise(answer).upper()
    return TaskOutput(
        answer=answer,
        sdk_version=version,
        citations=citations,
        refused=refused,
        raw=raw,
        parse_error="" if version_match else "no VERSION: line",
    )


# --------------------------------------------------------------------------
# the single LLM seam
# --------------------------------------------------------------------------


@dataclass
class ToolCall:
    id: str
    name: str
    arguments_json: str


@dataclass
class ChatResult:
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    usage_missing: bool = False
    latency_ms: float = 0.0
    rate_limit_wait_s: float = 0.0
    error: str = ""
    #: "" when ok, else one of ERROR_RATE_LIMIT_TPM / ERROR_QUOTA_TPD /
    #: ERROR_TRANSIENT / ERROR_OTHER.
    error_kind: str = ""
    #: What the provider said to wait, in seconds, when it said anything.
    retry_after_s: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.error

    @property
    def quota_exhausted(self) -> bool:
        return self.error_kind == ERROR_QUOTA_TPD


ChatFn = Callable[..., ChatResult]

_RETRY_AFTER = re.compile(r"try again in (?:(\d+)m)?([\d.]+)s")

# --------------------------------------------------------------------------
# provider error classification
# --------------------------------------------------------------------------
#
# Groq returns HTTP 429 for two completely different conditions, and the
# difference decides whether retrying is sensible or futile:
#
#   tokens per minute (TPM)  a throttle. Clears in seconds. Retry.
#   tokens per day   (TPD)   a quota. Clears in hours. Retrying is pointless,
#                            and the five bounded sleeps just burn four minutes
#                            before failing anyway.
#
# Both carry `code: rate_limit_exceeded`, so the code is useless for telling
# them apart; the window phrase in the message is the only signal. Classified
# here, once, so the race and the arms cannot disagree about what happened.

ERROR_RATE_LIMIT_TPM = "rate_limit_tpm"
ERROR_QUOTA_TPD = "quota_tpd"
ERROR_TRANSIENT = "transient"
ERROR_OTHER = "other"

_PER_DAY_MARKERS = ("per day", "(tpd)", "(rpd)", "tpd:", "rpd:")
_PER_MINUTE_MARKERS = ("per minute", "(tpm)", "(rpm)", "tpm:", "rpm:")
_TRANSIENT_MARKERS = (
    "apiconnectionerror", "apitimeouterror", "connection", "timed out",
    "timeout", "502", "503", "504", "temporarily unavailable",
)


def classify_provider_error(text: str) -> str:
    """Name the failure so the caller can decide whether retrying is rational.

    A 429 whose window cannot be identified is treated as TPM — the bounded
    retry is the safe default, and misreading a throttle as a quota would abort
    a race that would have recovered on its own.
    """
    if not text:
        return ""
    lowered = text.lower()
    throttled = "429" in lowered or "rate_limit" in lowered or "ratelimit" in lowered
    if throttled:
        if any(marker in lowered for marker in _PER_DAY_MARKERS):
            return ERROR_QUOTA_TPD
        if any(marker in lowered for marker in _PER_MINUTE_MARKERS):
            return ERROR_RATE_LIMIT_TPM
        return ERROR_RATE_LIMIT_TPM
    if any(marker in lowered for marker in _TRANSIENT_MARKERS):
        return ERROR_TRANSIENT
    return ERROR_OTHER


def retry_after_seconds(text: str) -> float | None:
    """The provider's own "try again in 9m11.664s", in seconds, if it said one."""
    found = _RETRY_AFTER.search(text or "")
    if not found:
        return None
    minutes = float(found.group(1) or 0.0)
    return minutes * 60.0 + float(found.group(2))


@lru_cache(maxsize=1)
def cached_client():
    """One client for the whole process.

    `generate._client()` builds a fresh `OpenAI()` on every call. Left alone that
    per-call construction is charged to whichever arm makes more calls — which
    would inflate the agent specifically, a confound pointing toward the expected
    conclusion. Cached here and primed during warm-up. `generate.py` is NOT
    edited: Weeks 5 and 6 measured its current behaviour.
    """
    from app.rag.generate import _client

    return _client()


def _estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """Crude 4-chars-per-token fallback. Never used in a reported figure.

    Only reached when the provider omits `usage`, and the call is flagged
    `usage_missing` so it shows up in the race CSV rather than silently
    becoming zero.
    """
    return max(1, len(json.dumps(messages)) // 4)


def chat_once(
    messages: list[dict[str, Any]],
    *,
    model: str,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str = "auto",
    temperature: float = 0.0,
    retries: int = 5,
) -> ChatResult:
    """The ONLY place either arm talks to the model.

    Retries transport failures; never retries a verdict. Groq's free tier caps
    this model at 8000 tokens/minute and a race call is ~2k tokens, so 429s are
    expected rather than exceptional. Time spent sleeping on a 429 is returned
    separately as `rate_limit_wait_s` so the caller can subtract it from latency
    and from any wall-clock budget: the throttle is a property of the account,
    not of either architecture.
    """
    result = ChatResult()
    started = time.perf_counter()
    for attempt in range(retries):
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "temperature": temperature,
                "messages": messages,
            }
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = tool_choice
            completion = cached_client().chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - one bad call must not lose the run
            text = str(exc)
            result.error = f"{type(exc).__name__}: {exc}"
            result.error_kind = classify_provider_error(text)
            # A per-DAY quota does not clear inside a bounded retry. The 45 s cap
            # below is right for a per-minute throttle and useless here: five
            # sleeps would burn four minutes and still fail. Stop immediately and
            # let the caller decide, rather than pretending this is transient.
            if result.error_kind == ERROR_QUOTA_TPD:
                result.retry_after_s = retry_after_seconds(text) or 0.0
                break
            if "rate_limit" not in text and "429" not in text:
                break
            wait = retry_after_seconds(text)
            wait = wait + 1.0 if wait is not None else 8.0 * (attempt + 1)
            wait = min(wait, 45.0)
            result.rate_limit_wait_s += wait
            time.sleep(wait)
            continue

        result.error = ""
        message = completion.choices[0].message
        result.content = message.content
        result.tool_calls = [
            ToolCall(id=call.id, name=call.function.name, arguments_json=call.function.arguments)
            for call in (message.tool_calls or [])
        ]
        usage = getattr(completion, "usage", None)
        if usage is None:
            result.usage_missing = True
            result.prompt_tokens = _estimate_tokens(messages)
            result.completion_tokens = max(1, len(result.content or "") // 4)
        else:
            result.prompt_tokens = usage.prompt_tokens or 0
            result.completion_tokens = usage.completion_tokens or 0
            details = getattr(usage, "prompt_tokens_details", None)
            result.cached_tokens = getattr(details, "cached_tokens", 0) or 0
        break

    result.latency_ms = (time.perf_counter() - started) * 1000
    return result


# --------------------------------------------------------------------------
# tool dispatch — both arms go through here
# --------------------------------------------------------------------------


def new_tool_context(cfg: ArmConfig) -> ToolContext:
    return ToolContext(cfg=cfg.tools)


def run_tool(ctx: ToolContext, call: ToolCall, cfg: ArmConfig) -> dict[str, Any]:
    """Dispatch one tool call and cap the payload, identically for both arms."""
    payload = dispatch(ctx, call.name, call.arguments_json)
    rendered = json.dumps(payload, ensure_ascii=False)
    if len(rendered) > cfg.max_tool_result_chars:
        return {
            "truncated": True,
            "note": f"result truncated at {cfg.max_tool_result_chars} chars",
            "content": rendered[: cfg.max_tool_result_chars],
        }
    return payload


# --------------------------------------------------------------------------
# token accounting
# --------------------------------------------------------------------------


class Price(BaseModel):
    model: str
    usd_per_1m_input: float | None = None
    usd_per_1m_cached_input: float | None = None
    usd_per_1m_output: float | None = None
    source: str = ""
    verified_on: str = ""

    @property
    def verified(self) -> bool:
        return bool(self.verified_on) and self.usd_per_1m_input is not None

    def cost(self, prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0) -> float:
        if not self.verified:
            return 0.0
        fresh = max(0, prompt_tokens - cached_tokens)
        cached_rate = (
            self.usd_per_1m_cached_input
            if self.usd_per_1m_cached_input is not None
            else self.usd_per_1m_input
        )
        return (
            fresh / 1e6 * (self.usd_per_1m_input or 0.0)
            + cached_tokens / 1e6 * (cached_rate or 0.0)
            + completion_tokens / 1e6 * (self.usd_per_1m_output or 0.0)
        )


def load_price(model: str, path: Path = DEFAULT_PRICES) -> Price:
    if not path.is_file():
        return Price(model=model, source=f"{path} not found")
    table = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entry = table.get(model)
    if not entry:
        return Price(model=model, source=f"{model} not listed in {path}")
    return Price(model=model, **{k: v for k, v in entry.items() if k != "note"})


class LapUsage(BaseModel):
    """One LLM call. The cumulative columns are the audit trail."""

    lap: int
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int = 0
    usage_missing: bool = False
    latency_ms: float = 0.0
    cumulative_prompt: int = 0
    cumulative_completion: int = 0
    cumulative_cost_usd: float = 0.0

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class Meter(BaseModel):
    """Sums EVERY call, not just the last one.

    The Week 7 brief names understating agent cost as a specific failure: the
    loop re-sends the whole message list every lap, so per-lap tokens must be
    summed or the agent's cost is understated by multiples. `laps` makes that
    visible — `prompt_tokens` grows lap over lap — and `check_consistency()`
    asserts the total exceeds any single lap on a multi-call run, so a later
    "optimisation" into reading only the final usage fails loudly.
    """

    price: Price
    laps: list[LapUsage] = Field(default_factory=list)

    @property
    def calls(self) -> int:
        return len(self.laps)

    @property
    def prompt_tokens(self) -> int:
        return sum(lap.prompt_tokens for lap in self.laps)

    @property
    def completion_tokens(self) -> int:
        return sum(lap.completion_tokens for lap in self.laps)

    @property
    def cached_tokens(self) -> int:
        return sum(lap.cached_tokens for lap in self.laps)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cost_usd(self) -> float:
        return self.price.cost(self.prompt_tokens, self.completion_tokens, self.cached_tokens)

    @property
    def usage_missing_calls(self) -> int:
        return sum(1 for lap in self.laps if lap.usage_missing)

    def add(self, result: ChatResult) -> LapUsage:
        lap = LapUsage(
            lap=len(self.laps) + 1,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            cached_tokens=result.cached_tokens,
            usage_missing=result.usage_missing,
            latency_ms=round(result.latency_ms, 3),
        )
        self.laps.append(lap)
        lap.cumulative_prompt = self.prompt_tokens
        lap.cumulative_completion = self.completion_tokens
        lap.cumulative_cost_usd = round(self.cost_usd, 8)
        return lap

    def check_consistency(self) -> list[str]:
        problems: list[str] = []
        summed = sum(lap.tokens for lap in self.laps)
        if summed != self.total_tokens:
            problems.append(f"lap sum {summed} != total_tokens {self.total_tokens}")
        if self.calls > 1:
            largest = max(lap.tokens for lap in self.laps)
            if self.total_tokens <= largest:
                problems.append(
                    f"total_tokens {self.total_tokens} <= largest lap {largest} across "
                    f"{self.calls} calls — per-lap tokens are not being summed"
                )
        return problems

    def render(self) -> str:
        lines = [
            f"{'lap':>4} {'prompt':>9} {'completion':>11} {'cached':>8} "
            f"{'cum prompt':>11} {'cum cost $':>11}"
        ]
        lines.append("-" * len(lines[0]))
        for lap in self.laps:
            lines.append(
                f"{lap.lap:>4} {lap.prompt_tokens:>9} {lap.completion_tokens:>11} "
                f"{lap.cached_tokens:>8} {lap.cumulative_prompt:>11} "
                f"{lap.cumulative_cost_usd:>11.6f}"
            )
        lines.append("-" * len(lines[0]))
        lines.append(
            f"{'sum':>4} {self.prompt_tokens:>9} {self.completion_tokens:>11} "
            f"{self.cached_tokens:>8} {self.total_tokens:>11} {self.cost_usd:>11.6f}"
        )
        return "\n".join(lines)


# --------------------------------------------------------------------------
# scoring — one scorer, blind to which arm produced the run
# --------------------------------------------------------------------------


class CheckResult(BaseModel):
    name: str
    ok: bool
    applicable: bool = True
    detail: str = ""
    evidence: list[str] = Field(default_factory=list)


class Score(BaseModel):
    question_id: str
    passed: bool
    checks: list[CheckResult] = Field(default_factory=list)

    @property
    def failed(self) -> list[CheckResult]:
        return [c for c in self.checks if c.applicable and not c.ok]


def score_output(
    question: Week7Question,
    output: TaskOutput,
    records: list[ToolCallRecord],
    texts: dict[str, str] | None = None,
) -> Score:
    """Deterministic pass/fail. No model, no network, no knowledge of the arm.

    Takes only the question, the parsed output and the tool records, so the same
    triple scores identically whichever arm produced it.
    """
    texts = texts if texts is not None else chunk_texts(question_docs_root(question))
    checks: list[CheckResult] = []
    body = loose(normalise(output.answer))

    if output.parse_error:
        checks.append(
            CheckResult(name="parsed", ok=False, detail=output.parse_error)
        )
        return Score(question_id=question.id, passed=False, checks=checks)
    checks.append(CheckResult(name="parsed", ok=True))

    # 1. refusal matches expectation
    checks.append(
        CheckResult(
            name="refusal_matches_expectation",
            ok=output.refused == question.expect_refusal,
            detail=f"refused={output.refused}, expected={question.expect_refusal}",
        )
    )

    answering = not question.expect_refusal

    # 2. must_contain
    missing = [lit for lit in question.must_contain if loose(normalise(lit)) not in body]
    checks.append(
        CheckResult(
            name="must_contain",
            ok=not missing,
            applicable=answering,
            detail=f"{len(question.must_contain) - len(missing)}/{len(question.must_contain)} present",
            evidence=missing,
        )
    )

    # 3. must_not_contain — the wrong-version discriminator
    present = [lit for lit in question.must_not_contain if loose(normalise(lit)) in body]
    checks.append(
        CheckResult(
            name="must_not_contain",
            ok=not present,
            detail="forbidden literal present" if present else "clean",
            evidence=present,
        )
    )

    # 4. citations resolve AND were actually returned by this run's tools
    returned = {cid for record in records for cid in record.result_chunk_ids}
    unresolved = [c for c in output.citations if "::" in c and c not in texts]
    unreturned = [c for c in output.citations if "::" in c and c in texts and c not in returned]
    checks.append(
        CheckResult(
            name="citations_grounded",
            ok=not unresolved and not unreturned,
            applicable=answering and bool(output.citations),
            detail=(
                f"{len(unresolved)} not in corpus, {len(unreturned)} not returned by a tool"
            ),
            evidence=unresolved + unreturned,
        )
    )

    # 5. version named when the answer states a version-divergent value
    version_named = bool(VERSION_TOKEN.search(prose_only(output.answer)))
    checks.append(
        CheckResult(
            name="version_named",
            ok=version_named and output.sdk_version == question.sdk_version,
            applicable=answering and question.sdk_version in ("v2", "v3", "both"),
            detail=f"VERSION={output.sdk_version}, expected {question.sdk_version}",
        )
    )

    return Score(
        question_id=question.id,
        passed=all(c.ok for c in checks if c.applicable),
        checks=checks,
    )


def question_docs_root(question: Week7Question) -> Path:
    from app.rag.index import DEFAULT_DOCS

    return DEFAULT_DOCS


# --------------------------------------------------------------------------
# fairness shas + import hygiene
# --------------------------------------------------------------------------


def contract_shas() -> dict[str, str]:
    """Everything a fairness record needs to prove the arms shared a contract."""

    def sha(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    return {
        "answer_contract": sha(ANSWER_CONTRACT),
        "agent_preamble": sha(AGENT_PREAMBLE),
        "workflow_preamble": sha(WORKFLOW_PREAMBLE),
        "parser": sha(inspect.getsource(parse_task_output)),
        "scorer": sha(inspect.getsource(score_output)),
        "chat": sha(inspect.getsource(chat_once)),
        "tool_schemas": sha(json.dumps(tool_schemas(), sort_keys=True)),
    }


FORBIDDEN_ARM_IMPORTS = ("openai", "app.rag.generate", "app.rag.index")


def imports_are_clean(module_path: Path) -> list[str]:
    """An arm must reach the outside world only through this contract."""
    if not module_path.is_file():
        return []
    source = module_path.read_text(encoding="utf-8")
    problems: list[str] = []
    for banned in FORBIDDEN_ARM_IMPORTS:
        if re.search(rf"^\s*(from|import)\s+{re.escape(banned)}\b", source, re.M):
            problems.append(
                f"{module_path.name} imports {banned} directly; it must go through "
                "week7_contract so both arms share one path"
            )
    return problems


def validate() -> tuple[list[str], list[str]]:
    failures: list[str] = []
    warnings: list[str] = []

    # the shared contract really is shared
    if not system_prompt("agent").startswith(ANSWER_CONTRACT):
        failures.append("agent prompt does not start with the shared contract")
    if not system_prompt("workflow").startswith(ANSWER_CONTRACT):
        failures.append("workflow prompt does not start with the shared contract")
    if AGENT_PREAMBLE == WORKFLOW_PREAMBLE:
        warnings.append("the two preambles are identical; expected one deliberate delta")

    # parser round-trips
    good = "ANSWER: The default is 250 ms.\nVERSION: v3\nCITATIONS: v3/client-send#parameters::1"
    parsed = parse_task_output(good)
    if parsed.parse_error or parsed.sdk_version != "v3":
        failures.append(f"parser failed on a well-formed reply: {parsed!r}")
    if parsed.citations != ["v3/client-send#parameters::1"]:
        failures.append(f"parser lost the citation: {parsed.citations}")
    refusal = parse_task_output(f"ANSWER: {REFUSAL_TOKEN}\nVERSION: none\nCITATIONS: none")
    if not refusal.refused:
        failures.append("parser did not detect a refusal")
    broken = parse_task_output("I think the answer is 250ms.")
    if broken.parsed:
        failures.append("parser accepted a reply with no ANSWER line")

    # meter sums every lap
    price = load_price("openai/gpt-oss-120b")
    meter = Meter(price=price)
    for prompt in (1000, 1800, 2600):
        meter.add(ChatResult(prompt_tokens=prompt, completion_tokens=100))
    if meter.total_tokens != 1000 + 1800 + 2600 + 300:
        failures.append(f"meter did not sum every lap: {meter.total_tokens}")
    failures.extend(f"meter: {p}" for p in meter.check_consistency())
    if not price.verified:
        warnings.append(
            f"price for {price.model} is unverified ({price.source}); cost columns "
            "will be suppressed"
        )

    # arms, if they exist yet, must not bypass the contract
    for name in ("week7_agent.py", "week7_workflow.py"):
        failures.extend(imports_are_clean(Path("app/rag") / name))

    # the questions load and are the same objects both arms will get
    try:
        questions = load_questions(DEFAULT_QUESTIONS_JSONL)
        if len(questions) != 10:
            failures.append(f"expected 10 questions, loaded {len(questions)}")
    except FileNotFoundError as exc:
        failures.append(str(exc))

    return failures, warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="The shared Week 7 execution contract.")
    parser.add_argument("--prompts", action="store_true", help="every prompt, verbatim")
    parser.add_argument("--price", action="store_true", help="the rate table")
    parser.add_argument("--parse", metavar="TEXT", default=None)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)

    if args.prompts:
        cfg = ArmConfig()
        print(f"model: {cfg.resolved_model()}   temperature: {cfg.temperature}")
        print(f"tools: {list(TOOL_NAMES)}")
        print()
        for key, value in contract_shas().items():
            print(f"  sha {key:18} {value}")
        print()
        print("=" * 74)
        print("ANSWER_CONTRACT — byte-identical in both arms")
        print("=" * 74)
        print(ANSWER_CONTRACT)
        for arm, preamble in (("AGENT", AGENT_PREAMBLE), ("WORKFLOW", WORKFLOW_PREAMBLE)):
            print("=" * 74)
            print(f"{arm}_PREAMBLE — the one deliberate difference")
            print("=" * 74)
            print(preamble)
        return 0

    if args.price:
        cfg = ArmConfig()
        price = load_price(cfg.resolved_model())
        print(json.dumps(price.model_dump(), indent=2))
        print(f"\nverified: {price.verified}")
        print(f"1k prompt + 200 completion tokens costs "
              f"${price.cost(1000, 200):.6f}")
        return 0

    if args.parse is not None:
        print(json.dumps(parse_task_output(args.parse).model_dump(), indent=2))
        return 0

    if args.validate:
        failures, warnings = validate()
        for warning in warnings:
            print(f"WARN  {warning}")
        for failure in failures:
            print(f"FAIL  {failure}")
        print()
        if failures:
            print(f"VALIDATION FAILED — {len(failures)} problem(s)")
            return 1
        print(f"VALIDATION PASSED — {len(warnings)} warning(s)")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
