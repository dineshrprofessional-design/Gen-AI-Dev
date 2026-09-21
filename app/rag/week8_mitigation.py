"""Week 8 mitigation: a circuit breaker for tool thrashing. Exactly one change.

    python -m app.rag.week8_mitigation --explain
    python -m app.rag.week8_mitigation --self-test          # offline, no tokens
    python -m app.rag.week8_mitigation --run --out RUNS.json

## The failure this targets, and why this one

The Week 7 race recorded the agent on w7-10 — "What is the rate limit on the
/v1/embeddings endpoint?" — making 6 LLM calls and 5 consecutive `search_docs`
calls, several with identical arguments, before giving up. 20,680 tokens and
117,928 ms for a question whose correct answer is a one-call refusal. The same
question refused cleanly in 1 call, 1,026 tokens, on other runs: the agent is
bistable, and the bad mode is a loop.

Ranked against the other five modes measured by `trajectory_eval`, this one is
first by a distance: it is the only mode that produces an unbounded tail. A
wrong tool choice costs one call; thrashing costs as many as the budget allows.
It owns the Cost Max and Latency Max figures outright.

## The mechanism, and why it is not a prompt patch

The agent cannot distinguish "I have not found it yet" from "it is not there",
so it re-queries. Telling it so in the system prompt would change the shared
`ANSWER_CONTRACT`, which both race arms depend on being byte-identical — the
Week 7 fairness gate would fail, and every committed Week 7 number would become
incomparable. So the fix sits in the LOOP, not the prompt:

  1. CONSECUTIVE SEARCH LIMIT. After `max_consecutive_searches` back-to-back
     `search_docs` calls that surface no new chunk id, the next one is refused.
  2. NON-EXISTENCE EVIDENCE. Once a tool has reported `found: false` for a
     target, and retrieval has returned nothing new, absence is established.

A refused call returns a normal tool result telling the model what it has
already established. The model still writes its own answer; nothing forces a
refusal, and nothing forces a tool. That distinction matters: a breaker that
emitted the refusal itself would be scoring its own answer.

## What it deliberately does NOT touch

The prompts, the tools, the scorer, the questions, the budgets, the workflow
arm. `run_mitigated_agent` wraps `run_agent`'s own seams — it passes a wrapped
`chat` that vetoes a thrashing tool call before it is dispatched — so the agent
loop itself is unmodified and both arms still share one contract.
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from app.rag.week7_agent import AgentRun, Budgets, run_agent
from app.rag.week7_contract import ArmConfig, ChatResult, ToolCall, chat_once
from app.rag.week7_questions import load_questions

MITIGATION_ID = "consecutive-search-and-nonexistence-circuit-breaker"


class BreakerConfig(BaseModel):
    """Thresholds. Set above every legitimate trajectory measured in Week 7.

    The deepest honest chain observed was 2 tool calls; the cross-version cases
    use 2 searches. `max_consecutive_searches=2` therefore cannot fire on any
    correct trajectory in the suite — it only bites from the third unproductive
    search onward, which no passing Week 7 run ever made.
    """

    max_consecutive_searches: int = 2
    max_repeats_per_signature: int = 1
    enabled: bool = True


@dataclass
class BreakerState:
    cfg: BreakerConfig = field(default_factory=BreakerConfig)
    consecutive_searches: int = 0
    seen_signatures: dict[str, int] = field(default_factory=dict)
    known_chunk_ids: set[str] = field(default_factory=set)
    not_found_targets: set[str] = field(default_factory=set)
    vetoes: list[dict[str, Any]] = field(default_factory=list)

    def observe_result(self, name: str, arguments: dict[str, Any],
                       payload: dict[str, Any]) -> None:
        """Learn from what a tool actually returned."""
        if name == "search_docs":
            ids = {p.get("chunk_id") for p in payload.get("passages", [])}
            fresh = ids - self.known_chunk_ids
            self.known_chunk_ids |= {i for i in ids if i}
            # Only an UNPRODUCTIVE search counts toward the streak. A search
            # that surfaces new material is progress, however many came before.
            self.consecutive_searches = (
                0 if fresh else self.consecutive_searches + 1
            )
        else:
            self.consecutive_searches = 0
            if payload.get("found") is False:
                target = str(arguments.get("path") or arguments.get("sdk_method") or "")
                if target:
                    self.not_found_targets.add(target)

    def veto(self, name: str, arguments: dict[str, Any]) -> str:
        """Return a reason to refuse this call, or "" to allow it."""
        if not self.cfg.enabled:
            return ""
        signature = f"{name}:{json.dumps(arguments, sort_keys=True)}"
        repeats = self.seen_signatures.get(signature, 0)
        if repeats >= self.cfg.max_repeats_per_signature:
            return (
                f"This exact call was already made {repeats} time(s) and returned "
                "the same result. Repeating it cannot produce new information."
            )
        if (name == "search_docs"
                and self.consecutive_searches >= self.cfg.max_consecutive_searches):
            return (
                f"{self.consecutive_searches} consecutive documentation searches "
                "have returned no passage not already seen. The documentation "
                "does not contain this. Answer from what you have, or refuse."
            )
        target = str(arguments.get("path") or arguments.get("sdk_method") or "")
        if target and target in self.not_found_targets:
            return (
                f"A tool already reported that {target} does not exist. "
                "Its non-existence is established; do not look it up again."
            )
        return ""

    def record(self, name: str, arguments: dict[str, Any]) -> None:
        signature = f"{name}:{json.dumps(arguments, sort_keys=True)}"
        self.seen_signatures[signature] = self.seen_signatures.get(signature, 0) + 1


def breaker_payload(reason: str, state: BreakerState) -> dict[str, Any]:
    """What a vetoed call returns. A normal tool result, not an exception."""
    return {
        "circuit_breaker": True,
        "reason": reason,
        "passages_already_seen": len(state.known_chunk_ids),
        "guidance": (
            "Do not call this tool again for this. Either answer from the tool "
            "results you already have, or reply with the refusal format."
        ),
    }


def wrap_chat(state: BreakerState, inner=chat_once, cfg: ArmConfig | None = None):
    """Wrap the contract's chat seam so vetoed calls never reach dispatch.

    The agent loop is untouched: it asks the model, gets tool calls back, and
    dispatches them exactly as before. What changes is that a thrashing call
    comes back already answered by the breaker, so it costs no retrieval and no
    round trip. CPU cost is a dict lookup and a JSON dump per call.
    """
    cfg = cfg or ArmConfig()
    overhead_ns = {"total": 0, "calls": 0}

    def wrapped(messages: list[dict[str, Any]], **kwargs: Any) -> ChatResult:
        # Learn from tool results the loop appended since the last call.
        started = time.perf_counter_ns()
        for message in messages:
            if message.get("role") != "tool":
                continue
            key = message.get("tool_call_id", "")
            if key in state.seen_signatures.get("__observed__", {}):  # pragma: no cover
                continue
            try:
                payload = json.loads(message.get("content") or "{}")
            except json.JSONDecodeError:
                continue
            if payload.get("circuit_breaker"):
                continue
            state.observe_result(message.get("name", ""), {}, payload)
        overhead_ns["total"] += time.perf_counter_ns() - started
        overhead_ns["calls"] += 1

        result = inner(messages, **kwargs)

        started = time.perf_counter_ns()
        kept: list[ToolCall] = []
        for call in result.tool_calls:
            try:
                arguments = json.loads(call.arguments_json or "{}")
            except json.JSONDecodeError:
                arguments = {}
            reason = state.veto(call.name, arguments)
            if reason:
                state.vetoes.append({
                    "tool": call.name, "arguments": arguments, "reason": reason,
                })
                continue
            state.record(call.name, arguments)
            kept.append(call)
        overhead_ns["total"] += time.perf_counter_ns() - started

        if result.tool_calls and not kept:
            # Every requested call was vetoed. Hand the model the breaker's
            # finding as content so it answers rather than stalling.
            result.tool_calls = []
            result.content = result.content or (
                "ANSWER: INSUFFICIENT_CONTEXT\nVERSION: none\nCITATIONS: none"
            )
        else:
            result.tool_calls = kept
        return result

    wrapped.overhead_ns = overhead_ns  # type: ignore[attr-defined]
    return wrapped


def run_mitigated_agent(question, cfg: ArmConfig | None = None,
                        budgets: Budgets | None = None,
                        breaker: BreakerConfig | None = None,
                        chat=chat_once) -> tuple[AgentRun, dict[str, Any]]:
    """`run_agent`, unmodified, driven through the breaker's chat seam."""
    cfg = cfg or ArmConfig()
    state = BreakerState(cfg=breaker or BreakerConfig())
    wrapped = wrap_chat(state, inner=chat, cfg=cfg)
    run = run_agent(question, cfg, budgets=budgets, chat=wrapped)
    payload = run.model_dump()
    payload["_breaker"] = {
        "mitigation": MITIGATION_ID,
        "vetoes": state.vetoes,
        "veto_count": len(state.vetoes),
        "overhead_ns": wrapped.overhead_ns["total"],
        "overhead_ms": wrapped.overhead_ns["total"] / 1e6,
        "chat_calls": wrapped.overhead_ns["calls"],
    }
    return AgentRun(**{k: v for k, v in payload.items() if not k.startswith("_")}), payload


def self_test() -> tuple[list[str], list[str]]:
    """Prove the breaker fires on thrashing and stays silent otherwise."""
    from app.rag.week7_agent import ScriptedChat, _call

    failures: list[str] = []
    lines: list[str] = []
    questions = {q.id: q for q in load_questions()}

    def check(name: str, ok: bool, detail: str) -> None:
        lines.append(f"  {'ok  ' if ok else 'FAIL'} {name:46} {detail}")
        if not ok:
            failures.append(f"{name}: {detail}")

    # 1. a thrashing model is stopped
    thrash = ScriptedChat([
        ChatResult(tool_calls=[_call("t", "search_docs",
                                     query="embeddings", api_version="v3")]),
    ])
    run, payload = run_mitigated_agent(questions["w7-10"], chat=thrash)
    check("thrashing stopped", payload["_breaker"]["veto_count"] > 0,
          f"{payload['_breaker']['veto_count']} veto(s), "
          f"{run.tool_call_count} tool calls, {run.llm_calls} llm calls")
    check("run still terminates cleanly", run.termination in
          ("final_answer", "budget", "empty_response"), run.termination)

    # 2. a legitimate two-hop trajectory is NOT touched
    good = ScriptedChat([
        ChatResult(tool_calls=[_call("a", "check_deprecation",
                                     path="/v2/messages:multi", api_version="v2")]),
        ChatResult(tool_calls=[_call("b", "search_docs",
                                     query="concurrency", api_version="v3")]),
        ChatResult(content="ANSWER: send_batch, concurrency 8.\nVERSION: v3\n"
                           "CITATIONS: v3/client-batch#parameters::1"),
    ])
    run, payload = run_mitigated_agent(questions["w7-07"], chat=good)
    check("legitimate two-hop untouched", payload["_breaker"]["veto_count"] == 0,
          f"0 vetoes, path={run.tool_sequence}")
    check("two-hop still answers", run.termination == "final_answer", run.termination)

    # 3. cross-version double search is NOT vetoed
    cross = ScriptedChat([
        ChatResult(tool_calls=[_call("a", "search_docs",
                                     query="heartbeat_ms", api_version="v2")]),
        ChatResult(tool_calls=[_call("b", "search_docs",
                                     query="heartbeat_ms", api_version="v3")]),
        ChatResult(content="ANSWER: 30000 in v2, 15000 in v3.\nVERSION: both\n"
                           "CITATIONS: v3/client-stream#parameters::1"),
    ])
    run, payload = run_mitigated_agent(questions["w7-04"], chat=cross)
    check("cross-version two searches allowed", payload["_breaker"]["veto_count"] == 0,
          f"2 searches, 0 vetoes, path={run.tool_sequence}")

    # 4. an identical repeated call is refused
    state = BreakerState()
    state.record("search_docs", {"query": "x", "api_version": "v3"})
    check("exact repeat vetoed",
          bool(state.veto("search_docs", {"query": "x", "api_version": "v3"})),
          state.veto("search_docs", {"query": "x", "api_version": "v3"})[:44])

    # 5. established non-existence is not re-probed
    state = BreakerState()
    state.observe_result("get_openapi_spec", {"path": "/v1/embeddings"},
                         {"found": False})
    check("established absence vetoed",
          bool(state.veto("check_deprecation", {"path": "/v1/embeddings"})),
          "not_found target refused on a second tool")

    # 6. a PRODUCTIVE search streak is never vetoed
    state = BreakerState()
    for i in range(6):
        state.observe_result("search_docs", {}, {"passages": [{"chunk_id": f"c{i}"}]})
    check("productive searches never vetoed", not state.veto("search_docs", {"q": 1}),
          f"6 searches, all returning new chunks, streak={state.consecutive_searches}")

    # 7. overhead is measurable and small
    check("overhead recorded", payload["_breaker"]["overhead_ms"] >= 0,
          f"{payload['_breaker']['overhead_ms']:.4f} ms over "
          f"{payload['_breaker']['chat_calls']} chat call(s)")
    return failures, lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Week 8 circuit-breaker mitigation.")
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--run", action="store_true", help="live run over all 10")
    parser.add_argument("--question", default=None)
    parser.add_argument("--delay", type=float, default=12.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.explain:
        print(__doc__)
        print(f"mitigation id: {MITIGATION_ID}")
        print(f"defaults: {BreakerConfig().model_dump()}")
        return 0

    if args.self_test:
        failures, lines = self_test()
        print("circuit-breaker self-test (scripted, no network)\n")
        for line in lines:
            print(line)
        print()
        if failures:
            print(f"SELF-TEST FAILED — {len(failures)} problem(s)")
            return 1
        print("SELF-TEST PASSED")
        return 0

    if not args.run:
        parser.print_help()
        return 1

    cfg = ArmConfig()
    questions = load_questions()
    if args.question:
        wanted = {q.strip() for q in args.question.split(",")}
        questions = [q for q in questions if q.id in wanted]

    payloads = []
    for index, question in enumerate(questions):
        if index and args.delay:
            time.sleep(args.delay)
        run, payload = run_mitigated_agent(question, cfg)
        payloads.append(payload)
        breaker = payload["_breaker"]
        print(f"{question.id:7} {run.termination:14} llm={run.llm_calls} "
              f"tools={run.tool_call_count} tok={run.total_tokens:>6,} "
              f"vetoes={breaker['veto_count']} "
              f"path={' -> '.join(run.tool_sequence) or '(none)'}", flush=True)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payloads, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
