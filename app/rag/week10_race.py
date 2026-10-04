"""Week 10: race the docs squad against the single agent on the same 10 Week-6 cases.

    python -m app.rag.week10_race --self-test           # offline: scripted models, real MCP + HTTP, 0 tokens
    python -m app.rag.week10_race --smoke               # the 1 unraced smoke case through both arms
    python -m app.rag.week10_race --cases 5 --delay 15  # race the next 5 selected cases (resumable)
    python -m app.rag.week10_race --failure             # the injected 500 on the failure case (squad only)
    python -m app.rag.week10_race --regrade-errors      # re-ask the judge where it errored; never re-runs an arm
    python -m app.rag.week10_race --report              # race_table.md, handoffs.log, the multiplier

Everything the race measures by was fixed in eval/week10_selection.yaml before
any run (commit cb9efad). The cases, the pass rule, the metrics and the keep/kill
rule are all read from there or from report/w10/prediction.txt, not decided here.

## Fairness

- Both arms get the same `Week7Question` objects, built from the Week 6 cases
  unchanged, and the same model and temperature through `week7_contract.chat_once`.
- They share one MCP client, so one docs server and one warm embedding model
  (warmed before the first timed run). They share one answer contract
  (`ANSWER_CONTRACT`); only the preambles differ, as in Week 7.
- The order alternates per case, single first on even positions and squad
  first on odd ones, so provider drift does not favour either arm.
- Grading is arm-blind: the same adapter, the same Week 6 assertions, and the
  same Judge v2 prompt.

## Runs are never re-run for their score

A completed run is saved to report/w10/runs/ and skipped on every later
invocation. A run lost to the provider (daily quota, a transient error) is not a
result. It is moved to runs/aborted/, logged in race.log, and the race stops;
the next invocation runs that case again. `--regrade-errors` repeats only a
judge call that failed in transport, never an arm.
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from starlette.testclient import TestClient

from app.mcp.client import McpClient, configure_logging, load_config
from app.rag.percentile import percentile
from app.rag.week10_cases import DEFAULT_SELECTION, build_document
from app.rag.week10_squad import (
    CODE, CODE_WORKER_PROMPT, ORCHESTRATOR_PREAMBLE, VERSION, VERSION_WORKER_PROMPT,
    FailurePlan, SquadRun, build_worker_app, run_squad,
)
from app.rag.week6_assertions import chunk_texts, divergent_defaults, run_case
from app.rag.week6_cases import Case, RecordedTrace, load_cases
from app.rag.week6_judge import judge_one, system_prompt
from app.rag.week7_contract import (
    ERROR_QUOTA_TPD, ERROR_RATE_LIMIT_TPM, ERROR_TRANSIENT, ArmConfig, ChatResult, TaskOutput, ToolCall,
    chat_once, classify_provider_error, load_price,
)
from app.rag.week7_questions import Week7Question
from app.rag.week9_agent import McpToolCallRecord, run_agent

OUT = Path("report/w10")
RUNS = OUT / "runs"
RACE_LOG = OUT / "race.log"
ARMS = ("single", "squad")
INFRA = (ERROR_QUOTA_TPD, ERROR_TRANSIENT, ERROR_RATE_LIMIT_TPM)


def log(line: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    RACE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with RACE_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"{stamp}  {line}\n")
    print(line, flush=True)


def git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()


def frozen_selection() -> dict:
    committed = yaml.safe_load(DEFAULT_SELECTION.read_text(encoding="utf-8"))
    if committed != build_document():
        raise SystemExit(f"{DEFAULT_SELECTION} no longer matches its rule or the Week 6 cases; refusing to race")
    return committed


def as_question(case: Case) -> Week7Question:
    return Week7Question(id=case.case_id, question=case.question, question_class=f"week6-mode-{case.mode}",
                         sdk_version=case.sdk_version, expect_refusal=case.expected_behavior == "refuse")


# --------------------------------------------------------------------------
# grading: the Week 6 instruments, unchanged, fed a new answer
# --------------------------------------------------------------------------


class Grader:
    """Build the Week 6 `recorded` block from an arm's run, then apply the Week 6 checks to it."""

    def __init__(self) -> None:
        self.cases = load_cases()
        self.by_id = {c.case_id: c for c in self.cases}
        self.texts = chunk_texts()
        self.divergent = divergent_defaults()
        self.judge_prompt = system_prompt("v2", self.cases)

    @staticmethod
    def context(tool_calls: list, transcripts: list[list[dict]]) -> tuple[list[str], dict[str, str]]:
        """Every chunk the arm's tools returned, plus any non-chunk tool result as its own block."""
        ids: list[str] = []
        for record in tool_calls:
            for cid in record.result_chunk_ids:
                if cid not in ids:
                    ids.append(cid)
        extra: dict[str, str] = {}
        for transcript in transcripts:
            for message in transcript:
                if message.get("role") != "tool":
                    continue
                name = message.get("name", "tool")
                try:
                    body = json.loads(message.get("content") or "")
                except json.JSONDecodeError:
                    body = None
                if not isinstance(body, dict) or "error" in body:
                    continue
                if name == "search_docs" and ("passages" in body or body.get("truncated")):
                    continue  # chunks are taken whole from the corpus, via the records
                key = f"tool:{name}#{sum(1 for k in extra if k.startswith(f'tool:{name}#')) + 1}"
                extra[key] = message["content"]
                ids.append(key)
        return ids, extra

    def grade(self, case_id: str, arm: str, output: TaskOutput, tool_calls: list,
              transcripts: list[list[dict]], model: str, judge: bool = True) -> dict[str, Any]:
        case = self.by_id[case_id]
        ids, extra = self.context(tool_calls, transcripts)
        texts = {**self.texts, **extra}
        if output.parse_error:
            return {"passed": False, "answer_text": "", "context_ids": ids, "assertions": [],
                    "judge": {"verdict": "NOT_JUDGED", "reason": f"no parsable answer: {output.parse_error}"}}
        recorded = RecordedTrace(
            trace_id=case.trace_id or f"week10-{case_id}", ts=datetime.now(timezone.utc).isoformat(),
            git_sha=git_sha(), prompt_version=f"week10/{arm}", model=model, params={},
            retrieval={"source": "the arm's own tool results"}, retrieved=[{"chunk_id": cid} for cid in ids],
            raw_output=output.answer, refused=output.refused, refusal_reason="", citations=output.citations,
            invalid_citations=[], generation_ran=True, retrieval_ms=0.0,
        )
        graded_case = case.model_copy(update={"recorded": recorded})
        checks = [r.model_dump() for r in run_case(graded_case, texts, self.divergent).results]
        verdict = (judge_one(graded_case, texts, None, prompt=self.judge_prompt).model_dump() if judge
                   else {"verdict": "NOT_JUDGED", "reason": "offline"})
        passed = verdict["verdict"] == "PASS" and not any(c["status"] == "fail" for c in checks)
        return {"passed": passed, "answer_text": output.answer, "context_ids": ids, "assertions": checks,
                "judge": {k: verdict.get(k, "") for k in ("verdict", "reason", "error")}}


# --------------------------------------------------------------------------
# running one arm on one case
# --------------------------------------------------------------------------


def run_arm(arm: str, question: Week7Question, client: McpClient, http: TestClient,
            cfg: ArmConfig, chat=chat_once) -> tuple[dict[str, Any], TaskOutput, list, list]:
    if arm == "single":
        run = run_agent(question, client, cfg, chat=chat)
        record = {
            "llm_calls": len(run.laps), "tool_path": list(run.tool_sequence),
            "worker_terminations": [], "hops": [],
        }
        tool_calls, transcripts = run.tool_calls, [run.transcript]
    else:
        run = run_squad(question, http, cfg, chat=chat)
        problems = run.check_accounting()
        if problems:
            raise AssertionError(f"squad accounting broken on {question.id}: {problems}")
        record = {
            "llm_calls": sum(h.llm_calls for h in run.hops),
            "tool_path": [d["worker"] for d in run.delegations],
            "worker_terminations": [w.termination for w in run.workers],
            "hops": [h.model_dump() for h in run.hops],
        }
        tool_calls, transcripts = run.tool_calls, [w.transcript for w in run.workers]
    record.update({
        "arm": arm, "case_id": question.id, "question": question.question,
        "recorded_at": datetime.now(timezone.utc).isoformat(), "git_sha": git_sha(), "model": cfg.resolved_model(),
        "termination": run.termination, "termination_detail": run.termination_detail,
        "prompt_tokens": run.prompt_tokens, "completion_tokens": run.completion_tokens,
        "cached_tokens": run.cached_tokens, "total_tokens": run.total_tokens, "cost_usd": run.cost_usd,
        "latency_ms": run.latency_ms, "rate_limit_wait_s": run.rate_limit_wait_s,
        "adjusted_latency_ms": round(run.latency_ms - run.rate_limit_wait_s * 1000.0, 3),
        "answer": run.output.model_dump(mode="json"), "run": run.model_dump(mode="json"),
    })
    return record, run.output, tool_calls, transcripts


def infra_failure(record: dict[str, Any]) -> str | None:
    """A provider failure anywhere in the run makes it a non-result."""
    details = [record["termination_detail"]] if record["termination"] == "chat_error" else []
    details += [t for t in record["worker_terminations"] if t.startswith("chat_error")]
    for detail in details:
        kind = classify_provider_error(detail)
        if kind in INFRA:
            return f"{kind}: {detail[:160]}"
    return None


def run_path(case_id: str, arm: str, folder: Path = RUNS) -> Path:
    return folder / f"{case_id}.{arm}.json"


def save(record: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def warm_up(client: McpClient) -> None:
    """Load the embedding model in the docs server before anything is timed (Week 7 rule)."""
    started = time.perf_counter()
    client.call_tool("search_docs", {"query": "warm-up", "api_version": "v3"})
    log(f"warm-up: docs server ready in {time.perf_counter() - started:.1f}s (not timed)")


def execute(case_ids: list[str], folder: Path, positions: dict[str, int], delay: float,
            failure: FailurePlan | None = None, arms: tuple[str, ...] = ARMS) -> int:
    cfg = ArmConfig()
    model = cfg.resolved_model()
    grader = Grader()
    done = 0
    with McpClient(load_config()) as client:
        warm_up(client)
        app = build_worker_app(client, chat=chat_once, cfg=cfg, model=model, price=load_price(model), failure=failure)
        with TestClient(app, raise_server_exceptions=False) as http:
            for case_id in case_ids:
                order = arms if positions.get(case_id, 0) % 2 == 0 else tuple(reversed(arms))
                for arm in order:
                    path = run_path(case_id, arm, folder)
                    if path.exists():
                        continue
                    question = as_question(grader.by_id[case_id])
                    record, output, tool_calls, transcripts = run_arm(arm, question, client, http, cfg)
                    lost = infra_failure(record)
                    if lost:
                        save(record, folder / "aborted" / f"{case_id}.{arm}.{int(time.time())}.json")
                        log(f"ABORTED {case_id} {arm}: {lost}. Not a result; the next run repeats this case.")
                        return done
                    if failure:
                        record["failure_injected"] = {"worker": failure.worker, "case_id": failure.case_id,
                                                      "http_500_attempts": failure.attempts}
                    record["grade"] = grader.grade(case_id, arm, output, tool_calls, transcripts, model)
                    save(record, path)
                    done += 1
                    g = record["grade"]
                    log(f"{case_id} {arm:6} {'PASS' if g['passed'] else 'FAIL'}  judge={g['judge']['verdict']:<10} "
                        f"tokens={record['total_tokens']:>6,}  {record['adjusted_latency_ms'] / 1000:5.1f}s  "
                        f"path={record['tool_path']}  termination={record['termination']}")
                    time.sleep(delay)
    return done


# --------------------------------------------------------------------------
# reports
# --------------------------------------------------------------------------


def load_records(case_ids: list[str]) -> dict[str, dict[str, dict]]:
    found: dict[str, dict[str, dict]] = {arm: {} for arm in ARMS}
    for case_id in case_ids:
        for arm in ARMS:
            path = run_path(case_id, arm)
            if path.exists():
                found[arm][case_id] = json.loads(path.read_text(encoding="utf-8"))
    return found


def arm_metrics(records: dict[str, dict]) -> dict[str, Any]:
    rows = list(records.values())
    latencies = [r["adjusted_latency_ms"] for r in rows]
    n = len(rows)
    return {
        "n": n,
        "passed": sum(1 for r in rows if r["grade"]["passed"]),
        "p50_ms": percentile(latencies, 0.50),
        "p99_ms": percentile(latencies, 0.99),
        "total_tokens": sum(r["total_tokens"] for r in rows),
        "cost_per_question": (sum(r["cost_usd"] for r in rows) / n) if n else 0.0,
        "llm_calls": sum(r["llm_calls"] for r in rows),
    }


def hop_totals(squad: dict[str, dict]) -> list[tuple[str, int, int]]:
    by_edge: dict[str, list[int]] = {}
    for record in squad.values():
        for hop in record["hops"]:
            by_edge.setdefault(hop["edge"], [0, 0])
            by_edge[hop["edge"]][0] += hop["total_tokens"]
            by_edge[hop["edge"]][1] += 1
    return sorted(((e, t, c) for e, (t, c) in by_edge.items()), key=lambda x: -x[1])


def write_reports(selection: dict) -> None:
    case_ids = [row["case_id"] for row in selection["selected"]]
    records = load_records(case_ids)
    missing = [f"{c}.{a}" for a in ARMS for c in case_ids if c not in records[a]]
    single, squad = arm_metrics(records["single"]), arm_metrics(records["squad"])
    multiplier = squad["total_tokens"] / single["total_tokens"] if single["total_tokens"] else 0.0
    edges = hop_totals(records["squad"])
    top = edges[0] if edges else ("-", 0, 0)
    top_share = top[1] / squad["total_tokens"] * 100 if squad["total_tokens"] else 0.0
    multiplier_line = (f"Context re-send multiplier: {squad['total_tokens']:,} squad tokens / {single['total_tokens']:,} "
                       f"single-agent tokens = {multiplier:.1f}x. Largest share: \"{top[0]}\", "
                       f"{top_share:.0f}% of all squad tokens ({top[1]:,} tokens over {top[2]} hand-offs).")

    status = "COMPLETE" if not missing else f"PARTIAL: missing {', '.join(missing)}"
    md = [f"# Week 10 race: single agent vs docs squad\n",
          f"**Status: {status}.** Cases: the 10 Week-6 cases in `eval/week10_selection.yaml`, unchanged, "
          f"rule-selected before any run. Model `{(next(iter(records['single'].values()), {}) or {}).get('model', '?')}`, "
          "temperature 0. Judge: Week 6 Judge v2 plus the Week 6 assertions, identical for both arms.\n",
          "| metric | single agent | squad (orchestrator + 2 workers) |", "|---|---|---|",
          f"| pass rate | {single['passed']}/{single['n']} ({single['passed'] / max(single['n'], 1):.0%}) "
          f"| {squad['passed']}/{squad['n']} ({squad['passed'] / max(squad['n'], 1):.0%}) |",
          f"| p50 latency | {single['p50_ms'] / 1000:.1f} s | {squad['p50_ms'] / 1000:.1f} s |",
          f"| p99 latency | {single['p99_ms'] / 1000:.1f} s | {squad['p99_ms'] / 1000:.1f} s |",
          f"| total tokens | {single['total_tokens']:,} | {squad['total_tokens']:,} |",
          f"| cost per question | ${single['cost_per_question']:.6f} | ${squad['cost_per_question']:.6f} |",
          f"| LLM calls (all cases) | {single['llm_calls']} | {squad['llm_calls']} |", "",
          f"**{multiplier_line}**\n", "## Per case\n",
          "| case | question | single | squad | single tokens | squad tokens | single s | squad s |",
          "|---|---|---|---|---|---|---|---|"]
    for row in selection["selected"]:
        cid = row["case_id"]
        s, q = records["single"].get(cid), records["squad"].get(cid)

        def cell(r, key, fmt):
            return fmt(r[key]) if r else "-"

        md.append(f"| {cid} | {row['question']} | "
                  f"{('PASS' if s['grade']['passed'] else 'FAIL') if s else '-'} | "
                  f"{('PASS' if q['grade']['passed'] else 'FAIL') if q else '-'} | "
                  f"{cell(s, 'total_tokens', lambda v: f'{v:,}')} | {cell(q, 'total_tokens', lambda v: f'{v:,}')} | "
                  f"{cell(s, 'adjusted_latency_ms', lambda v: f'{v / 1000:.1f}')} | "
                  f"{cell(q, 'adjusted_latency_ms', lambda v: f'{v / 1000:.1f}')} |")
    md += ["", "## How each number was computed\n"] + [f"- **{k}**: {v}" for k, v in selection["metrics"].items()]
    md += [f"- **pass rule**: {selection['pass_rule']}", "",
           "Ten cases give 10-point granularity on pass rate, and each arm ran once (see report/w10/prediction.txt "
           "for why a 1-case difference is treated as noise)."]
    (OUT / "race_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    lines = ["# Week 10 hand-off log: every hop of every squad run, with its provider-reported token count.",
             "# edge: who handed what to whom. A worker hop counts all of that worker's LLM calls for the hand-off;",
             "# an orchestrator hop is one LLM call, which re-sends everything before it.", "",
             f"{'case':7} {'hop':>3}  {'edge':52} {'calls':>5} {'prompt':>7} {'compl':>6} {'total':>7} {'http':>4}  note"]
    for cid in case_ids:
        for hop in (records["squad"].get(cid) or {}).get("hops", []):
            lines.append(f"{cid:7} {hop['seq']:>3}  {hop['edge']:52} {hop['llm_calls']:>5} {hop['prompt_tokens']:>7,} "
                         f"{hop['completion_tokens']:>6,} {hop['total_tokens']:>7,} {hop['http_status'] or '-':>4}  {hop['note']}")
    lines += ["", "TOTALS BY HAND-OFF (all squad runs)"]
    for edge, tokens, count in edges:
        share = tokens / squad["total_tokens"] * 100 if squad["total_tokens"] else 0.0
        lines.append(f"  {edge:52} {count:>3} hops {tokens:>9,} tokens  {share:5.1f}%")
    lines += [f"  {'squad total':52} {'':>8} {squad['total_tokens']:>9,} tokens", "", multiplier_line]
    (OUT / "handoffs.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT / 'race_table.md'} and {OUT / 'handoffs.log'} ({status})")
    print(multiplier_line)


# --------------------------------------------------------------------------
# offline self-test: scripted models through the real MCP + HTTP stack
# --------------------------------------------------------------------------


class ScriptedModels:
    """Plays every actor without a network. Each actor is told apart by its system prompt."""

    def __init__(self, misuse_tool: bool = True) -> None:
        self.calls = 0
        self.misuse_tool = misuse_tool

    def __call__(self, messages, **_) -> ChatResult:
        self.calls += 1
        system = messages[0]["content"]
        lap = sum(1 for m in messages if m.get("role") == "assistant") + 1
        usage = {"prompt_tokens": 100 * lap + 7, "completion_tokens": 10 + lap}
        call = lambda name, **args: ChatResult(tool_calls=[ToolCall(f"c{self.calls}", name, json.dumps(args))], **usage)
        answer = ChatResult(content="ANSWER: In v3, Client.send() retries a 5xx [v3/client-send#clientsend::0].\n"
                                    "VERSION: v3\nCITATIONS: v3/client-send#clientsend::0", **usage)
        if ORCHESTRATOR_PREAMBLE in system:
            if lap == 1:
                return call("delegate_version_check", question="q")
            if lap == 2:
                return call("delegate_code_sample", task="q", target_version="v3", version_notes="none")
            return answer
        if system == VERSION_WORKER_PROMPT:
            if lap == 1:
                return call("search_docs", query="retry 5xx", api_version="v3")
            return ChatResult(content="TARGET_VERSION: v3\nVERSION_NOTES: none found\nDEPRECATION: none found", **usage)
        if system == CODE_WORKER_PROMPT:
            if lap == 1 and self.misuse_tool:
                return call("get_package_versions", package="example-sdk")  # outside its allow-list
            if lap <= 2:
                return call("search_docs", query="retry 5xx", api_version="v3")
            return ChatResult(content="CONTENT: In v3 a 5xx is retried [v3/client-send#clientsend::0].\nCODE: none", **usage)
        if lap == 1:  # the single agent
            return call("search_docs", query="retry 5xx", api_version="v3")
        return answer


def self_test() -> int:
    results: list[tuple[str, bool, str]] = []
    check = lambda name, ok, detail="": results.append((name, bool(ok), detail))
    frozen_selection()
    check("selection file matches its rule and the Week 6 cases", True)
    grader = Grader()
    case = grader.by_id["w6-001"]
    original = [r.status for r in run_case(case, grader.texts, grader.divergent).results]
    rec = case.recorded
    rebuilt = case.model_copy(update={"recorded": rec.model_copy(update={
        "retrieved": [{"chunk_id": h["chunk_id"]} for h in rec.retrieved]})})
    check("grading adapter: a rebuilt recorded block gives the same Week 6 assertion results",
          [r.status for r in run_case(rebuilt, grader.texts, grader.divergent).results] == original, f"{original}")

    cfg = ArmConfig()
    question = as_question(case)
    with McpClient(load_config()) as client:
        for label, failure in (("clean", None), ("500", FailurePlan(VERSION, case.case_id))):
            models = ScriptedModels()
            app = build_worker_app(client, chat=models, cfg=cfg, model="scripted", price=load_price("scripted"),
                                   failure=failure)
            with TestClient(app, raise_server_exceptions=False) as http:
                squad = run_squad(question, http, cfg, chat=models, price=load_price("scripted"))
            hops = [(h.edge, h.http_status) for h in squad.hops]
            check(f"squad ({label}): hop tokens sum exactly to the run total", not squad.check_accounting(),
                  f"{squad.total_tokens} tokens over {len(squad.hops)} hops")
            if failure is None:
                code = next(w for w in squad.workers if w.worker == CODE)
                check("squad: the code-sample worker cannot use a tool outside its allow-list",
                      not code.tool_calls[0].ok and "not available" in code.tool_calls[0].error, code.tool_calls[0].error[:70])
                check("squad: hand-off edges follow orchestrator -> worker -> orchestrator",
                      [e for e, _ in hops] == ["question -> orchestrator", "orchestrator -> version/deprecation worker",
                                               "version/deprecation worker -> orchestrator", "orchestrator -> code-sample worker",
                                               "code-sample worker -> orchestrator"], f"{hops}")
                ids, _ = Grader.context(squad.tool_calls, [w.transcript for w in squad.workers])
                check("squad: grading context = the chunks the workers' tools returned", bool(ids), f"{len(ids)} chunk ids")
            else:
                hop500 = next((h for h in squad.hops if h.http_status == 500), None)
                check("failure plan: the version worker answers a real HTTP 500, costing no tokens",
                      hop500 is not None and hop500.total_tokens == 0 and failure.attempts == 1,
                      f"attempts={failure.attempts}, hop={hop500 and hop500.edge}")
                error_seen = any("HTTP 500" in (m.get("content") or "") for m in squad.transcript if m.get("role") == "tool")
                check("failure plan: the orchestrator receives the 500 as a tool error", error_seen)
        single = run_arm("single", question, client, None, cfg, chat=ScriptedModels())[0]
        check("single arm: scripted run completes and is metered", single["termination"] == "final_answer"
              and single["total_tokens"] > 0, f"{single['total_tokens']} tokens, path {single['tool_path']}")
    width = max(len(n) for n, _, _ in results)
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    failed = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{len(results) - failed}/{len(results)} checks passed (no LLM, no tokens)")
    return 1 if failed else 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Week 10 race: single agent vs docs squad.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--smoke", action="store_true", help="the unraced smoke case through both arms")
    mode.add_argument("--cases", type=int, metavar="N", help="race the next N selected cases not yet complete")
    mode.add_argument("--failure", action="store_true", help="squad on the failure case with the version worker at 500")
    mode.add_argument("--regrade-errors", action="store_true")
    mode.add_argument("--report", action="store_true")
    parser.add_argument("--delay", type=float, default=15.0, help="seconds between runs (provider per-minute cap)")
    args = parser.parse_args(argv)
    configure_logging()

    if args.self_test:
        return self_test()
    selection = frozen_selection()
    ids = [row["case_id"] for row in selection["selected"]]
    positions = {cid: i for i, cid in enumerate(ids)}
    if args.report:
        write_reports(selection)
        return 0
    if args.smoke:
        n = execute([selection["smoke_case"]], RUNS / "smoke", {}, args.delay)
        log(f"smoke: {n} run(s) recorded under {RUNS / 'smoke'} (not part of the race)")
        return 0
    if args.failure:
        plan = FailurePlan(VERSION, selection["failure_case"])
        execute([plan.case_id], RUNS / "failure", {}, args.delay, failure=plan, arms=("squad",))
        return 0
    if args.regrade_errors:
        grader, n = Grader(), 0
        for path in sorted(RUNS.rglob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            if "grade" not in record or record["grade"]["judge"]["verdict"] != "ERROR":
                continue
            run = SquadRun(**record["run"]) if record["arm"] == "squad" else None
            tool_calls = run.tool_calls if run else [McpToolCallRecord(**c) for c in record["run"]["tool_calls"]]
            transcripts = [w.transcript for w in run.workers] if run else [record["run"]["transcript"]]
            record["grade"] = grader.grade(record["case_id"], record["arm"], TaskOutput(**record["answer"]),
                                           tool_calls, transcripts, record["model"])
            save(record, path)
            n += 1
            log(f"regraded {path.name}: judge={record['grade']['judge']['verdict']}")
        print(f"{n} judge error(s) regraded")
        return 0
    todo = [cid for cid in ids if not all(run_path(cid, arm).exists() for arm in ARMS)][: args.cases]
    if not todo:
        print("all 10 cases are complete; run --report")
        return 0
    log(f"race: {len(todo)} case(s) this session: {todo}")
    n = execute(todo, RUNS, positions, args.delay)
    remaining = [cid for cid in ids if not all(run_path(cid, arm).exists() for arm in ARMS)]
    log(f"race: {n} run(s) recorded this session; cases still to run: {remaining or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
