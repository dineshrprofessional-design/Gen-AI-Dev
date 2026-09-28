"""Offline checks for the Week 9 MCP stack: both servers, the client, the agent loop.

    .venv/bin/python -m app.mcp.selfcheck          # ~40 s; no LLM call, no tokens

Every check talks to the real servers over real MCP (stdio JSON-RPC, started
from config/mcp_servers.json or a temporary config). The one agent check drives
`week9_agent.run_agent` through its `chat=` seam with a scripted model. It
proves the loop's plumbing: the diagnostic reaches the model, and the model's
next call is executed. It says nothing about what a real model would choose;
report/w9/error-recovery.md has the real-model runs.

Nothing here edits a source file. The first docs search loads the embedding
model inside the docs server process, which is most of the runtime.
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from app.mcp.client import (
    DEFAULT_CONFIG,
    DuplicateToolError,
    McpClient,
    McpConnectError,
    ServerConfig,
    UnknownToolError,
    load_config,
)
from app.mcp.servers import package_server
from app.rag.week7_contract import ChatResult, ToolCall
from app.rag.week7_questions import Week7Question

BASELINE_COMMIT = "486c83f"  # Step 6: the agent and client must be unchanged since this commit

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, bool(ok), detail))
    return bool(ok)


def _text(result) -> str:
    return "\n".join(block.text for block in result.content if getattr(block, "type", "") == "text")


def check_servers_and_client() -> None:
    with McpClient(load_config(DEFAULT_CONFIG)) as client:
        routes = client.routes
        check("client: discovery merges every configured server",
              set(routes.values()) == {s.name for s in load_config(DEFAULT_CONFIG)},
              f"{routes}")
        docs = next((t for t in client.tools if t.server == "docs-search"), None)
        pkg = [t.name for t in client.tools if t.server == "package-registry"]
        check("docs server: tools/list exposes one tool", docs is not None and
              [t.name for t in client.tools if t.server == "docs-search"] == [docs.name], docs and docs.name)

        from app.mcp.servers import docs_server
        description = docs.tool.description or ""
        check("docs server: tools/list carries the rewritten description",
              description == docs_server.DESCRIPTION and "package-registry questions" in description
              and all(m in description for methods in docs_server.DOCUMENTED.values() for m in methods),
              f"{len(description)} chars")
        from app.rag import week7_tools
        week7 = next(s for s in week7_tools.tool_schemas() if s["function"]["name"] == docs.name)
        check("docs server: inputSchema is still the Week 7 schema",
              docs.tool.inputSchema == week7["function"]["parameters"])

        query = "default timeout_ms of Client.send"
        good = client.call_tool(docs.name, {"query": query, "api_version": docs_server.NEWEST})
        passages = (good.structuredContent or {}).get("passages", [])
        check("docs server: a valid search returns k passages from the requested version",
              not good.isError and len(passages) == docs_server._cfg.k
              and all(p["chunk_id"].startswith(docs_server.NEWEST + "/") for p in passages),
              f"{[p['chunk_id'] for p in passages]}")

        for bad, expect in (("v4", "No documentation exists"), ("3.2.0", "is not a documentation version"),
                            ("latest", "is not a version name")):
            r = client.call_tool(docs.name, {"query": query, "api_version": bad})
            text = _text(r)
            named = set(re.findall(r'"(v\d+)"', text)) - {bad}  # the diagnostic echoes the input
            check(f"error recovery: {bad!r} gets a diagnostic, not the generic enum error",
                  r.isError and expect in text and named <= set(docs_server.VERSIONS), text[:90])
            suggested = re.findall(r'api_version "(v\d+)"', text)
            if suggested:
                retry = client.call_tool(docs.name, {"query": query, "api_version": suggested[-1]})
                check(f"error recovery: the correction named for {bad!r} succeeds",
                      not retry.isError and retry.structuredContent == good.structuredContent,
                      f"retried with {suggested[-1]!r}")
        missing = client.call_tool(docs.name, {"api_version": "v3"})
        check("docs server: other schema errors keep the SDK wording",
              missing.isError and _text(missing) == "Input validation error: 'query' is a required property",
              _text(missing))

        registry = package_server.load_registry(package_server.DEFAULT_REGISTRY)
        check("package server: tools/list exposes its two tools",
              pkg == list(package_server.TOOLS), f"{pkg}")
        for name, args, direct in (
            ("get_package_versions", {"package": "example-sdk"}, package_server.package_versions),
            ("get_package_deprecation", {"package": "example-sdk", "version": "2.1.0"}, package_server.package_deprecation),
            ("get_package_deprecation", {"package": "example-sdk", "version": "4.0.0"}, package_server.package_deprecation),
        ):
            r = client.call_tool(name, args)
            check(f"package server: {name}{json.dumps(args)} over MCP == the lookup itself",
                  not r.isError and r.structuredContent == direct(registry, **args),
                  f"found={r.structuredContent.get('found')}")

        before = len(client.wire.messages)
        try:
            client.call_tool("made_up_tool", {})
            check("client: an unknown tool is refused before any request", False)
        except UnknownToolError:
            check("client: an unknown tool is refused before any request", len(client.wire.messages) == before)


def check_client_failures() -> None:
    docs_entry = next(s for s in load_config(DEFAULT_CONFIG) if s.name == "docs-search")
    twice = [docs_entry, docs_entry.model_copy(update={"name": "docs-copy"})]
    try:
        with McpClient(twice):
            check("client: two servers listing the same tool is refused", False)
    except DuplicateToolError as exc:
        check("client: two servers listing the same tool is refused", True, str(exc)[:80])
    try:
        with McpClient([ServerConfig(name="broken", command=".venv/bin/no-such-python")]):
            check("client: a server that cannot start is reported", False)
    except McpConnectError as exc:
        check("client: a server that cannot start is reported", "could not start" in str(exc), str(exc)[:80])


class _ScriptedModel:
    """Lap 1: a bad api_version. Lap 2: whatever the diagnostic told it. Lap 3: an answer."""

    def __init__(self, tool: str) -> None:
        self.tool, self.lap, self.retried_with = tool, 0, None

    def __call__(self, messages, **_):
        self.lap += 1
        if self.lap == 1:
            return ChatResult(tool_calls=[ToolCall("c1", self.tool, json.dumps(
                {"query": "default max_retries of Client.send", "api_version": "3.2.0"}))],
                prompt_tokens=1, completion_tokens=1)
        if self.lap == 2:
            error = json.loads(messages[-1]["content"])["error"]
            self.retried_with = re.findall(r'api_version "(v\d+)"', error)[-1]
            return ChatResult(tool_calls=[ToolCall("c2", self.tool, json.dumps(
                {"query": "default max_retries of Client.send", "api_version": self.retried_with}))],
                prompt_tokens=1, completion_tokens=1)
        return ChatResult(content="ANSWER: scripted\nVERSION: v3\nCITATIONS: none",
                          prompt_tokens=1, completion_tokens=1)


def check_agent_loop() -> None:
    from app.rag import week7_tools
    from app.rag.week9_agent import run_agent

    ran_in_process: list[str] = []
    original = week7_tools.search_docs
    week7_tools.search_docs = lambda *a, **k: ran_in_process.append("search_docs")  # must never run here
    try:
        with McpClient(load_config(DEFAULT_CONFIG)) as client:
            tool = next(t.name for t in client.tools if t.server == "docs-search")
            model = _ScriptedModel(tool)
            q = Week7Question(id="selfcheck", question="scripted", question_class="selfcheck", sdk_version="v3")
            run = run_agent(q, client, chat=model)
    finally:
        week7_tools.search_docs = original
    calls = run.tool_calls
    check("agent loop: the diagnostic reaches the model as a tool error",
          len(calls) == 2 and not calls[0].ok and "3.2.0" in calls[0].error, calls[0].error[:80] if calls else "")
    check("agent loop: the model's corrected call is executed over MCP",
          len(calls) == 2 and calls[1].ok and calls[1].server == "docs-search"
          and calls[1].arguments["api_version"] == model.retried_with and run.termination == "final_answer",
          f"retried with {model.retried_with!r}, termination={run.termination}")
    check("agent loop: no tool ran inside the agent process", not ran_in_process)


def check_agent_unchanged() -> None:
    paths = ["app/rag/week9_agent.py", "app/mcp/client.py"]
    diff = subprocess.run(["git", "diff", "--stat", BASELINE_COMMIT, "--", *paths],
                          capture_output=True, text=True, check=False)
    check(f"agent + client unchanged since the Step 6 baseline {BASELINE_COMMIT}",
          diff.returncode == 0 and not diff.stdout.strip(), diff.stdout.strip() or "0 lines")


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    for step in (check_servers_and_client, check_client_failures, check_agent_loop, check_agent_unchanged):
        try:
            step()
        except Exception as exc:  # noqa: BLE001 - a crashing check is a failing check
            check(f"{step.__name__} ran to completion", False, f"{type(exc).__name__}: {exc}")
    width = max(len(name) for name, _, _ in results)
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    failed = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{len(results) - failed}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
