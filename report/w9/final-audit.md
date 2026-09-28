# Week 9 final audit

**Verdict: COMPLETE on the audit matrix: 13 of 13 requirements pass on repository evidence.**
Two real gaps remain (section 7):
- one part of the original brief is only partly evidenced
- the Step 7 to 9 work is not committed yet

Every value below was re-checked during the audit, by running the command or parsing the file
it cites. Nothing is carried over from earlier reports unchecked.

## 1. Repository state

- Branch: `ragew8/trajectory-evals`.
- Week 9 commits:
  - `486c83f`: Steps 3 to 5; the config lists docs-search only.
  - `ffdf62a`: Step 6; the config adds package-registry.
- Not committed:
  - the Step 8 change to `app/mcp/servers/docs_server.py`
  - `app/mcp/selfcheck.py`
  - every Step 6 to 9 evidence file in `report/w9/`
- Nine staged changes that predate Week 9 (`.claude/settings.json`, `.gitignore`, `corpus/`, `report/*.pdf`,
  `report/build_report.py`, `sample-corpus/`, `traces/`) belong to no Week 9 commit. They were left untouched.
- Week 7 files (`week7_agent.py`, `week7_tools.py`, `week7_contract.py`): 0 diff lines against HEAD.

## 2. Inventory

| area | file | purpose |
|---|---|---|
| MCP client | `app/mcp/client.py` | read the config, start servers over stdio, `initialize`, `tools/list`, route `tools/call`, convert tools to the OpenAI shape, record the wire |
| docs server | `app/mcp/servers/docs_server.py` | `search_docs` over MCP, adapting `week7_tools.search_docs`; rich description; recoverable version diagnostic |
| package server | `app/mcp/servers/package_server.py` | `get_package_versions`, `get_package_deprecation`, resource `registry://example-sdk/support` |
| registry data | `registry/packages.yaml`, `registry/README.md` | locally authored stand-in data, disclosed as such |
| agent | `app/rag/week9_agent.py` | the tool-calling loop: tools only from the MCP client; the LLM call through `week7_contract.chat_once` |
| config | `config/mcp_servers.json` | server registration (`mcpServers`) |
| tests | `app/mcp/selfcheck.py` | 22 offline checks |
| zero-code evidence | `report/w9/zero_code_discovery.md`, `report/w9/agent_diff.txt`, `step6_{A,B}_run.json`, `step6_{A,B}_mcp_messages.jsonl` | config A vs B, same agent |
| wire evidence | `report/w9/wire.json`, `report/w9/wire.md`, `step7_run.json`, `step7_mcp_messages.jsonl` | raw JSON-RPC, annotated |
| error evidence | `report/w9/error-recovery.md`, `step8_before.json`, `step8_after.json`, `step8_{before,after}_provoked.json` | before and after of the error path and the description |
| risk | `report/w9/risk_note.md`, `report/w9/README.md` (trust boundaries) | security review |
| summary | `report/w9/README.md`, `report/w9/interview-notes.md`, this file | |

## 3. The MCP lifecycle, traced in source

| # | step | where |
|---|---|---|
| 1 | the client starts each configured server as a stdio child process | `client.py:378` `stdio_client(...)`, with a `ClientSession` over wire-tapped streams (`:386`) |
| 2 | `initialize` | `client.py:395` `session.initialize()` |
| 3 | `notifications/initialized` | sent **by the SDK** inside `session.initialize()` (`mcp/client/session.py:174`), not by our code |
| 4 | `tools/list`, every page | `client.py:418` `session.list_tools(cursor)`; duplicate names refused (`:439`); name → server routes (`:444`) |
| 5 | discovered tools reach the agent | `week9_agent.py:318` `schemas = client.openai_tools()` (`client.py:458`) |
| 6 | the LLM chooses a tool | `week9_agent.py:348–352` `chat(..., tools=schemas, tool_choice="auto")` → `week7_contract.py:411` `completions.create` |
| 7 | `tools/call` | `week9_agent.py:389` → `:266` `client.call_tool` → `client.py:477` `session.call_tool` on the owning server |
| 8 | the server executes | `docs_server.py:229` `week7_tools.search_docs(...)`; `package_server.py:374` `lookup(load_registry(...))` |
| 9 | the result returns to the agent | `week9_agent.py:270` `tool_payload(result)`, then `:394`, appended as a tool message |
| 10 | the LLM gives its final answer | the next lap with no tool call: `week9_agent.py:403–404` `parse_task_output`, `final_answer` |

Where the real implementation differs from the textbook diagram:
- Step 3 is the SDK's own work.
- Steps 1 to 4 run **once per server, when the client connects**, before the question reaches the LLM.
  They are not repeated per question or per lap.
- The same discovered tool list is re-sent to the LLM on every lap.

## 4. Requirement matrix

| Week 9 requirement | evidence | status |
|---|---|---|
| Docs MCP server | `docs_server.py`. `selfcheck`: tools/list, valid search, three diagnostics, schema errors. Real-LLM `tools/call` in `step4_mcp_messages_w7-01.jsonl` and `step6_A_mcp_messages.jsonl` | PASS |
| Package MCP server | `package_server.py`. `selfcheck`: tools/list, and 3 calls equal to the lookup. Real-LLM `tools/call` in `step6_B_mcp_messages.jsonl` and `wire.json` #11–12 | PASS |
| MCP client | `client.py`. `selfcheck`: merged discovery, unknown tool refused before any request, duplicate names refused, a server that fails to start is reported | PASS |
| Dynamic discovery | `step6_A_mcp_messages.jsonl`: `tools/list` = `[search_docs]`. `step6_B_mcp_messages.jsonl`: + `[get_package_versions, get_package_deprecation]`. The same agent, and in B it called the new tool | PASS |
| Zero agent-code change | `agent_diff.txt`: `git diff 486c83f ffdf62a -- app/rag/week9_agent.py` is empty. Git blob `dd8734a…` at both commits. SHA-256 `51345f5b…` recorded in `zero_code_discovery.md` and unchanged now. `--stat`: config only | PASS |
| `initialize` | `wire.json` #1–2 and #6–7: request, and response with the same `id` | PASS |
| `tools/list` | `wire.json` #4–5 and #9–10 | PASS |
| `tools/call` | `wire.json` #11–12, package-registry. Docs `tools/call` from real runs in `step4_*` and `step6_A_*` | PASS |
| Tool count from live discovery | `wire.json` `tools/list` responses: docs-search 1 (`search_docs`), package-registry 2 (`get_package_versions`, `get_package_deprecation`), total 3. Before → after: 1 → 3 | PASS |
| Rich tool guidance | Live `tools/list` at audit time: the description equals `docs_server.DESCRIPTION` and the `step8_after.json` capture (2,435 chars, sha256 `1121b0ae…`). It covers all 10 checked points: what it searches, when to use it, corpus contents, `api_version`, release numbers, `query`, empty results, the recoverable error, limitations, and the package-registry distinction | PASS |
| Recoverable diagnostic | `error-recovery.md` §1–2, from `step8_before/after.json`. For the same three invalid calls: the old SDK error, then the new diagnostic, then the corrected request, then a successful result identical to before the change | PASS |
| `risk_note.md` | Checked mechanically: 5 lines, no blank line, heading or bullet, one sentence each; topics in order: author, reach, logging, token impact, verdict (**DO NOT SHIP**) | PASS |
| MCP / LLM separation | **Source:** no server imports an LLM, HTTP client or agent module. The only model call is `week7_contract.py:411`, sent to `GROQ_BASE_URL` (`generate.py:128`). Servers get a minimal environment without `GROQ_API_KEY`, and the client offers no `sampling`. **Runtime:** audited server processes made 0 connections to Groq. The docs server contacted only `huggingface.co` (model load); the package server made no network contact | PASS |

The brief's own requirements and deliverables:

| brief | where | status |
|---|---|---|
| 1. one query provably calls a tool from the new server | `step6_B_run.json`: `get_package_deprecation` → package-registry | PASS |
| 2. git diff, zero agent lines | `agent_diff.txt` | PASS |
| 3. tool counts before → after, with names, from `tools/list` | `zero_code_discovery.md` §1–2; `wire.json` `tool_counts_from_tools_list` | PASS |
| 4. raw exchange, annotated; one line on where the model call happens | `wire.json` (`annotations`, `about.model_call`), `wire.md` | PASS |
| 5. docstring as a prompt, recoverable error, **before/after transcript of the model handling that failing call** | description and diagnostic: PASS. The model transcript: see gap 1 | PARTIAL |
| 6. five-line risk note | `risk_note.md` | PASS |
| file `error_before_after.md` | provided as `error-recovery.md`, the name chosen in Step 8 | name differs |
| bonus: gateway and token scoping | not attempted | not done |

## 5. Every captured `tools/call`

| capture | server | tool | arguments | result |
|---|---|---|---|---|
| `step4_mcp_messages_w7-01.jsonl`, real LLM | docs-search | `search_docs` | `{"api_version": "v3", "query": "Client.send retry 4xx"}` | 5 passages, top `v3/client-send#clientsend::0` |
| `step6_A_mcp_messages.jsonl`, real LLM | docs-search | `search_docs` ×2 | v3 `"example-sdk release 2.1.0 deprecated"`; v2 `"example-sdk deprecated"` | 5 passages each; none about releases, so the answer was a refusal |
| `step6_B_…` and `wire.json` #11–12, real LLM | package-registry | `get_package_deprecation` | `{"package": "example-sdk", "version": "2.1.0"}` | `deprecated: true`, since 2026-01-31, replacement `3.2.0` |
| `step8_after.json` probe, direct client call | docs-search | `search_docs` | `api_version` `"v4"`, `"3.2.0"`, `"latest"`, then `"v3"` | three `isError` diagnostics, then 5 passages |

No credentials appear in any of them: 28 Week 9 files were scanned for the real Groq key value,
the real Hugging Face token value, token shapes, auth header values and home paths.

## 6. Tests run during this audit

| test | result |
|---|---|
| `python -m app.mcp.selfcheck`: both servers, client, agent loop, error recovery, agent unchanged since `486c83f`; no LLM | **22 / 22 PASS** |
| `python -m json.tool report/w9/wire.json`, plus: all 12 messages identical to `step7_mcp_messages.jsonl`, every response paired with its request by (server, id), 2 notifications without an `id` | PASS |
| live smoke test: the real LLM and both servers, one question needing each ("Is example-sdk 2.1.0 deprecated, and in v3 does Client.send() retry a 4xx?") | PASS. Path `get_package_deprecation` → `search_docs`, each routed to its own server; the answer is correct; 7,194 tokens. Not saved to a file. |
| network audit of both real server processes (Python audit hook, scratch tool) | docs-search: 5 events, `huggingface.co` only. package-registry: 0. Groq: 0 from either |

**Does the agent retry automatically?** It hands every `isError` result back to the model as
`{"error": "<diagnostic>"}` and calls the model again. It has no retry logic of its own.
A corrected call happens only if the model chooses to make one on its next lap.
- `selfcheck` proves the plumbing with a **scripted** model: diagnostic → corrected call → success.
- The real model was never observed doing it, because it never made the failing call (gap 1).

## 7. Remaining gaps

1. **No real-model transcript of handling the failing call** (brief requirement 5).
   - In six real runs, three before the change and three after, gpt-oss-120b never sent an
     invalid `api_version`, even when told to. The schema's `["v2", "v3"]` enum stopped it.
   - The evidence is therefore the protocol-level before/after, plus the scripted-model loop check.
   - It could be closed without code changes by running the same questions on a model that does not
     honour enums (`ArmConfig(model=...)`). That was not done.
2. **The Step 7 to 9 work is not committed.**
   - `docs_server.py` is modified, and `selfcheck.py` and every Step 6 to 9 evidence file are untracked.
   - A clone of `ragew8/trajectory-evals` would lack `wire.json`, `risk_note.md`, `error-recovery.md`,
     `agent_diff.txt`, the diagnostic and the tests.
3. **Two supporting checks used scratch tools that are not in the repo:** the wire tap's 12/12
   byte-fidelity check, and the network audit. Their method and results are documented (`wire.md` §1,
   `README.md`, this file), but they cannot be re-run from the repository alone.

These are documented findings, not requirement gaps:
- The docs server contacts `huggingface.co` on first search and would send the stored Hugging Face
  token. The recommended fix is `HF_HUB_OFFLINE=1` on its config entry; it has not been applied.
- The servers are not sandboxed, and nothing pins their code before launch.
- The package server's support-matrix resource is never read by the host.
- `wire.json` predates the Step 8 description rewrite, so it shows the Step 3 description, which
  was correct when it was captured.

## 8. Fix made during the audit

`report/w9/agent_diff.txt` was missing. It is a named deliverable in the brief's submission checklist.
It was generated from real `git` output: the agent diff between the two commits is empty, the blob
is identical at both, and the stat and config diff are included. Nothing else was changed.
