# Week 9: MCP, the standard way the agent reaches tools

The Week 9 agent no longer imports its tools. It reads a config file naming MCP servers, starts
each one, and asks it over the Model Context Protocol what tools it has. A second server was
then added by editing that config alone, and the agent used its tools without a line of agent
code changing.

```text
User
 ↓
Week 9 agent (app/rag/week9_agent.py) ──── HTTPS, OpenAI-compatible ───→ Groq LLM
 ↓
MCP client (app/mcp/client.py)           reads config/mcp_servers.json
 ├──────── stdio JSON-RPC ────────┐
 ↓                                ↓
docs-search server              package-registry server
(app/mcp/servers/docs_server)   (app/mcp/servers/package_server)
 ↓                                ↓
week7_tools.search_docs         registry/packages.yaml
 ↓                              (locally authored stand-in data)
Chroma index over docs/
```

| commit | what |
|---|---|
| `486c83f` | Steps 3 to 5: both servers, the client, the agent, the registry. The config lists docs-search only. |
| `ffdf62a` | Step 6: the config adds package-registry. It is the only file changed. |

## Evidence

| file | what it proves |
|---|---|
| [zero_code_discovery.md](zero_code_discovery.md), [agent_diff.txt](agent_diff.txt) | The config gained one server, and the agent's source did not change. |
| `step6_A_*`, `step6_B_*` | The same agent, the same question, with config A and then config B. |
| [wire.json](wire.json), [wire.md](wire.md) | The real `initialize → tools/list → tools/call` exchange, annotated field by field. |
| `step7_*` | The run and the raw capture that `wire.json` was built from. |
| [error-recovery.md](error-recovery.md) | The description rewrite and the recoverable error, before and after. |
| `step8_before*.json`, `step8_after*.json` | The captures behind `error-recovery.md`. |
| [risk_note.md](risk_note.md) | Five lines on the package-registry server. |
| `step4_*` | The first end-to-end run over MCP, with the docs server only. |
| [final-audit.md](final-audit.md) | The Step 9 requirement matrix, with every item re-checked. |
| [interview-notes.md](interview-notes.md) | Week 9 in plain terms. |

**Dynamic discovery (Step 6).** The config changed and the agent code did not.
- `git diff 486c83f ffdf62a --stat` shows only `config/mcp_servers.json`, 5 lines.
- `app/rag/week9_agent.py` has the same SHA-256 and the same git blob at both commits.
- **Before:** 1 tool (`search_docs`). The same question was refused.
- **After:** 3 tools (`search_docs`, `get_package_versions`, `get_package_deprecation`). The LLM
  chose `get_package_deprecation`, the call was routed to package-registry, and the answer passed.
- Every count comes from the live `tools/list` responses.

**Wire protocol (Step 7).** Each connection runs `initialize → notifications/initialized →
tools/list`, and the chosen server then gets `tools/call`.
- `wire.json` holds all 12 messages, unchanged.
- The client's recorder was checked against the raw pipe bytes: 12 of 12 were identical.
- The LLM call never appears on this wire. It shows up as a 5.8 s silence between the package
  server's `tools/list` result and the `tools/call`.

**Tool guidance (Step 8).** The docs tool's description is now a guidance prompt.
- It covers what the corpus holds, generated from `docs/`, and how to pass a version, including
  release numbers.
- It explains that a search is never empty, what the tool cannot answer, and that package-release
  questions belong elsewhere.
- The client receives it through `tools/list` (captured in `step8_after.json`). The input schema
  and the search are unchanged.

**Error handling (Step 8).** An invalid `api_version` used to get the SDK's generic
`'v4' is not one of ['v2', 'v3']`. It now gets a diagnostic built from the corpus:
- `"v4"`: not documented; `v3` is the newest; do not answer from another version's values.
- `"3.2.0"`: that is a release number; use `"v3"`.
- `"latest"`: not a version name; here are the accepted values.

The corrected request succeeds, with the same result as before the change. The real model never
sent an invalid version in six attempts, three before the change and three after, even when told
to; the schema enum stopped it. So no real-model transcript of that failure exists, and none was
fabricated. See `error-recovery.md`, section 3.

**Security (Step 8).** The trust boundaries are reviewed below, and the production risk is in
`risk_note.md`. The verdict is **DO NOT SHIP**.

## Trust boundaries

What was measured, not assumed. Network access was audited by running each real server
process under a Python audit hook that logged every DNS lookup and socket connect, while
the unchanged client drove a real `tools/list` and `tools/call`.

| boundary | what crosses it | what is actually there |
|---|---|---|
| agent → Groq LLM | prompts, including every tool description and tool result | HTTPS with `GROQ_API_KEY` from `.env`. It is the only model call site (`week7_contract.chat_once`). Tool text written by a server reaches the model verbatim, so **any server can steer the model through its descriptions and results.** |
| agent → MCP client | tool choices and arguments; results back | Same process. The client refuses duplicate tool names and names that are not valid function names, and routes each call to the server that listed the tool. |
| MCP client → either server | JSON-RPC over the child process's stdin/stdout | No port is opened, so only the parent can reach a server. Servers are started with only `HOME`, `LOGNAME`, `PATH`, `SHELL`, `TERM` and `USER`, not `GROQ_API_KEY`. The client offers no `sampling`, so a server cannot call the host's model. Servers are **not sandboxed**: each runs as the user, with the repo root as its working directory. |
| docs-search server → internet | model-revision checks | **Real outbound traffic.** On its first search the server resolved `huggingface.co` twice and opened two HTTPS connections on port 443, although `bge-m3` is already cached. In that same environment `huggingface_hub` builds an `Authorization: Bearer` header from the Hugging Face token stored at `~/.cache/huggingface/token`, so those requests would carry it. Chroma telemetry is a no-op in chromadb 1.5.9, and `posthog` is not installed. |
| package-registry server → internet | nothing | **Zero** DNS lookups and zero socket connections in the audited run. |
| registry data → model | release numbers, dates, deprecation notices | `registry/packages.yaml` is locally authored stand-in data, and every result carries its `provenance` line. Whoever can edit it controls what the model is told about releases. |
| evidence files | full tool arguments and results | Written only on request. Scanned for the real key value, token shapes, auth headers and home paths: none found. |

Not done, because Step 8 is a review: the fixes below would each be a config or
infrastructure change for the owner to decide.
- Set `"env": {"HF_HUB_OFFLINE": "1"}` on the docs-search entry in `config/mcp_servers.json`, to stop
  the Hugging Face traffic and the token being sent.
- Run servers in a sandbox without access to `.env` or the home directory.
- Pin each server's code and data by hash before launch.

## Requirement check

| requirement | evidence |
|---|---|
| docs MCP server, package-registry MCP server, MCP client | `app/mcp/servers/docs_server.py`, `package_server.py`, `app/mcp/client.py` |
| agent uses only the MCP client | `week9_agent.py` imports no Week 7 tool and names no tool or server. `selfcheck` booby-traps the in-process tool, and it is never called. |
| real MCP protocol | official `mcp==1.12.4` SDK, stdio; `wire.json` |
| both servers discovered through `tools/list` | `wire.json` → `tool_counts_from_tools_list`: docs-search 1, package-registry 2, total 3 |
| config-only addition, agent unchanged | `zero_code_discovery.md` §3–4 and `agent_diff.txt`, both generated from `git diff 486c83f ffdf62a`. |
| raw `initialize`, `tools/list`, `tools/call`, annotated; MCP vs LLM stated | `wire.json`, `wire.md` §3–6 |
| tool counts before → after, with names, from `tools/list` | 1 → 3: `search_docs` → `+ get_package_versions, get_package_deprecation` |
| docstring as a prompt, visible through `tools/list` | `error-recovery.md` §4; `step8_after.json` → `probe_wire` |
| recoverable error: before, after, corrected request | `error-recovery.md` §1–2. No version is invented: all come from `docs/`. |
| the model handling the failing call | not obtainable: the model never made the call (`error-recovery.md` §3) |
| risk note, exactly 5 lines | `risk_note.md` |
| bonus: gateway and token scoping | not built |

## Reproduce

```text
.venv/bin/python -m app.mcp.selfcheck                 # 22 offline checks: both servers, client, agent loop, error recovery
.venv/bin/python -m app.mcp.client                    # live tools/list from every configured server
.venv/bin/python -m app.rag.week9_agent --question w7-01
git diff 486c83f ffdf62a --stat                       # Step 6: config only
```

Captures made before Step 8 (`step4_*`, `step6_*`, `step7_*`, `wire.json`) show the docs tool's
earlier, Step 3 description, which was correct at the time they were taken. `step8_after*.json`
shows the current one.

The rewrite made every model call larger. `w7-01` used 3,034 tokens in Step 4, with the docs server
alone, and 4,742 after Step 8. That rise combines the longer description with the two package tools
that are now also sent on every call.
