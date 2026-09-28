# Week 9 Step 6: zero-code tool discovery

Adding the package-registry MCP server took one config entry and no code. The
same Week 9 agent was run on the same question before and after that one change.
Before the change it discovered 1 tool and refused. After it, the agent
discovered 3 tools and answered by calling one it had never been told about.

| | |
|---|---|
| branch | `ragew8/trajectory-evals` |
| baseline commit (config A: docs only) | `486c83f` |
| after commit (config B: docs + package-registry) | `ffdf62a` |
| `git diff 486c83f ffdf62a --stat` | `config/mcp_servers.json \| 5 +++++`, and nothing else |

Every tool count and name below was read from the live `tools/list` responses
recorded in `step6_A_mcp_messages.jsonl` and `step6_B_mcp_messages.jsonl`. None
of them was written by hand.

## 1. Baseline

```text
Configuration:  Docs Server only (config/mcp_servers.json @ 486c83f)
Servers:        docs-search
Tool count:     1
Tool names:     search_docs
```

The live `tools/list` from `docs-search` returned 1 tool: `search_docs`.

## 2. After configuration change

```text
Configuration:  Docs + Package Registry (config/mcp_servers.json @ ffdf62a)
Docs tools:     search_docs                                      (1, from docs-search's tools/list)
Package tools:  get_package_versions, get_package_deprecation    (2, from package-registry's tools/list)
Total discovered tools: 3
```

The MCP client merged the two `tools/list` responses into one tool set and one
routing table:

```text
search_docs              -> docs-search
get_package_versions     -> package-registry
get_package_deprecation  -> package-registry
```

The tools the LLM was offered are **exactly** the union of the two live
`tools/list` responses, with name, description and inputSchema compared field by
field (`tools_offered` in `step6_B_run.json`). No duplicate names were found, so
the client's duplicate check passed.

## 3. Agent source proof

```text
Agent file:  app/rag/week9_agent.py

             before (config A)                           after (config B)
sha256       51345f5bbc62c2409186bfd483d20122050145776e29c6e86a3572a165e00226
                                                         51345f5bbc62c2409186bfd483d20122050145776e29c6e86a3572a165e00226
git blob     dd8734ae2ec098a0cfeeb87ffd7dbfff1c3616ac    dd8734ae2ec098a0cfeeb87ffd7dbfff1c3616ac

Changed:     NO
```

`app/mcp/client.py`, `app/mcp/servers/docs_server.py` and
`app/mcp/servers/package_server.py` are also byte-identical before and after.
The client's git blob is `0aec3e2` at both commits.

## 4. Git diff proof

The agent module, working tree against the baseline commit:

```text
$ git diff -- app/rag/week9_agent.py
(empty: 0 lines)
```

Commit to commit, the agent module and the whole `app/` tree:

```text
$ git diff 486c83f ffdf62a -- app/rag/week9_agent.py
(empty: 0 lines)

$ git diff 486c83f ffdf62a -- app/
(empty: 0 lines)
```

Everything that changed:

```text
$ git diff 486c83f ffdf62a --stat
 config/mcp_servers.json | 5 +++++
 1 file changed, 5 insertions(+)
```

```diff
$ git diff 486c83f ffdf62a
diff --git a/config/mcp_servers.json b/config/mcp_servers.json
index 40453a2..968891e 100644
--- a/config/mcp_servers.json
+++ b/config/mcp_servers.json
@@ -4,6 +4,11 @@
       "command": ".venv/bin/python",
       "args": ["-m", "app.mcp.servers.docs_server"],
       "cwd": "."
+    },
+    "package-registry": {
+      "command": ".venv/bin/python",
+      "args": ["-m", "app.mcp.servers.package_server"],
+      "cwd": "."
     }
   }
 }
```

Nothing is hidden. The index still holds nine staged changes that predate
Week 9 and belong to neither commit: `.claude/settings.json`, `.gitignore`,
three `corpus/github_swagger/*.pdf`, `report/Gen-AI-Dev-RAG-Report.pdf`,
`report/build_report.py`, a `sample-corpus/` file and `traces/traces.jsonl`.
Both commits were made by explicit path, so neither includes them. The evidence
files in `report/w9/step6_*` are new, untracked output files, not source.

## 5. Runtime proof

### The same command, run twice

The agent CLI accepts only Week 7 question ids, the Week 7 question set has no
package question, and that set must not be edited. So both runs call the
unchanged `run_agent()` through this command, character for character. Only the
label changes, `A` or `B`:

```python
# Run ONE question through the unchanged Week 9 agent, using whatever servers
# config/mcp_servers.json lists. Usage: .venv/bin/python - <label> < this file
import sys
from pathlib import Path

from app.mcp.client import McpClient, configure_logging, load_config
from app.rag.week7_contract import score_output
from app.rag.week7_questions import Week7Question
from app.rag.week9_agent import render_run, run_agent

label = sys.argv[1]
configure_logging()
question = Week7Question(
    id="w9-pkg-01",
    question="Is release 2.1.0 of the example-sdk package deprecated, and if so, "
             "which release should we install instead?",
    question_class="package_registry",
    sdk_version="none",
    must_contain=["deprecated", "3.2.0"],
)
with McpClient(load_config()) as client:
    run = run_agent(question, client)
out = Path("report/w9")
(out / f"step6_{label}_run.json").write_text(run.model_dump_json(indent=2) + "\n")
client.wire.write_jsonl(out / f"step6_{label}_mcp_messages.jsonl")
print(render_run(run, question, score_output(question, run.output, run.tool_calls)))
```

It names no tool and no server. `load_config()` reads the default
`config/mcp_servers.json`, the same file the agent CLI reads.

| | config A (docs only) | config B (docs + package-registry) |
|---|---|---|
| tools discovered | 1 | 3 |
| LLM's tool path | `search_docs`, `search_docs` | `get_package_deprecation` |
| `tools/call` routed to | docs-search (twice) | package-registry (once) |
| answer | `INSUFFICIENT_CONTEXT` (correct refusal: no tool holds release data) | "Yes, release 2.1.0 of the example‑sdk package is deprecated, and the recommended replacement is release 3.2.0." |
| score | FAIL | PASS |
| tokens | 4,637 | 3,304 |

### The package-registry exchange in run B (from `step6_B_mcp_messages.jsonl`)

```text
#6  client->server  initialize                 protocolVersion 2025-06-18, clientInfo gen-ai-dev-week9 0.1.0
#7  server->client  result id=0                serverInfo package-registry 0.1.0, capabilities: resources, tools
#8  client->server  notifications/initialized
#9  client->server  tools/list      id=1
#10 server->client  result id=1                tools: get_package_versions, get_package_deprecation
#11 client->server  tools/call      id=2       get_package_deprecation {"package": "example-sdk", "version": "2.1.0"}
#12 server->client  result id=2                isError: false
                    structuredContent: {"package": "example-sdk", "version": "2.1.0", "found": true,
                      "deprecated": true, "deprecated_on": "2026-01-31",
                      "notice": "The 2.x line reached end of life on 2026-01-31 and receives no further fixes.",
                      "replacement": "3.2.0"}
```

The agent's log for run B shows who made each decision:

```text
[week9.mcp]   server 'package-registry': tools/list returned 2 tool(s): get_package_versions, get_package_deprecation
[week9.mcp]   discovered 3 tool(s) across 2 server(s): search_docs@docs-search, get_package_versions@package-registry, get_package_deprecation@package-registry
[week9.agent] lap 1: calling the LLM with 3 discovered tool(s): search_docs, get_package_versions, get_package_deprecation
[week9.agent] lap 1: LLM selected tool get_package_deprecation with arguments {"package":"example-sdk","version":"2.1.0"}
[week9.mcp]   tools/call get_package_deprecation -> server 'package-registry'
[week9.mcp]   tools/call get_package_deprecation <- server 'package-registry': ok in 114 ms
[week9.agent] lap 2: final answer generated (the LLM requested no further tool)
```

## 6. Why adding the server needed no agent change

```text
config/mcp_servers.json        which servers to start. Nothing about tools.
        |
McpClient                      starts one stdio process per entry
        |
initialize                     per server
        |
tools/list                     per server: name, description, inputSchema
        |
runtime discovery              tools merged, name -> server routing table built,
        |                      duplicate names refused
        |
agent: client.openai_tools()   the ONLY source of the tools sent to the LLM
        |
LLM picks a tool   ->   client.call_tool(name, args)   ->   tools/call on the owning server
```

The agent contains no tool name and no server name. It asks the connected
client for whatever tools discovery found, and hands the model's tool choice
back to the client, which routes it by name. The agent learned about
`get_package_versions` and `get_package_deprecation` at runtime, from the
package-registry server's `tools/list` response. Nothing about them was added
to the agent.

## Notes

- The package-registry server also lists one resource,
  `registry://example-sdk/support`, the support matrix. The client does not read
  resources, so this run did not use it. Its tools alone were discovered and
  called.
- Run B's model put the tool name `get_package_deprecation` on its `CITATIONS`
  line. The Week 7 answer contract asks for chunk ids or operationIds, and a
  package result carries neither. The scorer checks only chunk-id citations, so
  this did not affect the PASS.
- Stand-in data: every package fact above comes from `registry/packages.yaml`,
  locally authored for this experiment. See `registry/README.md`.
