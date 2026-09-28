# Week 9 Step 7: the MCP wire, captured

`wire.json` holds every JSON-RPC message from one real run of the Week 9 agent,
answering question `w9-pkg-01` with both servers configured (commit `ffdf62a`).
The messages are copied unchanged from what the client recorded, and
annotations sit beside each message, never inside it. This file explains them.
Every value quoted below is read from the capture.

> **The model call happens in the agent, via week7_contract.chat_once as an HTTPS request to Groq's OpenAI-compatible API; it never happens on this wire, and never inside either MCP server.**

## 1. How it was captured, and why it can be trusted

- **Mechanism:** `app/mcp/client.py`, unchanged since Step 4. `_WireTap` wraps the
  two streams between the MCP SDK's `ClientSession` and its stdio transport, and
  `WireLog` records each message there, in both directions, per server. No source
  file changed in Step 7.
- **Fidelity:** a scratch byte relay was placed between the client and each server
  on the same system, and it recorded the actual stdin/stdout lines.
  **12/12 messages were byte-identical** to what the client recorded, in both
  directions and for both servers. So `json.dumps(message, separators=(",", ":"),
  ensure_ascii=False)` reproduces the exact line that crossed the pipe. The relay
  was used once, with a separate config, and is not part of the repo.
- **Nothing added, nothing dropped:** the capture has 12 messages and
  `wire.json` has 12. Each `message` in `wire.json` equals the
  corresponding line of `step7_mcp_messages.jsonl`, and this was checked when the
  file was written.
- **The run it came from** (`step7_run.json`): the agent answered "Yes, release 2.1.0 of the example‑sdk package is deprecated, and the recommended replacement is release 3.2.0."
  in 2 LLM calls and 1 MCP tool call(s), and scored PASS.

Reproduce: run the capture command in section 9 and compare with
`step7_mcp_messages.jsonl`.

## 2. The actual sequence

| # | t (ms) | connection | direction | phase | id | top-level fields |
|---|---|---|---|---|---|---|
| 1 | 56.3 | docs-search | client->server | `initialize (request)` | 0 | `method`, `params`, `jsonrpc`, `id` |
| 2 | 2190.8 | docs-search | server->client | `initialize (response)` | 0 | `jsonrpc`, `id`, `result` |
| 3 | 2205.7 | docs-search | client->server | `notifications/initialized (notification)` | (none) | `method`, `jsonrpc` |
| 4 | 2213.8 | docs-search | client->server | `tools/list (request)` | 1 | `method`, `jsonrpc`, `id` |
| 5 | 2222.3 | docs-search | server->client | `tools/list (response)` | 1 | `jsonrpc`, `id`, `result` |
| 6 | 2413.4 | package-registry | client->server | `initialize (request)` | 0 | `method`, `params`, `jsonrpc`, `id` |
| 7 | 3285.1 | package-registry | server->client | `initialize (response)` | 0 | `jsonrpc`, `id`, `result` |
| 8 | 3287.7 | package-registry | client->server | `notifications/initialized (notification)` | (none) | `method`, `jsonrpc` |
| 9 | 3295.8 | package-registry | client->server | `tools/list (request)` | 1 | `method`, `jsonrpc`, `id` |
| 10 | 3301.0 | package-registry | server->client | `tools/list (response)` | 1 | `jsonrpc`, `id`, `result` |
| 11 | 9099.8 | package-registry | client->server | `tools/call (request)` | 2 | `method`, `params`, `jsonrpc`, `id` |
| 12 | 9248.3 | package-registry | server->client | `tools/call (response)` | 2 | `jsonrpc`, `id`, `result` |

The client opens the connections one after the other, docs-search then
package-registry, and runs the same handshake on each: `initialize`, its
response, `notifications/initialized`, then `tools/list` and its response. Only
package-registry receives a `tools/call`, because that is the tool the LLM chose.
No message follows #12. MCP over stdio has no shutdown request: the
client ends each session by closing the server's stdin.

## 3. The top-level fields

| field | present in | meaning |
|---|---|---|
| `jsonrpc` | all 12 messages | Always `"2.0"`. It marks the line as a JSON-RPC 2.0 message. |
| `id` | requests and responses, never the notifications | Pairs a response with its request. The client numbers its requests 0, 1, 2 … **per connection**, so both servers see `initialize` as id 0. A message without an `id` is a notification. |
| `method` | requests and notifications | The operation: `initialize`, `notifications/initialized`, `tools/list`, `tools/call`. Responses carry no method; the `id` says what they answer. |
| `params` | `initialize` and `tools/call` requests | The operation's arguments. `tools/list` was sent **without** `params`: first page, no cursor, and the SDK omits empty fields. |
| `result` | every response | The successful answer. Its shape depends on the method it answers. |
| `error` | **no message in this capture** | In JSON-RPC, a failed request is answered with `error` (`code`, `message`) instead of `result`. Nothing failed at protocol level here. In MCP, a failing *tool* is not a protocol error either: it comes back as a normal `result` with `isError: true`. So even a tool failure would not have produced an `error` field. |

## 4. The four phases, on the package-registry connection

### 4.1 `initialize`

Sent by the MCP client (the agent's host process) to the package-registry server,
as the first message on the connection. It is needed because the two sides must
agree on a protocol version and learn each other's capabilities before anything
else. The server may not be asked for tools until this is done.

Request, #6, client → server:
```json
{
  "method": "initialize",
  "params": {
    "protocolVersion": "2025-06-18",
    "capabilities": {},
    "clientInfo": {
      "name": "gen-ai-dev-week9",
      "version": "0.1.0"
    }
  },
  "jsonrpc": "2.0",
  "id": 0
}
```
- `protocolVersion`: the client proposes `2025-06-18`.
- `capabilities: {}`: the client offers the server **nothing**: no `sampling`, so
  the server cannot ask this host's LLM for anything, and no `roots` and no
  `elicitation`.
- `clientInfo`: identifies the host.

Response, #7, server → client, same `id`:
```json
{
  "jsonrpc": "2.0",
  "id": 0,
  "result": {
    "protocolVersion": "2025-06-18",
    "capabilities": {
      "experimental": {},
      "resources": {
        "subscribe": false,
        "listChanged": false
      },
      "tools": {
        "listChanged": false
      }
    },
    "serverInfo": {
      "name": "package-registry",
      "version": "0.1.0"
    },
    "instructions": "Stand-in package registry for the Week 9 experiment. Its data is locally authored, not retrieved from any real package registry, and it is not an official third-party server."
  }
}
```
- The server agrees to `2025-06-18`.
- It declares `tools` (it answers `tools/list` and `tools/call`) and `resources`.
  `listChanged: false` means it will never push a changed list.
- It identifies itself as `package-registry 0.1.0`.
- `instructions` carries its stand-in disclosure.

### 4.2 `notifications/initialized`

#8, client → server:
```json
{
  "method": "notifications/initialized",
  "jsonrpc": "2.0"
}
```
This closes the handshake: the client has accepted the server's answer, and
normal requests may begin. It is a **notification** because it carries no `id`.
JSON-RPC defines such a message as one the receiver must not answer, and the
capture shows no response to it: the next message is the client's own
`tools/list`. In the lifecycle it sits after the `initialize` response and before
the first ordinary request.

### 4.3 `tools/list`

Request, #9, client → server:
```json
{
  "method": "tools/list",
  "jsonrpc": "2.0",
  "id": 1
}
```
The client asks which tools the server exposes. It sends no `params`.

Response, #10, server → client:
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "tools": [
      {
        "name": "get_package_versions",
        "description": "List every published release of one package in the package registry, with the date each release was published, and name the latest release.\n\nUse it for questions about a package as something you install: which versions exist, which is the newest, when a release came out, or which release to pin.\n\npackage is the package's registry name, the name you would install it by, not an SDK class or method name. Case and the separators - _ . do not matter.\n\nReturns JSON: package, found, summary, latest, and releases oldest first, each with version and released (YYYY-MM-DD). An unknown package comes back as found: false, with a reason naming the packages the registry holds.\n\nA release number such as 1.4.2 is a package version: one published build. It is not an API or documentation version such as v3, though a package's major number may track one. This tool does not say whether a release is deprecated; get_package_deprecation answers that for one release. It says nothing about what SDK methods do, their parameters or defaults, or HTTP operations and endpoints, and it has no changelog or release notes.",
        "inputSchema": {
          "type": "object",
          "properties": {
            "package": {
              "type": "string",
              "minLength": 1,
              "description": "Registry name of the package: the name it is installed by."
            }
          },
          "required": [
            "package"
          ],
          "additionalProperties": false
        }
      },
      {
        "name": "get_package_deprecation",
        "description": "Report whether one published release of a package has been deprecated on the package registry by its maintainers, meaning that build should no longer be installed. If it has, also return the maintainers' notice, the date, and the newest release that is not deprecated.\n\nUse it for questions like \"is release 1.4.2 of this package deprecated?\" or \"should we still pin that release?\". This is package deprecation: it is about which build of a package to install.\n\npackage is the registry name, as for get_package_versions. version is one exact release such as 1.4.2; a leading v is accepted. A release line such as 1.x or v1 is not a release: it comes back as found: false, listing the releases in that line. A release that was never published comes back as found: false, naming the latest published release. In both cases the registry answered; the release simply does not exist.\n\nReturns JSON: package, version, found, deprecated, and when deprecated also deprecated_on, notice and replacement.\n\nIt does not answer whether an SDK method, a parameter, or an HTTP API operation or endpoint is deprecated or retired. A deprecated release says nothing about any one operation. It has no changelog or release notes.",
        "inputSchema": {
          "type": "object",
          "properties": {
            "package": {
              "type": "string",
              "minLength": 1,
              "description": "Registry name of the package: the name it is installed by."
            },
            "version": {
              "type": "string",
              "minLength": 1,
              "description": "One exact published release, e.g. 1.4.2. Not a release line such as 1.x, and not an API version such as v3."
            }
          },
          "required": [
            "package",
            "version"
          ],
          "additionalProperties": false
        }
      }
    ]
  }
}
```
The server returns **2** tool definitions:
`get_package_versions`, `get_package_deprecation`. For each tool:
- **name**: the identifier the model will call it by.
- **description**: the text the LLM reads when deciding which tool to use.
- **inputSchema**: a JSON Schema its arguments must satisfy.

All three are defined in the server's own code (`app/mcp/servers/package_server.py`)
and reach the agent **only through this message**. There is no `nextCursor`, so
this one page is the complete list.

**The agent's tool inventory comes from this live response.** The client converts
each entry to an OpenAI function tool, and the agent sends those to the LLM. The
`tools_offered` field in `step7_run.json` is exactly the union of the two
`tools/list` results in this capture, compared field by field. No tool name
appears in `week9_agent.py`.

### 4.4 `tools/call`

Request, #11, client → server:
```json
{
  "method": "tools/call",
  "params": {
    "name": "get_package_deprecation",
    "arguments": {
      "package": "example-sdk",
      "version": "2.1.0"
    }
  },
  "jsonrpc": "2.0",
  "id": 2
}
```
The client sends it after the LLM has chosen a tool.
- **Which tool:** `params.name` = `get_package_deprecation`, chosen by the LLM.
- **Which arguments:** `params.arguments` = `{"package": "example-sdk", "version": "2.1.0"}`,
  the LLM's own arguments, forwarded unchanged.
- **Where it runs:** the client routed it to **package-registry**, because that
  server listed the tool in #10.
- **Who executes it:** the server executes it in its own process, by looking the
  release up in `registry/packages.yaml`.

Response, #12, server → client:
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{\n  \"package\": \"example-sdk\",\n  \"version\": \"2.1.0\",\n  \"found\": true,\n  \"deprecated\": true,\n  \"deprecated_on\": \"2026-01-31\",\n  \"notice\": \"The 2.x line reached end of life on 2026-01-31 and receives no further fixes.\",\n  \"replacement\": \"3.2.0\",\n  \"provenance\": \"locally authored stand-in package registry data for the Week 9 experiment; not retrieved from any real package registry\"\n}"
      }
    ],
    "structuredContent": {
      "package": "example-sdk",
      "version": "2.1.0",
      "found": true,
      "deprecated": true,
      "deprecated_on": "2026-01-31",
      "notice": "The 2.x line reached end of life on 2026-01-31 and receives no further fixes.",
      "replacement": "3.2.0",
      "provenance": "locally authored stand-in package registry data for the Week 9 experiment; not retrieved from any real package registry"
    },
    "isError": false
  }
}
```
- `isError: false`: the tool ran and answered.
- `structuredContent`: release `2.1.0` of `example-sdk` is
  deprecated (`deprecated: true`) since `2026-01-31`,
  with replacement `3.2.0`.
- `content`: the same result as JSON text, for clients that read only text.

The client handed that object to the agent, the agent appended it to the
conversation, and the LLM wrote the final answer from it. That last LLM call is
not on this wire.

## 5. MCP JSON-RPC vs the LLM API

```text
MCP JSON-RPC                                   LLM API
------------                                   -------
week9_agent -> McpClient                       week9_agent
      | stdio pipe, one JSON line per message        | HTTPS, OpenAI-compatible chat.completions
      v                                              v
MCP server subprocess                          Groq LLM
(docs-search / package-registry)               (week7_contract.chat_once, week7_contract.py:411)
```

- **`initialize`, `tools/list` and `tools/call` are MCP protocol operations
  between the MCP client and the MCP servers.** Every one of them is in this
  capture.
- **The LLM call is separate, and no MCP server calls the LLM.** Neither server
  imports an LLM client, an HTTP client, or any agent module. Neither receives
  `GROQ_API_KEY`: the client starts servers with only the SDK's minimal
  environment. And the client offers no `sampling` capability (#6
  above), so a server has no protocol route to the host's model either.

The model calls appear in the capture as silence:

| LLM call | where it falls in the capture | MCP silence | LLM latency (run record) | outcome |
|---|---|---|---|---|
| lap 1 | between #10 (tools/list (response), 3301.0 ms) and #11 (tools/call (request), 9099.8 ms) | 5798.8 ms | 5711.973 ms | the model asked for ['get_package_deprecation'] |
| lap 2 | after #12 (tools/call (response), 9248.3 ms); no MCP message follows | n/a | 849.287 ms | the model wrote the final answer |

The 5798.8 ms gap before the `tools/call` is the lap-1 model call
(5711.973 ms) plus 87 ms of work in the
agent process: building the request, parsing the reply, and handing the call to
the client. The wire is quiet while the model decides.

## 6. Where each operation occurs

| Operation | Sender | Receiver | Transport | Purpose |
|---|---|---|---|---|
| `initialize` | MCP client (`client.py`, `session.initialize()`) | each MCP server | JSON-RPC over stdio | agree on a protocol version and exchange capabilities |
| `notifications/initialized` | MCP client | each MCP server | JSON-RPC over stdio, no response | complete the handshake |
| `tools/list` | MCP client (`session.list_tools()`) | each MCP server | JSON-RPC over stdio | discover the server's tools: names, descriptions, schemas |
| `tools/call` | MCP client (`session.call_tool()`) | the server that listed the tool | JSON-RPC over stdio | execute the tool the LLM chose |
| LLM request | Week 9 agent (`chat_once`) | Groq LLM API | HTTPS, OpenAI-compatible, **not JSON-RPC and not on this wire** | have the model choose a tool or write the answer |

## 7. Tool counts, from the captured `tools/list` results

| server | count | tools |
|---|---|---|
| docs-search | 1 | `search_docs` |
| package-registry | 2 | `get_package_versions`, `get_package_deprecation` |
| **total** | **3** | |

These are computed from the `tools/list` responses in `wire.json`. They match
the 3 tools the agent offered the LLM in this run.

## 8. Validation and privacy

- `wire.json` parses as JSON: `python -m json.tool report/w9/wire.json`.
- 12 messages, none truncated and none invented, each equal to the
  capture line. The method names, ids, tool names, arguments and results are the
  captured ones. The `tools/call` name and arguments equal the run record's tool
  path and arguments.
- **Redactions: none were needed.** The capture and the run record were scanned
  before anything was written:

| check | result |
|---|---|
| GROQ_API_KEY value | not present |
| gsk_ token shape | not present |
| sk- token shape | not present |
| Bearer / Authorization | not present |
| password / secret | not present |
| home-directory path | not present |

The LLM API key never enters the MCP path. It is used only by `chat_once` for the
HTTPS call, and servers are started without it.

## 9. Capture command

The unchanged Week 9 agent, the servers in `config/mcp_servers.json`, one
question:

```python
# Step 7 capture: one question through the unchanged Week 9 agent, using the
# servers config/mcp_servers.json lists; keeps every JSON-RPC message.
from pathlib import Path

from app.mcp.client import McpClient, configure_logging, load_config
from app.rag.week7_contract import score_output
from app.rag.week7_questions import Week7Question
from app.rag.week9_agent import render_run, run_agent

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
Path("report/w9/step7_run.json").write_text(run.model_dump_json(indent=2) + "\n")
client.wire.write_jsonl(Path("report/w9/step7_mcp_messages.jsonl"))
print(render_run(run, question, score_output(question, run.output, run.tool_calls)))
```

Run from the repo root as `.venv/bin/python - < capture.py`. `wire.json` is then
assembled from `step7_mcp_messages.jsonl` by adding `phase` and `annotations`
beside each message.
