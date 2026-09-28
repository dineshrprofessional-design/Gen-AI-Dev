# Week 9 interview notes: MCP

### 1. What is MCP?
The Model Context Protocol is a standard way for an AI application to find and use tools. Each
tool lives in a server that describes itself. The application connects, asks the server what it
offers, and calls it, all as JSON-RPC 2.0 messages. Here those messages travel over a child
process's stdin and stdout.

### 2. Why MCP instead of importing tools into the agent?
An imported tool is compiled into the agent, so every new tool means an agent code change and a
release. With MCP the agent only knows a config file listing servers; the tools themselves are
discovered at runtime. We added a second server by editing the config alone.

### 3. What does an MCP server do?
It exposes capabilities. It answers `tools/list` with each tool's name, description and input
schema, and it answers `tools/call` by running the tool. Ours are thin adapters:
- docs-search wraps the Week 7 RAG search
- package-registry reads a local data file

A server holds no model and makes no decisions.

### 4. What does an MCP client do?
It lives in the host: starting each configured server, running the handshake and discovery,
merging every server's tools into one list, remembering which server owns which tool, and routing
each call there. Ours also:
- converts MCP tools into the OpenAI tool format the LLM expects
- refuses two servers that expose the same tool name
- records every JSON-RPC message

### 5. What happens during `initialize`?
It is the first request on every connection.
- The client proposes a protocol version (`2025-06-18`) and states its capabilities. Ours offers
  none, so a server cannot borrow the host's model.
- The server replies with the version it agrees to, what it offers (for example `tools`), and its
  name and version.
- The client then sends `notifications/initialized`, a message with no `id` that gets no reply,
  and ordinary requests may begin.

### 6. What happens during `tools/list`?
The client asks, "what tools do you have?" The server returns each tool's name, description and
JSON Schema for its arguments. That response is the only place the agent learns about tools:
docs-search listed 1 tool and package-registry listed 2, 3 in total.

### 7. What happens during `tools/call`?
When the LLM picks a tool, the client sends `tools/call` with the tool name and the LLM's
arguments to the server that listed it. The server runs the tool and returns `content` (text),
`structuredContent` (JSON) and `isError`. A tool failure comes back as `isError: true` inside a
normal result, not as a protocol error, so the model can read it.

### 8. Where does the LLM call happen?
Only in the agent: `week7_contract.chat_once`, an HTTPS request to Groq's OpenAI-compatible API.
Never on the MCP wire, and never inside a server.

### 9. Why should the MCP server not contain the LLM?
The server offers capabilities, and the host owns the model and the reasoning. Keeping them apart
lets servers be shared and swapped, keeps the model's keys and cost in one place, and keeps
tools deterministic and testable. A server that calls a model is an agent pretending to be a tool.

### 10. How did we prove dynamic tool discovery?
We ran the same agent on the same question with two configs:
- **Docs server only:** it discovered 1 tool and refused.
- **After adding the package-registry entry:** it discovered 3 tools, called
  `get_package_deprecation` by itself, and answered correctly.

Every count came from the live `tools/list` responses.

### 11. How did we prove the agent code did not need to change?
The two configs are two commits, `486c83f` and `ffdf62a`.
- `git diff` between them touches only `config/mcp_servers.json`: 5 lines added.
- The agent file has the same git blob and the same SHA-256 at both (`agent_diff.txt`).
- The agent source contains no tool name and no server name.

### 12. What is the difference between MCP JSON-RPC and the LLM API call?

| | MCP JSON-RPC | LLM API |
|---|---|---|
| between | client and server | agent and the model provider |
| travels over | a local pipe | HTTPS |
| messages | `initialize`, `tools/list`, `tools/call` | a chat completion |
| carries | tool definitions and results | prompts |

In the capture, the model call shows up as a 5.8 s silence on the MCP wire between `tools/list`
and `tools/call`.

### 13. Why are tool descriptions important?
The description is the only thing the LLM reads when choosing a tool and filling in its
arguments, and it arrives through `tools/list`. Ours says:
- what the corpus holds
- how to pass a version, including release numbers
- that a search is never empty
- what the tool cannot answer, and that package questions belong elsewhere

A vague description leads to wrong tool choices and wrong arguments.

### 14. Why is recoverable error information useful for an agent?
The model can only fix what it can understand. The old error, `'v4' is not one of ['v2', 'v3']`,
does not say whether v4 is missing or misspelled. The new one says v4 is not documented, v3 is
the newest, and not to answer from another version. So the model can correct itself or refuse
correctly.

One honest caveat: our model never actually sent a bad version, because the schema's enum
prevented it. The diagnostic protects clients that ignore the schema.

### 15. What security or trust-boundary concerns did we find?
- Every server runs inside the agent's trust boundary, and its descriptions and results go
  straight into the model's prompt, so a bad server can steer the model.
- Servers start with a minimal environment and no Groq key. But they are not sandboxed: they run
  as the user and could read `.env` from disk.
- The docs server contacts `huggingface.co` when it loads its model, and would send the stored
  Hugging Face token. The package server makes no network calls.
- Nothing verifies server code before launch.
- Verdict for the package server: **DO NOT SHIP**. It serves stand-in data, runs unsandboxed, and
  nothing pins it.

## Week 9 learning flow

```text
MCP Server
    ↓
Expose capabilities as tools
    ↓
MCP Client
    ↓
initialize
    ↓
tools/list
    ↓
Dynamic tool discovery
    ↓
LLM chooses tool
    ↓
tools/call
    ↓
MCP Server executes
    ↓
Result
    ↓
LLM final answer
```

```text
Core lesson:

MCP separates the Agent from the implementation of tools and allows
tool capabilities to be discovered and used through a standard protocol.
```
