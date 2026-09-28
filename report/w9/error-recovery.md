# Week 9 Step 8: recoverable error, before and after

This covers the Week 9 requirement to rewrite one tool docstring on our own server as a prompt, make
one of its error paths recoverable, and show a before/after transcript. The brief calls this file
`error_before_after.md`.

**The error path.** `search_docs` on the docs-search server, when the request carries an
`api_version` the corpus does not document.

**The change.** `app/mcp/servers/docs_server.py` only.
- The tool description was rewritten as a guidance prompt (section 4).
- Arguments are now validated in the adapter, against the same schema: `@server.call_tool(validate_input=False)`
  plus `jsonschema`. An invalid `api_version` gets a diagnostic built from the corpus.
- Every other schema error keeps the SDK's own wording.
- The search call, its arguments, its result shape and the input schema are unchanged.

Everything below comes from two captures of the same probe and the same questions, run against the
committed server (**before**, `ffdf62a`) and the changed server (**after**):
`step8_before.json`, `step8_after.json`, `step8_before_provoked.json` and `step8_after_provoked.json`.

## 1. Before and after on the wire (real MCP `tools/call`)

Before the change, the SDK rejected the value against the schema's enum and returned its generic
message. That message says the value is wrong, but not what kind of mistake it is or what to do next.
In particular, "v4 does not exist" and "3.2.0 is spelled wrongly" look identical.

### `api_version: "v4"`: version that is not documented

**Input** (`tools/call`, identical before and after):
```json
{
  "name": "search_docs",
  "arguments": {
    "query": "default timeout_ms of Client.send",
    "api_version": "v4"
  }
}
```

**Before**: `isError: true`
```text
Input validation error: 'v4' is not one of ['v2', 'v3']
```

**After**: `isError: true`, diagnostic:
```text
No documentation exists for api_version "v4". This corpus documents "v2" and "v3" only, and v3 is the newest. The search did not fail: that version is simply not documented. If the question is about "v4", say the documentation does not cover it, and do not answer with another version's values, because defaults differ between versions. If the question is really about the current SDK, call search_docs again with api_version "v3".
```

### `api_version: "3.2.0"`: package release number instead of a documentation version

**Input** (`tools/call`, identical before and after):
```json
{
  "name": "search_docs",
  "arguments": {
    "query": "default timeout_ms of Client.send",
    "api_version": "3.2.0"
  }
}
```

**Before**: `isError: true`
```text
Input validation error: '3.2.0' is not one of ['v2', 'v3']
```

**After**: `isError: true`, diagnostic:
```text
api_version "3.2.0" is not a documentation version. Documentation is published once per major SDK version, as "v2" and "v3", and "3.2.0" belongs to major version 3. Call search_docs again with api_version "v3".
```

### `api_version: "latest"`: not a version at all

**Input** (`tools/call`, identical before and after):
```json
{
  "name": "search_docs",
  "arguments": {
    "query": "default timeout_ms of Client.send",
    "api_version": "latest"
  }
}
```

**Before**: `isError: true`
```text
Input validation error: 'latest' is not one of ['v2', 'v3']
```

**After**: `isError: true`, diagnostic:
```text
api_version "latest" is not a version name. Pass one of "v2" or "v3"; v3 is the newest. A package release number maps to its major version: any 3.x.y release is "v3". To cover more than one version, call once per version.
```

The three diagnostics call for three different recoveries, and only one of them is "retry":

| input | kind of mistake | what the diagnostic tells the model |
|---|---|---|
| `"v4"` | a version with no documentation | it is not documented, `v3` is the newest, do not answer from another version's values; retry with `v3` only if the question is really about the current SDK |
| `"3.2.0"` | a package release number, not a documentation version | it belongs to major version 3; retry with `"v3"` |
| `"latest"` | not a version at all | the accepted values, which one is newest, and how a release number maps |

Every version a diagnostic names comes from `docs/` itself (`week7_corpus.load_pages`). The server
refuses to start if that list and the schema's enum ever disagree. No version is invented.

## 2. The corrected request

**Corrected request.** Every diagnostic above ends by naming the same
correction, `api_version "v3"`:
```json
{
  "name": "search_docs",
  "arguments": {
    "query": "default timeout_ms of Client.send",
    "api_version": "v3"
  }
}
```

**Result** (`isError: false`): `count` 5, passages
`v3/client-send#parameters::1`, `v3/client-send#notes::3`, `v3/client-batch#parameters::1`, `v3/client-send#clientsend::0`, `v3/client-batch#notes::3`. The top passage,
`v3/client-send#parameters::1` (Client.send() > Parameters), begins:
```text
Client.send() > Parameters

## Parameters

| name             | type | default   | required | description                                          |
|------------------|------|-----------|----------|------------------------------------------------------|
| payload          | dict | —         | yes      | The request body, serialised as JSON before sending. |
| timeout_ms       | int  | 30000     | no       | Total bu …
```
This result is **identical** before and after, compared as whole JSON objects:
true.
Only the error text changed; the search did not.

## 3. The real model, before and after

The same three questions went to the real agent and LLM (gpt-oss-120b on Groq) in both phases. Each
question tells the model which bad `api_version` to use, so that the failing call would be made:

- `w9-err-v4`: "Search the documentation with api_version "v4" and tell me the default timeout_ms of Client.send() in v4."
- `w9-err-release`: "Our lockfile pins example-sdk 3.2.0. Search the documentation with api_version "3.2.0" and tell me the default max_retries of Client.send() for that release."
- `w9-err-provoked`: "Call search_docs with api_version set to exactly "v4" and query "default timeout_ms of Client.send", then tell me the default timeout_ms of Client.send() in v4."

| question | phase | api_version the model sent | answer | score |
|---|---|---|---|---|
| `w9-err-v4` | before | (no tool call) | INSUFFICIENT_CONTEXT | PASS |
| `w9-err-v4` | after | (no tool call) | INSUFFICIENT_CONTEXT | PASS |
| `w9-err-release` | before | v3 | The default max_retries for Client.send() is 3. | FAIL |
| `w9-err-release` | after | v3 | The default max_retries for Client.send() is 3. | FAIL |
| `w9-err-provoked` | before | (no tool call) | INSUFFICIENT_CONTEXT | PASS |
| `w9-err-provoked` | after | (no tool call) | INSUFFICIENT_CONTEXT | PASS |

**Finding: the model never sent an invalid `api_version`, in either phase, even when told to.** The
schema's enum (`["v2", "v3"]`) was enough.
- Asked about v4, it called no tool and refused.
- Given a release number, it chose `"v3"` on its own.
- Told to call the tool with exactly `"v4"`, it still refused.

So **there is no real transcript of this model handling the failing call, and none was fabricated.**
The `version_named` FAIL on `w9-err-release` is the Week 7 scorer wanting "v3" in the answer prose.
It happens in both phases and is unrelated to this change.

What that means for the change:
- The enum stays. It is the first line of defence, and it works.
- The diagnostic is the second line, for any MCP client that does not honour the schema: another
  host, another model, a hand-written client, or a gateway. For them the before and after on the
  wire (section 1) is the whole story.
- Removing the enum just to provoke a transcript would have made the tool worse, so it was not done.

## 4. The tool description, as the client receives it from `tools/list`

**Before** (Step 3's description, live `tools/list` in `step8_before.json`, 1423 chars):
```text
Search the SDK reference documentation for prose describing what a Client method does: its behaviour, its parameter names, types and defaults, whether a parameter is required, and the caveats written in the Notes sections.

Use it when the question is about how a Client method behaves, or what one of its parameters means, defaults to, or requires.

api_version selects which version's reference pages are searched: "v2" or "v3". Only that version is searched, never both. The two versions document different defaults for the same parameter, and some methods and parameters exist only in v3. If the question names a version, pass it. If it asks to compare versions, or does not say and the answer could differ, call once per version.

query is what to look for, in words: the method and the parameter or behaviour, or just a parameter name. Leave the version out of the query; api_version already filters by it.

Returns JSON: api_version, count, and passages, best match first, each with rank, chunk_id, heading_path, score and text. Passages are ranked by similarity, not filtered for relevance, so the top one can still be off-topic: read the text before relying on it, and cite the chunk_id of any passage you use.

This tool knows nothing about HTTP paths, operation ids, or whether anything is retired, and it has no changelog or release numbers. If no passage states the answer, the documentation does not state it.
```

**After** (live `tools/list`, raw JSON-RPC message #5 in `step8_after.json` →
`probe_wire`, 2435 chars):
```text
Search the SDK's reference documentation and return the passages that best match a query. The documentation describes what each Client method does: its behaviour, its parameters with their types and defaults and whether each is required, and the caveats in its Notes.

Use it for questions about how a Client method behaves, or what one of its parameters means, defaults to, or requires.

What the corpus holds: one reference page per Client method per documented version, each with an overview, a parameters table and Notes, most with an Example:
  v2: Client.close, Client.configure, Client.send, Client.stream
  v3: Client.close, Client.configure, Client.send, Client.send_batch, Client.stream, Client.upload
A method not listed for a version is not documented for that version.

api_version: which documented version to search, "v2" or "v3"; v3 is the newest. It is the SDK's major version, not a package release number: for any release numbered 3.x.y, pass "v3". Only that version's pages are searched. The versions document different defaults for the same parameter, so pass the version the question is about. To compare versions, or when the question does not say and the answer could differ, call once per version. Any other value returns an error that names the documented versions: correct api_version and call again.

query: what to look for, in words: the method and the parameter or behaviour, or just a parameter name. Leave the version out of the query; api_version already filters by it.

Returns JSON: api_version, count, and passages, best match first, each with rank, chunk_id, heading_path, score and text. Passages are ranked by similarity, not filtered for relevance, so the top one can still be off-topic: read the text before relying on it, and cite the chunk_id of any passage you use. A search of a documented version always returns 5 passages and is never empty, so a missing answer shows up as passages that do not state it. If none states it, the documentation does not contain it: say so rather than filling the gap from memory.

What it cannot answer: HTTP paths, operation ids, whether an endpoint or operation is deprecated or retired, changelogs, or release history. Questions about the package itself (which releases exist, when one was published, whether a release is deprecated, which to install) are package-registry questions, not documentation questions: this corpus contains no release numbers.
```

What the rewrite adds:
- **What the corpus holds.** The method list per version is generated from `docs/`, so it cannot drift.
- **How a release number maps to `api_version`.**
- **A search is never empty.** It always returns 5 passages, so a missing answer has to be
  read from the passages.
- **The version error is recoverable**, and what to do with it.
- **Package-release questions belong elsewhere.** The description draws this line without naming the
  other server's tools, because servers stay independent.

The input schema is unchanged.

## 5. Does the agent loop recover on its own?

Yes, and without any retry logic of its own.
- An `isError` result is handed back to the model as a tool message, `{"error": "<diagnostic>"}`.
- The loop then calls the model again.
- Whether a corrected call follows is the model's decision, on its next lap, within the usual
  budgets (at most 8 iterations).

`python -m app.mcp.selfcheck` shows this plumbing with a **scripted** model through the agent's
`chat=` seam, against the real docs server over real MCP:
1. It sends `"3.2.0"` and gets the diagnostic back as a tool error.
2. It reads the version the diagnostic names and calls again with it.
3. That call is executed over MCP and succeeds.

The scripted model proves the loop delivers the diagnostic and executes the correction. It does
not show what a real model would choose. That is section 3.

## 6. Reproduce

```text
git checkout ffdf62a -- app/mcp/servers/docs_server.py   # the before server (restore afterwards)
.venv/bin/python -m app.mcp.selfcheck                     # after: 22 offline checks, incl. the probe above
.venv/bin/python -m app.mcp.client --call search_docs '{"query": "default timeout_ms of Client.send", "api_version": "v4"}'
```
