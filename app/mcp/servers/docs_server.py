"""Docs MCP server: Week 7's `search_docs`, exposed over MCP and nothing else.

    .venv/bin/python -m app.mcp.servers.docs_server      # run from the repo root

This is an adapter, not a second implementation. `week7_tools.search_docs` stays
the only place documentation is retrieved; this module only translates between
MCP and that function:

    tools/list  ->  one Tool: name, DESCRIPTION, the Week 7 input schema
    tools/call  ->  arguments checked against that schema; a bad api_version
                    gets a diagnostic naming the documented versions
                ->  week7_tools.search_docs(...)
                ->  the same dict, returned as an MCP tool result

The server holds no model and makes no model call. It never decides which tool
to use; the host's LLM does that from what `tools/list` returns.

The description and the version diagnostic are built from the corpus itself
(`week7_corpus.load_pages` over `docs/`), so they can only name versions and
methods that are actually documented. Import fails if the corpus and the Week 7
schema enum disagree about which versions exist.

Transport is stdio: the client launches this process and exchanges one JSON-RPC
message per line over stdin/stdout. Stdout therefore belongs to the protocol,
and anything else that writes there corrupts the stream.

Paths in `ToolConfig` are relative (`docs/`, `.rag_index/`), exactly as in
Week 7, so the process must start with the repo root as its working directory.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from typing import Any

import anyio
import jsonschema
import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from app.rag import week7_tools
from app.rag.week7_corpus import load_pages
from app.rag.week7_tools import ToolConfig, ToolContext

SERVER_NAME = "docs-search"
TOOL_NAME = "search_docs"

_cfg = ToolConfig()


# --------------------------------------------------------------------------
# what the corpus actually documents
# --------------------------------------------------------------------------


def _major(version: str) -> int:
    return int(version.lstrip("vV"))


def _documented_methods() -> dict[str, list[str]]:
    """Documented version -> the Client methods it has a page for, oldest version first."""
    methods: dict[str, list[str]] = {}
    for (version, _), page in load_pages(_cfg.docs_root).items():
        methods.setdefault(version, []).append(page.sdk_method)
    return {version: sorted(methods[version]) for version in sorted(methods, key=_major)}


def _quoted(values: list[str], joiner: str = "or") -> str:
    quoted = [f'"{value}"' for value in values]
    return quoted[0] if len(quoted) == 1 else ", ".join(quoted[:-1]) + f" {joiner} " + quoted[-1]


DOCUMENTED = _documented_methods()
VERSIONS = list(DOCUMENTED)
NEWEST = VERSIONS[-1]


def _week7_input_schema(name: str) -> dict[str, Any]:
    """The Week 7 parameter schema for one tool, reused rather than re-typed."""
    for schema in week7_tools.tool_schemas():
        if schema["function"]["name"] == name:
            return copy.deepcopy(schema["function"]["parameters"])
    raise LookupError(f"week7_tools.tool_schemas() has no tool named {name!r}")


INPUT_SCHEMA = _week7_input_schema(TOOL_NAME)
if INPUT_SCHEMA["properties"]["api_version"]["enum"] != VERSIONS:
    raise RuntimeError(
        f"schema enum {INPUT_SCHEMA['properties']['api_version']['enum']} disagrees with the "
        f"versions documented under {_cfg.docs_root}: {VERSIONS}"
    )


# --------------------------------------------------------------------------
# what the model reads
# --------------------------------------------------------------------------

# The description rewritten as a prompt (Week 9 requirement 5). The "before" is
# the version first published over MCP in Step 3 (commit 486c83f), which itself
# grew from week7_tools.SEARCH_DOCS_DESCRIPTION. What this adds: what the
# corpus holds, read from docs/ so it cannot drift; how a release number maps to
# api_version; that a search never comes back empty, so absence has to be read
# from the passages; what to do with the version error; and that package-release
# questions do not belong here.
DESCRIPTION = (
    "Search the SDK's reference documentation and return the passages that best "
    "match a query. The documentation describes what each Client method does: its "
    "behaviour, its parameters with their types and defaults and whether each is "
    "required, and the caveats in its Notes.\n\n"
    "Use it for questions about how a Client method behaves, or what one of its "
    "parameters means, defaults to, or requires.\n\n"
    "What the corpus holds: one reference page per Client method per documented "
    "version, each with an overview, a parameters table and Notes, most with an "
    "Example:\n"
    + "".join(f"  {version}: {', '.join(methods)}\n" for version, methods in DOCUMENTED.items())
    + "A method not listed for a version is not documented for that version.\n\n"
    f"api_version: which documented version to search, {_quoted(VERSIONS)}; {NEWEST} "
    "is the newest. It is the SDK's major version, not a package release number: "
    f"for any release numbered {_major(NEWEST)}.x.y, pass \"{NEWEST}\". Only that "
    "version's pages are searched. The versions document different defaults for "
    "the same parameter, so pass the version the question is about. To compare "
    "versions, or when the question does not say and the answer could differ, "
    "call once per version. Any other value returns an error that names the "
    "documented versions: correct api_version and call again.\n\n"
    "query: what to look for, in words: the method and the parameter or "
    "behaviour, or just a parameter name. Leave the version out of the query; "
    "api_version already filters by it.\n\n"
    "Returns JSON: api_version, count, and passages, best match first, each with "
    "rank, chunk_id, heading_path, score and text. Passages are ranked by "
    "similarity, not filtered for relevance, so the top one can still be "
    "off-topic: read the text before relying on it, and cite the chunk_id of any "
    f"passage you use. A search of a documented version always returns {_cfg.k} "
    "passages and is never empty, so a missing answer shows up as passages that "
    "do not state it. If none states it, the documentation does not contain it: "
    "say so rather than filling the gap from memory.\n\n"
    "What it cannot answer: HTTP paths, operation ids, whether an endpoint or "
    "operation is deprecated or retired, changelogs, or release history. "
    "Questions about the package itself (which releases exist, when one was "
    "published, whether a release is deprecated, which to install) are "
    "package-registry questions, not documentation questions: this corpus "
    "contains no release numbers."
)

_VERSION_LIKE = re.compile(r"^[vV]?(\d+)(?:\.(?:\d+|x))*$")


def version_diagnostic(value: Any) -> str:
    """The recoverable error for an api_version the corpus does not document.

    Built from the corpus, so it names only versions that exist. It says which
    kind of mistake was made, because each needs a different recovery. A release
    number or another spelling of a documented version is fixed by retrying. A
    version with no documentation must not be answered from a different one.
    """
    shown = json.dumps(value)
    accepted = _quoted(VERSIONS)
    documented = _quoted(VERSIONS, "and")
    text = str(value).strip() if isinstance(value, (str, int)) and not isinstance(value, bool) else ""
    match = _VERSION_LIKE.match(text)
    if match and f"v{int(match.group(1))}" in VERSIONS:
        right = f"v{int(match.group(1))}"
        return (
            f"api_version {shown} is not a documentation version. Documentation is published "
            f"once per major SDK version, as {documented}, and {shown} belongs to major version "
            f"{right[1:]}. Call {TOOL_NAME} again with api_version \"{right}\"."
        )
    if match:
        return (
            f"No documentation exists for api_version {shown}. This corpus documents {documented} "
            f"only, and {NEWEST} is the newest. The search did not fail: that version is simply not "
            f"documented. If the question is about {shown}, say the documentation does not cover "
            f"it, and do not answer with another version's values, because defaults differ between "
            f"versions. If the question is really about the current SDK, call {TOOL_NAME} again "
            f"with api_version \"{NEWEST}\"."
        )
    return (
        f"api_version {shown} is not a version name. Pass one of {accepted}; {NEWEST} is the "
        f"newest. A package release number maps to its major version: any {_major(NEWEST)}.x.y "
        f"release is \"{NEWEST}\". To cover more than one version, call once per version."
    )


def _argument_problem(arguments: dict[str, Any]) -> str | None:
    """Check `arguments` against INPUT_SCHEMA as the SDK would, but explain a bad api_version."""
    try:
        jsonschema.validate(instance=arguments, schema=INPUT_SCHEMA)
    except jsonschema.ValidationError as exc:
        if list(exc.absolute_path) == ["api_version"]:
            return version_diagnostic(arguments["api_version"])
        return f"Input validation error: {exc.message}"
    return None


# --------------------------------------------------------------------------
# the MCP server
# --------------------------------------------------------------------------

server = Server(SERVER_NAME)


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [types.Tool(name=TOOL_NAME, description=DESCRIPTION, inputSchema=INPUT_SCHEMA)]


@server.call_tool(validate_input=False)
async def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    # Arguments are validated here rather than by the SDK, against the same
    # INPUT_SCHEMA, so that a bad api_version can be answered with a diagnostic
    # instead of the SDK's generic "'v4' is not one of ['v2', 'v3']". Every other
    # schema failure keeps the SDK's own wording.
    #
    # Raising here is the SDK's way to return a tool error: it becomes a result
    # with isError: true and the message as text, not a JSON-RPC protocol error,
    # so the model can read it and try again.
    if name != TOOL_NAME:
        raise ValueError(f"unknown tool {name!r}: this server exposes only {TOOL_NAME!r}")
    problem = _argument_problem(arguments)
    if problem:
        raise ValueError(problem)

    # Runs on the event loop thread, so calls are served one at a time. Fine for
    # one local client; revisit if a gateway ever fans out concurrent calls.
    result = week7_tools.search_docs(
        ToolContext(cfg=_cfg),
        query=arguments["query"],
        api_version=arguments["api_version"],
    )
    if "error" in result:
        raise ValueError(result["error"])
    # A dict is sent twice by the SDK: as structuredContent, and serialised as
    # the text content block, for clients that only read text.
    return result


async def serve() -> None:
    async with stdio_server() as (read_stream, write_stream):
        # stdio_server has captured the real stdout for JSON-RPC. From here on a
        # stray print() anywhere in the retrieval stack lands on stderr instead.
        sys.stdout = sys.stderr
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(
        description="Docs MCP server over stdio. Exposes one tool: search_docs."
    ).parse_args(argv)
    anyio.run(serve)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
