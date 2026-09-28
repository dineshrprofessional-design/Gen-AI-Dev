"""Docs MCP server: Week 7's `search_docs`, exposed over MCP and nothing else.

    .venv/bin/python -m app.mcp.servers.docs_server      # run from the repo root

This is an adapter, not a second implementation. `week7_tools.search_docs` stays
the only place documentation is retrieved; this module only translates between
MCP and that function:

    tools/list  ->  one Tool: name, DESCRIPTION, the Week 7 input schema
    tools/call  ->  SDK validates `arguments` against that schema
                ->  week7_tools.search_docs(...)
                ->  the same dict, returned as an MCP tool result

The server holds no model and makes no model call. It never decides which tool
to use; the host's LLM does that from what `tools/list` returns.

Transport is stdio: the client launches this process and exchanges one JSON-RPC
message per line over stdin/stdout. Stdout therefore belongs to the protocol,
and anything else that writes there corrupts the stream.

Paths in `ToolConfig` are relative (`docs/`, `.rag_index/`), exactly as in
Week 7, so the process must start with the repo root as its working directory.
"""

from __future__ import annotations

import argparse
import copy
import sys
from typing import Any

import anyio
import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from app.rag import week7_tools
from app.rag.week7_tools import ToolConfig, ToolContext

SERVER_NAME = "docs-search"
TOOL_NAME = "search_docs"

# Starts from week7_tools.SEARCH_DOCS_DESCRIPTION, which is left unchanged as the
# "before". What is added: when to use it, what the two arguments mean, the
# result shape, and how to read a weak result. Deliberately general: it names no
# method or value that any evaluation question is about.
DESCRIPTION = (
    "Search the SDK reference documentation for prose describing what a Client "
    "method does: its behaviour, its parameter names, types and defaults, whether "
    "a parameter is required, and the caveats written in the Notes sections.\n\n"
    "Use it when the question is about how a Client method behaves, or what one "
    "of its parameters means, defaults to, or requires.\n\n"
    "api_version selects which version's reference pages are searched: \"v2\" or "
    "\"v3\". Only that version is searched, never both. The two versions document "
    "different defaults for the same parameter, and some methods and parameters "
    "exist only in v3. If the question names a version, pass it. If it asks to "
    "compare versions, or does not say and the answer could differ, call once per "
    "version.\n\n"
    "query is what to look for, in words: the method and the parameter or "
    "behaviour, or just a parameter name. Leave the version out of the query; "
    "api_version already filters by it.\n\n"
    "Returns JSON: api_version, count, and passages, best match first, each with "
    "rank, chunk_id, heading_path, score and text. Passages are ranked by "
    "similarity, not filtered for relevance, so the top one can still be "
    "off-topic: read the text before relying on it, and cite the chunk_id of any "
    "passage you use.\n\n"
    "This tool knows nothing about HTTP paths, operation ids, or whether anything "
    "is retired, and it has no changelog or release numbers. If no passage states "
    "the answer, the documentation does not state it."
)


def _week7_input_schema(name: str) -> dict[str, Any]:
    """The Week 7 parameter schema for one tool, reused rather than re-typed."""
    for schema in week7_tools.tool_schemas():
        if schema["function"]["name"] == name:
            return copy.deepcopy(schema["function"]["parameters"])
    raise LookupError(f"week7_tools.tool_schemas() has no tool named {name!r}")


INPUT_SCHEMA = _week7_input_schema(TOOL_NAME)

server = Server(SERVER_NAME)
_cfg = ToolConfig()


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [types.Tool(name=TOOL_NAME, description=DESCRIPTION, inputSchema=INPUT_SCHEMA)]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    # By the time this runs the SDK has checked `arguments` against INPUT_SCHEMA:
    # both keys present, api_version in the enum, nothing extra. A failure there
    # never reaches this function; the client gets isError: true instead.
    #
    # Raising here is the SDK's way to return a tool error: it becomes a result
    # with isError: true and the message as text, not a JSON-RPC protocol error,
    # so the model can read it and try again.
    if name != TOOL_NAME:
        raise ValueError(f"unknown tool {name!r}: this server exposes only {TOOL_NAME!r}")

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
