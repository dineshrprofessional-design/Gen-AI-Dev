"""MCP client: start the configured servers, discover their tools, route calls.

    .venv/bin/python -m app.mcp.client                   # connect, list every discovered tool
    .venv/bin/python -m app.mcp.client --openai          # the same tools, as the LLM receives them
    .venv/bin/python -m app.mcp.client --call TOOL '{"arg": "value"}'   # one tools/call, raw result
    .venv/bin/python -m app.mcp.client --wire /tmp/wire.jsonl   # also keep every JSON-RPC message

## What the host knows, and what it discovers

The host knows a config file: which servers to start, and how. It does not know
which tools exist. That is whatever each server answers to `tools/list`, asked
fresh on every connect, and no tool name appears anywhere in this module.

    config/mcp_servers.json -> start each server (stdio) -> initialize
                            -> tools/list (every page)   -> tool name -> server
    call_tool(name, args)   -> that server's session     -> tools/call

The config uses the `mcpServers` shape other MCP hosts use:

    {"mcpServers": {"<name>": {"command": "...", "args": [...], "cwd": "...", "env": {...}}}}

Relative `command` and `cwd` paths are relative to the directory the host was
started in, which in this repo is always the root. `env` is optional and is added
to the SDK's minimal default environment (HOME, LOGNAME, PATH, SHELL, TERM, USER).
A server is never handed the host's full environment, so it never sees
GROQ_API_KEY.

## What the SDK does, and what this module adds

The SDK owns the protocol: framing, request ids, the initialize handshake,
timeouts. On top of it this module adds only what a host needs:

  routing      each discovered tool maps to the one server that listed it. Two
               servers listing the same name is a hard error at connect time,
               never a silent pick.
  conversion   MCP Tool -> the OpenAI function-tool shape `chat_once` sends. The
               inputSchema is copied through untouched; nothing is invented.
  a sync face  the SDK is async-only; the agent loop and `chat_once` are not. One
               background event loop (an anyio blocking portal) owns every
               session, and the blocking methods here hand work to it.
  a wire log   every JSON-RPC message in both directions, per server.

## The wire log is what the SDK sent and parsed, not raw bytes

Outgoing messages are recorded from the same model the stdio transport
serialises one line later, so they match the wire exactly. Incoming ones are
recorded after the transport parsed them, re-dumped with nulls dropped; the
content is the same, but key order and whitespace are the SDK's rather than the
server's.

The client offers the server no capabilities: no sampling, no elicitation, no
roots. A server therefore has no way to ask this host's LLM for anything.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import re
import sys
import time
from contextlib import AsyncExitStack, ExitStack, asynccontextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Any, TextIO

import mcp.types as types
from anyio.from_thread import start_blocking_portal
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.exceptions import McpError
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_CONFIG = Path("config/mcp_servers.json")

CLIENT_INFO = types.Implementation(name="gen-ai-dev-week9", version="0.1.0")

# OpenAI function names. An MCP tool name outside this is refused at connect
# time rather than renamed: a renamed tool is a tool the server never listed.
OPENAI_TOOL_NAME = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

# A server that keeps handing back a cursor is broken, not large.
MAX_TOOL_PAGES = 100

CLIENT_TO_SERVER = "client->server"
SERVER_TO_CLIENT = "server->client"

log = logging.getLogger("week9.mcp")

__all__ = [
    "DEFAULT_CONFIG", "ServerConfig", "load_config", "DiscoveredTool", "to_openai_tool",
    "WireLog", "McpClient", "McpClientError", "McpConfigError", "McpConnectError",
    "DuplicateToolError", "UnknownToolError", "ToolCallError",
]


# --------------------------------------------------------------------------
# errors: each names the server and the phase, so a failure reads on its own
# --------------------------------------------------------------------------


class McpClientError(RuntimeError):
    """Anything the host has to stop on or report."""


class McpConfigError(McpClientError):
    """The config file is missing, malformed, or lists no server."""


class McpConnectError(McpClientError):
    """A server could not be started, initialised, or asked for its tools."""


class DuplicateToolError(McpClientError):
    """Two servers expose the same tool name; routing would be a guess."""


class UnknownToolError(McpClientError):
    """A call named a tool no connected server listed. No request is sent."""


class ToolCallError(McpClientError):
    """tools/call produced no result: a JSON-RPC error, a timeout, a lost server.

    A result with `isError: true` is NOT this. That is the server answering, and
    it is returned to the caller like any other result.
    """


def _describe(exc: BaseException) -> str:
    if isinstance(exc, McpError):
        return f"{exc.error.message} (JSON-RPC error {exc.error.code})"
    return f"{type(exc).__name__}: {exc}"


def _leaf(exc: BaseException) -> BaseException:
    """The one exception that explains a failure, out of anyio's exception groups."""
    while isinstance(exc, BaseExceptionGroup):
        ours = [e for e in exc.exceptions if isinstance(e, McpClientError)]
        exc = (ours or list(exc.exceptions))[0]
    return exc


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


class ServerConfig(BaseModel):
    """One entry under `mcpServers`. Unknown keys are errors, so a typo'd
    `arg` fails loudly instead of starting a server with no arguments."""

    model_config = ConfigDict(extra="forbid")

    name: str
    command: str
    args: list[str] = Field(default_factory=list)
    cwd: str | None = None
    env: dict[str, str] | None = None

    def stdio_parameters(self) -> StdioServerParameters:
        # abspath, not resolve(): .venv/bin/python is a symlink, and following it
        # lands on the base interpreter, which has none of the venv's packages.
        command = self.command
        if os.sep in command and not os.path.isabs(command):
            command = os.path.abspath(command)
        cwd = os.path.abspath(self.cwd) if self.cwd else None
        return StdioServerParameters(command=command, args=self.args, cwd=cwd, env=self.env)

    def describe(self) -> str:
        """For logs. Never includes env values."""
        return " ".join([self.command, *self.args]) + (f"  (cwd={self.cwd})" if self.cwd else "")


def load_config(path: Path = DEFAULT_CONFIG) -> list[ServerConfig]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise McpConfigError(f"MCP config {path} not found") from exc
    except json.JSONDecodeError as exc:
        raise McpConfigError(f"MCP config {path} is not valid JSON: {exc}") from exc
    servers = data.get("mcpServers") if isinstance(data, dict) else None
    if not isinstance(servers, dict) or not servers:
        raise McpConfigError(f"MCP config {path} lists no servers under 'mcpServers'")
    try:
        return [ServerConfig(name=name, **spec) for name, spec in servers.items()]
    except (TypeError, ValueError) as exc:
        raise McpConfigError(f"MCP config {path}: {exc}") from exc


# --------------------------------------------------------------------------
# discovered tools, and the conversion the LLM needs
# --------------------------------------------------------------------------


def to_openai_tool(tool: types.Tool) -> dict[str, Any]:
    """MCP Tool -> OpenAI function tool.

    name, description and inputSchema are copied, not rewritten. MCP-only fields
    (title, annotations, outputSchema) have no slot in the OpenAI shape and are
    not sent to the model.
    """
    if not OPENAI_TOOL_NAME.match(tool.name):
        raise McpClientError(
            f"tool name {tool.name!r} is not a valid OpenAI function name "
            f"({OPENAI_TOOL_NAME.pattern}); refusing to rename it"
        )
    function: dict[str, Any] = {"name": tool.name}
    if tool.description:
        function["description"] = tool.description
    function["parameters"] = copy.deepcopy(tool.inputSchema)
    return {"type": "function", "function": function}


@dataclass(frozen=True)
class DiscoveredTool:
    server: str
    tool: types.Tool

    @property
    def name(self) -> str:
        return self.tool.name

    def openai_schema(self) -> dict[str, Any]:
        return to_openai_tool(self.tool)


# --------------------------------------------------------------------------
# the wire log
# --------------------------------------------------------------------------


@dataclass
class WireLog:
    """Every JSON-RPC message, in order, across every server."""

    started: float = field(default_factory=time.perf_counter)
    messages: list[dict[str, Any]] = field(default_factory=list)

    def record(self, server: str, direction: str, item: Any) -> None:
        entry: dict[str, Any] = {
            "seq": len(self.messages) + 1,
            "t_ms": round((time.perf_counter() - self.started) * 1000.0, 1),
            "server": server,
            "direction": direction,
        }
        if isinstance(item, Exception):
            # The transport could not parse a line the server wrote.
            entry["transport_error"] = f"{type(item).__name__}: {item}"
        else:
            entry["message"] = item.message.model_dump(by_alias=True, mode="json", exclude_none=True)
        self.messages.append(entry)

    def write_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in self.messages),
            encoding="utf-8",
        )


class _WireTap:
    """Stands in for one of the SDK's memory streams and records what crosses it.

    ClientSession uses its streams through `async with`, `async for` and `send()`
    and nothing else, so those are the only methods forwarded. Nothing is changed
    in flight.
    """

    def __init__(self, stream: Any, *, server: str, direction: str, wire: WireLog) -> None:
        self._stream = stream
        self._server = server
        self._direction = direction
        self._wire = wire

    async def __aenter__(self) -> "_WireTap":
        await self._stream.__aenter__()
        return self

    async def __aexit__(self, *exc_info: Any) -> Any:
        return await self._stream.__aexit__(*exc_info)

    def __aiter__(self) -> "_WireTap":
        return self

    async def __anext__(self) -> Any:
        item = await self._stream.__anext__()
        self._wire.record(self._server, self._direction, item)
        return item

    async def send(self, item: Any) -> None:
        self._wire.record(self._server, self._direction, item)
        await self._stream.send(item)


# --------------------------------------------------------------------------
# the client
# --------------------------------------------------------------------------


class McpClient:
    """Connect to every configured server, discover, route. Blocking API.

        with McpClient(load_config()) as client:
            client.tools            # from live tools/list, every server
            client.openai_tools()   # what the LLM is sent
            client.call_tool(name, arguments)  # routed to the server that listed it

    Connecting is all-or-nothing: if any server fails to start, initialise or
    list its tools, or two servers list the same tool, `__enter__` raises and
    every server already started is shut down.
    """

    def __init__(
        self,
        servers: list[ServerConfig],
        *,
        connect_timeout_s: float = 60.0,
        call_timeout_s: float = 300.0,
        errlog: TextIO | None = None,
    ) -> None:
        names = [server.name for server in servers]
        if len(set(names)) != len(names):
            raise McpConfigError(f"server names must be unique, got {names}")
        self.servers = servers
        self.connect_timeout_s = connect_timeout_s
        # Generous on purpose: the docs server loads its embedding model on the
        # first search, which measured ~35 s in Step 3.
        self.call_timeout_s = call_timeout_s
        self.errlog = errlog if errlog is not None else sys.stderr
        self.wire = WireLog()
        self.server_info: dict[str, dict[str, Any]] = {}
        self.tools: list[DiscoveredTool] = []
        self._routes: dict[str, str] = {}
        self._sessions: dict[str, ClientSession] = {}
        self._stack = ExitStack()
        self._portal: Any = None

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> "McpClient":
        try:
            self._portal = self._stack.enter_context(start_blocking_portal())
            self._stack.enter_context(self._portal.wrap_async_context_manager(self._connect_all()))
        except BaseException as exc:
            self._stack.close()
            leaf = _leaf(exc)
            if isinstance(leaf, McpClientError):
                raise leaf from None
            raise McpConnectError(f"could not connect: {_describe(leaf)}") from leaf
        log.info(
            "discovered %d tool(s) across %d server(s): %s",
            len(self.tools), len(self._sessions),
            ", ".join(f"{t.name}@{t.server}" for t in self.tools) or "none",
        )
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self._stack.close()
        log.info("disconnected from %d server(s)", len(self._sessions))

    @asynccontextmanager
    async def _connect_all(self):
        async with AsyncExitStack() as stack:
            for server in self.servers:
                await self._connect(stack, server)
            yield

    async def _connect(self, stack: AsyncExitStack, server: ServerConfig) -> None:
        log.info("server %r: starting %s", server.name, server.describe())
        try:
            read, write = await stack.enter_async_context(
                stdio_client(server.stdio_parameters(), errlog=self.errlog)
            )
        except OSError as exc:
            raise McpConnectError(
                f"server {server.name!r}: could not start {server.command!r}: {exc}"
            ) from exc

        session = await stack.enter_async_context(
            ClientSession(
                _WireTap(read, server=server.name, direction=SERVER_TO_CLIENT, wire=self.wire),
                _WireTap(write, server=server.name, direction=CLIENT_TO_SERVER, wire=self.wire),
                read_timeout_seconds=timedelta(seconds=self.connect_timeout_s),
                client_info=CLIENT_INFO,
            )
        )

        try:
            init = await session.initialize()
        except Exception as exc:  # noqa: BLE001 - reported with the server's name
            raise McpConnectError(
                f"server {server.name!r}: initialize failed: {_describe(exc)}"
            ) from exc
        self.server_info[server.name] = {
            "protocolVersion": init.protocolVersion,
            "serverInfo": init.serverInfo.model_dump(exclude_none=True),
            "capabilities": init.capabilities.model_dump(exclude_none=True),
        }
        log.info(
            "server %r: initialize ok (protocol %s, server %s %s)",
            server.name, init.protocolVersion, init.serverInfo.name, init.serverInfo.version,
        )
        if init.capabilities.tools is None:
            log.info("server %r: declares no tools capability; nothing to list", server.name)
            self._sessions[server.name] = session
            return

        listed: list[types.Tool] = []
        cursor: str | None = None
        try:
            for _ in range(MAX_TOOL_PAGES):
                page = await session.list_tools(cursor)
                listed.extend(page.tools)
                cursor = page.nextCursor
                if not cursor:
                    break
            else:
                raise McpConnectError(f"tools/list still paginating after {MAX_TOOL_PAGES} pages")
        except McpConnectError as exc:
            raise McpConnectError(f"server {server.name!r}: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise McpConnectError(
                f"server {server.name!r}: tools/list failed: {_describe(exc)}"
            ) from exc
        log.info(
            "server %r: tools/list returned %d tool(s): %s",
            server.name, len(listed), ", ".join(tool.name for tool in listed) or "none",
        )

        for tool in listed:
            owner = self._routes.get(tool.name)
            if owner is not None:
                raise DuplicateToolError(
                    f"tool {tool.name!r} is listed by both server {owner!r} and server "
                    f"{server.name!r}; refusing to pick one. Remove or rename it on one of them."
                )
            to_openai_tool(tool)  # refuse an unusable name now, not mid-run
            self._routes[tool.name] = server.name
            self.tools.append(DiscoveredTool(server=server.name, tool=tool))
        self._sessions[server.name] = session

    # -- discovery ---------------------------------------------------------

    @property
    def routes(self) -> dict[str, str]:
        """Tool name -> the server that listed it."""
        return dict(self._routes)

    def server_for(self, name: str) -> str | None:
        return self._routes.get(name)

    def openai_tools(self) -> list[dict[str, Any]]:
        return [tool.openai_schema() for tool in self.tools]

    # -- calling -----------------------------------------------------------

    def call_tool(self, name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        """Route one call to the server that listed `name`, and wait for the result."""
        if self._portal is None:
            raise McpClientError("McpClient is not connected; use it as a context manager")
        server = self._routes.get(name)
        if server is None:
            raise UnknownToolError(f"unknown tool {name!r}; available: {list(self._routes)}")

        session = self._sessions[server]
        log.info("tools/call %s -> server %r", name, server)
        started = time.perf_counter()
        try:
            result = self._portal.call(
                partial(
                    session.call_tool, name, arguments,
                    read_timeout_seconds=timedelta(seconds=self.call_timeout_s),
                )
            )
        except Exception as exc:  # noqa: BLE001
            leaf = _leaf(exc)
            raise ToolCallError(
                f"tools/call {name!r} on server {server!r} failed: {_describe(leaf)}"
            ) from leaf
        log.info(
            "tools/call %s <- server %r: %s in %.0f ms",
            name, server, "isError" if result.isError else "ok",
            (time.perf_counter() - started) * 1000.0,
        )
        return result


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def configure_logging() -> None:
    """INFO for the week9.* loggers on stderr; everything else stays at WARNING."""
    logging.basicConfig(
        level=logging.WARNING,
        stream=sys.stderr,
        format="%(asctime)s.%(msecs)03d [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("week9").setLevel(logging.INFO)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Connect to the configured MCP servers and list their tools.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--openai", action="store_true", help="print the tools in the shape the LLM is sent")
    parser.add_argument("--call", nargs=2, metavar=("TOOL", "ARGUMENTS_JSON"), help="call one tool and print the result")
    parser.add_argument("--wire", type=Path, help="write every JSON-RPC message to this JSONL file")
    args = parser.parse_args(argv)
    configure_logging()

    try:
        with McpClient(load_config(args.config)) as client:
            if args.call:
                name, raw = args.call
                try:
                    arguments = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise McpClientError(f"--call arguments are not valid JSON: {exc}") from exc
                result = client.call_tool(name, arguments)
                print(json.dumps(result.model_dump(by_alias=True, mode="json", exclude_none=True), indent=2))
            elif args.openai:
                print(json.dumps(client.openai_tools(), indent=2))
            else:
                print(f"{len(client.tools)} tool(s) discovered via tools/list:")
                for tool in client.tools:
                    print(f"  {tool.name:24} server={tool.server}")
            wire = client.wire
    except McpClientError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.wire:
        wire.write_jsonl(args.wire)
        print(f"wrote {len(wire.messages)} JSON-RPC messages to {args.wire}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
