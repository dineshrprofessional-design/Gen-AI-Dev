"""Package Registry MCP server: release metadata and deprecation notices for packages.

    .venv/bin/python -m app.mcp.servers.package_server        # run from the repo root
    .venv/bin/python -m app.mcp.servers.package_server --registry path/to/packages.yaml

## Stand-in data, disclosed first

Everything this server returns comes from `registry/packages.yaml`: locally
authored stand-in package registry data for the Week 9 experiment. It was not
retrieved from any real package registry, it describes no real package, and
this is not an official third-party server. The disclosure travels with the
data: every tool result and resource carries the file's `provenance` line, and
the server's `instructions` in `initialize` say the same. See registry/README.md.

## What it exposes

    get_package_versions      tool      which releases of a package exist, and when
                                        each was published
    get_package_deprecation   tool      whether one release is deprecated, why, and
                                        the newest release that is not
    registry://<pkg>/support  resource  which release lines are maintained

Support status is a resource, not a tool. It is a small table that is the same
for every question, so it belongs in the prompt, attached by the host. A tool
would cost the model a turn to fetch it, and Task Set E names that as a mistake.

The two tools share only identity keys (package, version, found, reason,
provenance); every key that carries an answer belongs to exactly one of them.

## Package deprecation is not API-operation deprecation

A deprecated *release* is a build that should no longer be installed. It says
nothing about whether any HTTP operation or SDK method is deprecated. That is
Week 7's `api/` layer, which this server never reads.

## "Not found" is an answer; an unreadable registry is an error

    unknown package or release   found: false + a reason naming what does exist
    a release line, e.g. "v3"    found: false + the releases in that line
    registry file unreadable     isError: true, stating that the registry failed

So "4.0.0 does not exist yet" can never look like "the registry is down". The
registry file is read on every call, and nothing is cached between calls.

The server holds no model and makes no model call. It imports no agent, and it
decides nothing about which tool to use. Transport is stdio, as for the docs
server: stdout carries JSON-RPC and nothing else.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

import anyio
import mcp.types as types
import yaml
from mcp.server.lowlevel import Server
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.server.stdio import stdio_server
from pydantic import AnyUrl, BaseModel, Field, ValidationError, model_validator

SERVER_NAME = "package-registry"
SERVER_VERSION = "0.1.0"
DEFAULT_REGISTRY = Path("registry/packages.yaml")

SEMVER = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
# A release line rather than a release: "3", "3.x", "3.1", "3.1.x".
RELEASE_LINE = re.compile(r"^(\d+)(?:\.(\d+))?(?:\.x)?$")
SUPPORT_URI = re.compile(r"^registry://([^/]+)/support$")


# --------------------------------------------------------------------------
# the registry data
# --------------------------------------------------------------------------


class Release(BaseModel):
    version: str
    released: str


class DeprecationNotice(BaseModel):
    versions: list[str]
    deprecated_on: str
    notice: str


class SupportLine(BaseModel):
    line: str
    status: str
    since: str | None = None


class Package(BaseModel):
    summary: str = ""
    releases: list[Release]
    deprecations: list[DeprecationNotice] = Field(default_factory=list)
    support: list[SupportLine] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> "Package":
        published = [release.version for release in self.releases]
        bad = [v for v in published if not SEMVER.match(v)]
        if bad:
            raise ValueError(f"release versions must be MAJOR.MINOR.PATCH, got {bad}")
        if len(set(published)) != len(published):
            raise ValueError(f"duplicate release versions in {published}")
        for notice in self.deprecations:
            unknown = [v for v in notice.versions if v not in published]
            if unknown:
                raise ValueError(f"deprecation notice names unpublished releases {unknown}")
        return self

    def published(self) -> list[str]:
        """Release versions, oldest first by version number (not by list order)."""
        return sorted((r.version for r in self.releases), key=_version_key)

    def notice_for(self, version: str) -> DeprecationNotice | None:
        return next((n for n in self.deprecations if version in n.versions), None)


class Registry(BaseModel):
    provenance: str
    packages: dict[str, Package]


class RegistryUnavailable(RuntimeError):
    """The registry itself failed. Never used for "that package/release does not exist"."""


def load_registry(path: Path) -> Registry:
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        registry = Registry.model_validate(raw)
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise RegistryUnavailable(
            f"registry unavailable: could not load {path} ({type(exc).__name__}: {exc}). "
            "This is a failure of the registry, not an answer about any package."
        ) from exc
    # Registry names compare the way PyPI's do: case-insensitive, - _ . equivalent.
    registry.packages = {normalise_name(name): pkg for name, pkg in registry.packages.items()}
    return registry


def normalise_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name.strip()).lower()


def _version_key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


# --------------------------------------------------------------------------
# the lookups: pure functions of (registry, arguments)
# --------------------------------------------------------------------------


def _no_package(registry: Registry, package: str) -> dict[str, Any]:
    return {
        "package": package,
        "found": False,
        "reason": (
            f"this registry has no package named {package!r}. "
            f"Packages it holds: {sorted(registry.packages)}."
        ),
        "provenance": registry.provenance,
    }


def package_versions(registry: Registry, *, package: str) -> dict[str, Any]:
    name = normalise_name(package)
    entry = registry.packages.get(name)
    if entry is None:
        return _no_package(registry, package)
    by_version = {release.version: release.released for release in entry.releases}
    published = entry.published()
    return {
        "package": name,
        "found": True,
        "summary": entry.summary,
        "latest": published[-1],
        "releases": [{"version": v, "released": by_version[v]} for v in published],
        "provenance": registry.provenance,
    }


def package_deprecation(registry: Registry, *, package: str, version: str) -> dict[str, Any]:
    name = normalise_name(package)
    entry = registry.packages.get(name)
    if entry is None:
        return _no_package(registry, package)

    published = entry.published()
    wanted = version.strip().removeprefix("v").removeprefix("V")
    if wanted not in published:
        line = RELEASE_LINE.match(wanted)
        if line and not SEMVER.match(wanted):
            prefix = ".".join(part for part in line.groups() if part is not None) + "."
            members = [v for v in published if v.startswith(prefix)]
            if members:
                reason = (
                    f"{version!r} names a release line, not a single release. "
                    f"Releases of {name} in that line: {members}. Call again with one of them."
                )
            else:
                reason = (
                    f"{name} has no releases in the {version!r} line. The latest published "
                    f"release is {published[-1]}; published releases: {published}."
                )
        else:
            reason = (
                f"{name} has no release {version!r}. The latest published release is "
                f"{published[-1]}; published releases: {published}."
            )
        return {
            "package": name,
            "version": version,
            "found": False,
            "reason": reason,
            "provenance": registry.provenance,
        }

    notice = entry.notice_for(wanted)
    result: dict[str, Any] = {
        "package": name,
        "version": wanted,
        "found": True,
        "deprecated": notice is not None,
    }
    if notice is not None:
        current = [v for v in published if entry.notice_for(v) is None]
        result["deprecated_on"] = notice.deprecated_on
        result["notice"] = notice.notice
        # Derived, not authored: the newest release that carries no notice.
        result["replacement"] = current[-1] if current else None
    result["provenance"] = registry.provenance
    return result


def support_matrix(registry: Registry, package: str) -> dict[str, Any] | None:
    entry = registry.packages.get(normalise_name(package))
    if entry is None or not entry.support:
        return None
    published = entry.published()
    lines = []
    for line in entry.support:
        prefix = line.line.removesuffix("x")
        members = [v for v in published if v.startswith(prefix)]
        row = line.model_dump(exclude_none=True)
        row["latest"] = members[-1] if members else None
        lines.append(row)
    return {"package": normalise_name(package), "lines": lines, "provenance": registry.provenance}


# --------------------------------------------------------------------------
# what the model reads: descriptions and schemas
# --------------------------------------------------------------------------

GET_PACKAGE_VERSIONS_DESCRIPTION = (
    "List every published release of one package in the package registry, with "
    "the date each release was published, and name the latest release.\n\n"
    "Use it for questions about a package as something you install: which "
    "versions exist, which is the newest, when a release came out, or which "
    "release to pin.\n\n"
    "package is the package's registry name, the name you would install it by, "
    "not an SDK class or method name. Case and the separators - _ . do not "
    "matter.\n\n"
    "Returns JSON: package, found, summary, latest, and releases oldest first, "
    "each with version and released (YYYY-MM-DD). An unknown package comes back "
    "as found: false, with a reason naming the packages the registry holds.\n\n"
    "A release number such as 1.4.2 is a package version: one published build. "
    "It is not an API or documentation version such as v3, though a package's "
    "major number may track one. This tool does not say whether a release is "
    "deprecated; get_package_deprecation answers that for one release. It says "
    "nothing about what SDK methods do, their parameters or defaults, or HTTP "
    "operations and endpoints, and it has no changelog or release notes."
)

GET_PACKAGE_DEPRECATION_DESCRIPTION = (
    "Report whether one published release of a package has been deprecated on the "
    "package registry by its maintainers, meaning that build should no longer be "
    "installed. If it has, also return the maintainers' notice, the date, and the "
    "newest release that is not deprecated.\n\n"
    "Use it for questions like \"is release 1.4.2 of this package deprecated?\" or "
    "\"should we still pin that release?\". This is package deprecation: it is "
    "about which build of a package to install.\n\n"
    "package is the registry name, as for get_package_versions. version is one "
    "exact release such as 1.4.2; a leading v is accepted. A release line such as "
    "1.x or v1 is not a release: it comes back as found: false, listing the "
    "releases in that line. A release that was never published comes back as "
    "found: false, naming the latest published release. In both cases the "
    "registry answered; the release simply does not exist.\n\n"
    "Returns JSON: package, version, found, deprecated, and when deprecated also "
    "deprecated_on, notice and replacement.\n\n"
    "It does not answer whether an SDK method, a parameter, or an HTTP API "
    "operation or endpoint is deprecated or retired. A deprecated release says "
    "nothing about any one operation. It has no changelog or release notes."
)

_PACKAGE_PARAM = {
    "type": "string",
    "minLength": 1,
    "description": "Registry name of the package: the name it is installed by.",
}

TOOLS: dict[str, tuple[str, dict[str, Any], Callable[..., dict[str, Any]]]] = {
    "get_package_versions": (
        GET_PACKAGE_VERSIONS_DESCRIPTION,
        {
            "type": "object",
            "properties": {"package": _PACKAGE_PARAM},
            "required": ["package"],
            "additionalProperties": False,
        },
        package_versions,
    ),
    "get_package_deprecation": (
        GET_PACKAGE_DEPRECATION_DESCRIPTION,
        {
            "type": "object",
            "properties": {
                "package": _PACKAGE_PARAM,
                "version": {
                    "type": "string",
                    "minLength": 1,
                    "description": (
                        "One exact published release, e.g. 1.4.2. Not a release line "
                        "such as 1.x, and not an API version such as v3."
                    ),
                },
            },
            "required": ["package", "version"],
            "additionalProperties": False,
        },
        package_deprecation,
    ),
}

INSTRUCTIONS = (
    "Stand-in package registry for the Week 9 experiment. Its data is locally "
    "authored, not retrieved from any real package registry, and it is not an "
    "official third-party server."
)


# --------------------------------------------------------------------------
# the MCP server
# --------------------------------------------------------------------------


def build_server(registry_path: Path = DEFAULT_REGISTRY) -> Server:
    server = Server(SERVER_NAME, version=SERVER_VERSION, instructions=INSTRUCTIONS)

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(name=name, description=description, inputSchema=schema)
            for name, (description, schema, _) in TOOLS.items()
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        # The SDK has already validated `arguments` against the tool's schema.
        # A raise here becomes a result with isError: true, used for exactly two
        # things: a tool this server does not have, and a registry it cannot read.
        if name not in TOOLS:
            raise ValueError(f"unknown tool {name!r}: this server exposes {list(TOOLS)}")
        _, _, lookup = TOOLS[name]
        return lookup(load_registry(registry_path), **arguments)

    @server.list_resources()
    async def list_resources() -> list[types.Resource]:
        registry = load_registry(registry_path)
        return [
            types.Resource(
                uri=AnyUrl(f"registry://{name}/support"),
                name=f"{name} support matrix",
                description=(
                    f"Which release lines of {name} are maintained and which reached "
                    "end of life. Context for the host to attach, not a tool."
                ),
                mimeType="application/json",
            )
            for name, package in registry.packages.items()
            if package.support
        ]

    @server.read_resource()
    async def read_resource(uri: AnyUrl) -> list[ReadResourceContents]:
        found = SUPPORT_URI.match(str(uri))
        matrix = support_matrix(load_registry(registry_path), found.group(1)) if found else None
        if matrix is None:
            raise ValueError(f"no such resource {str(uri)!r}")
        return [ReadResourceContents(content=json.dumps(matrix, indent=2), mime_type="application/json")]

    return server


async def serve(registry_path: Path) -> None:
    server = build_server(registry_path)
    async with stdio_server() as (read_stream, write_stream):
        # stdio_server has captured the real stdout for JSON-RPC; see docs_server.
        sys.stdout = sys.stderr
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Package Registry MCP server over stdio (locally authored stand-in data)."
    )
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY, help="registry data file")
    args = parser.parse_args(argv)
    anyio.run(serve, args.registry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
