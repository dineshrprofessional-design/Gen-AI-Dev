"""The three tools the Week 7 agent and workflow both call.

    python -m app.rag.week7_tools --list
    python -m app.rag.week7_tools --schema
    python -m app.rag.week7_tools --call search_docs '{"query":"retry backoff","api_version":"v3"}'
    python -m app.rag.week7_tools --diff        # the graded third-tool description diff
    python -m app.rag.week7_tools --validate    # T1 payload disjointness, T3 corpus partition

Both arms of the race reach the outside world **only** through `dispatch()`. That
is what makes "same tools" structurally true rather than asserted: this module is
the single call site of `app.rag.index.search` in all of Week 7, and both arms
pass the same `ToolConfig`, so neither can quietly retrieve more than the other.

## One job each, and how that is checked

The brief's named failure is fixing tool thrash by bolting "use the correct tool"
onto the system prompt instead of sharpening overlapping descriptions. So the
descriptions here are written to own disjoint vocabulary, and the claim is tested
rather than asserted — see `--validate`:

    search_docs         what a method DOES, in prose, from docs/
    get_openapi_spec    which wire operation a method CALLS, from api/
    check_deprecation   whether that operation is RETIRED, and what replaced it

`api/openapi.json` on disk carries `deprecated` and `x-replaced-by`, because a
real OpenAPI document would. `get_openapi_spec` **projects those fields out**.
The files overlap; the tools do not. Two tools able to answer the same question
would make the agent's tool choice unmeasurable, which is the thing being raced.

## Why `api_version` is required with no default

A defaulted version is how "answered from the wrong version" happens silently —
Week 5 taxonomy mode 2, the one tagged *ships broken code*. Forcing the model to
choose makes a wrong-version answer attributable to a specific tool call in the
trace instead of invisible.
"""

import argparse
import hashlib
import inspect
import json
import re
import sys
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.rag.index import DEFAULT_DOCS, DEFAULT_INDEX, search
from app.rag.week7_corpus import DEPRECATIONS_JSON, OPENAPI_JSON

TOOL_NAMES: tuple[str, str, str] = ("search_docs", "get_openapi_spec", "check_deprecation")

SDK_METHODS = [
    "Client.send",
    "Client.stream",
    "Client.close",
    "Client.configure",
    "Client.upload",
    "Client.send_batch",
]
API_VERSIONS = ["v2", "v3"]

# Keys that address a thing or explain its absence, rather than answer a question.
# Every tool may return these; sharing them is not overlap. T1 asserts disjointness
# of the PAYLOAD keys, which are the actual answers. See `payload_keys()`.
#
# `reason` is in here because it only ever accompanies `found: False` — it explains
# an absence exactly as `error` does, and is never an answer. That is not taken on
# trust: `validate()` asserts `reason` never co-occurs with `found: True`, so if a
# tool later starts using it to carry a result, T1 stops being satisfiable by
# reclassification and the build fails instead.
IDENTITY_KEYS = {
    "found", "error", "reason", "api_version", "sdk_method",
    "operation_id", "path", "method",
}


class ToolConfig(BaseModel):
    """Retrieval settings. Both arms receive the same instance."""

    docs_root: Path = DEFAULT_DOCS
    index_dir: Path = DEFAULT_INDEX
    strategy: str = "structure"
    embedder: str = "bge-m3"
    k: int = 5
    hybrid: bool = False
    # Return the whole chunk, not a window into it.
    #
    # The structure chunker keeps a `## Parameters` table atomic precisely so it
    # is never split; truncating the returned text re-splits it at read time and
    # undoes that guarantee. Measured on this corpus: 13 of 38 chunks exceed 400
    # chars and the longest is 1950, so a 400-char window silently drops the
    # lower rows of every large table — the answer can rank first and still be
    # invisible. 2000 is above the longest chunk, so no chunk is ever cut.
    #
    # This is a property of the corpus, not of any question: both arms get the
    # same value from the same ToolConfig.
    snippet_chars: int = 2000

    def sha(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


class ToolCallRecord(BaseModel):
    seq: int
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    ok: bool = True
    error: str = ""
    latency_ms: float = 0.0
    result_chunk_ids: list[str] = Field(default_factory=list)
    files_read: list[str] = Field(default_factory=list)


@dataclass
class ToolContext:
    cfg: ToolConfig = field(default_factory=ToolConfig)
    records: list[ToolCallRecord] = field(default_factory=list)

    @property
    def path(self) -> tuple[str, ...]:
        """Tool names in call order — the input to the verdict's path analysis."""
        return tuple(record.name for record in self.records)


# --------------------------------------------------------------------------
# descriptions
# --------------------------------------------------------------------------

SEARCH_DOCS_DESCRIPTION = (
    "Search the SDK reference documentation for prose describing what a Client "
    "method does: its behaviour, its parameter names, types and defaults, whether "
    "a parameter is required, and the caveats written in the Notes sections.\n\n"
    "Returns documentation passages with the chunk id each came from.\n\n"
    "This tool knows nothing about HTTP paths, operation ids, or whether anything "
    "is retired."
)

GET_OPENAPI_SPEC_DESCRIPTION = (
    "Return the HTTP operation that one SDK method calls: its operationId, verb, "
    "path, and the request parameters it accepts.\n\n"
    "Use it to map an SDK method to the wire operation behind it, or to map a "
    "path you already have back to its SDK method.\n\n"
    "This tool does not report retirement status and returns no documentation prose."
)

# The graded third tool. Both revisions are kept so `--diff` generates the
# submission artifact from the shipped strings instead of a hand-written diff
# that could drift from what actually ran.
CHECK_DEPRECATION_DESCRIPTION_V1 = (
    "Get deprecation and migration information for the API. Use this to find out "
    "if something is deprecated, what changed between versions, what replaced a "
    "deprecated endpoint, and any relevant notes from the changelog or the spec. "
    "Pass the version you are interested in."
)

CHECK_DEPRECATION_DESCRIPTION = (
    "Report whether one HTTP operation is retired, and if it is, which operation "
    "replaced it.\n\n"
    "Returns only: retirement status, the version it was retired in, the replacing "
    "operationId, and that operation's SDK method and api_version.\n\n"
    "It returns no parameters, no defaults and no documentation text. Once you "
    "have the replacement, look up its details with the other tools."
)

CHECK_DEPRECATION_PARAMS_V1 = {
    "query": {"type": "string"},
    "version": {"type": "string"},
}


def _version_param(required_note: str) -> dict[str, Any]:
    return {
        "type": "string",
        "enum": API_VERSIONS,
        "description": required_note,
    }


def tool_schemas() -> list[dict[str, Any]]:
    """The tool list sent to the model. Identical for both arms."""
    return [
        {
            "type": "function",
            "function": {
                "name": "search_docs",
                "description": SEARCH_DOCS_DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What to look for, in words or as a parameter name.",
                        },
                        "api_version": _version_param(
                            "Which SDK version's pages to search. Required: the two "
                            "versions document different defaults."
                        ),
                    },
                    "required": ["query", "api_version"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_openapi_spec",
                "description": GET_OPENAPI_SPEC_DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sdk_method": {
                            "type": "string",
                            "enum": SDK_METHODS,
                            "description": "Look up by SDK method. Give this or path.",
                        },
                        "path": {
                            "type": "string",
                            "description": "Look up by HTTP path, e.g. /v2/messages. Give this or sdk_method.",
                        },
                        "api_version": _version_param(
                            "Which version's operation to return. Required."
                        ),
                    },
                    "required": ["api_version"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "check_deprecation",
                "description": CHECK_DEPRECATION_DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sdk_method": {
                            "type": "string",
                            "enum": SDK_METHODS,
                            "description": "Check by SDK method. Give this or path.",
                        },
                        "path": {
                            "type": "string",
                            "description": "Check by HTTP path, e.g. /v2/messages. Give this or sdk_method.",
                        },
                        "api_version": _version_param(
                            "Which version's operation to check. Required."
                        ),
                    },
                    "required": ["api_version"],
                    "additionalProperties": False,
                },
            },
        },
    ]


# --------------------------------------------------------------------------
# the api/ layer
# --------------------------------------------------------------------------


@lru_cache(maxsize=2)
def _openapi(path_str: str) -> dict[str, Any]:
    return json.loads(Path(path_str).read_text(encoding="utf-8"))


@lru_cache(maxsize=2)
def _deprecations(path_str: str) -> dict[str, Any]:
    return json.loads(Path(path_str).read_text(encoding="utf-8"))


@lru_cache(maxsize=2)
def _operation_index(path_str: str) -> dict[str, dict[str, Any]]:
    """operationId -> {path, method, op}, plus lookup by (sdk_method, version)."""
    spec = _openapi(path_str)
    index: dict[str, dict[str, Any]] = {}
    for path, verbs in spec["paths"].items():
        for verb, operation in verbs.items():
            index[operation["operationId"]] = {
                "path": path,
                "method": verb.upper(),
                "op": operation,
            }
    return index


def _find_operation(
    *, sdk_method: str | None, path: str | None, api_version: str
) -> dict[str, Any] | None:
    for entry in _operation_index(str(OPENAPI_JSON)).values():
        operation = entry["op"]
        if operation["x-api-version"] != api_version:
            continue
        if path is not None and entry["path"] == path:
            return entry
        if sdk_method is not None and operation["x-sdk-method"] == sdk_method:
            return entry
    return None


# --------------------------------------------------------------------------
# the tools
# --------------------------------------------------------------------------


def search_docs(ctx: ToolContext, *, query: str, api_version: str) -> dict[str, Any]:
    """Documentation prose. The only call site of index.search in Week 7."""
    if api_version not in API_VERSIONS:
        return {"error": f"api_version must be one of {API_VERSIONS}, got {api_version!r}"}
    hits = search(
        query,
        strategy=ctx.cfg.strategy,
        embedder_name=ctx.cfg.embedder,
        k=ctx.cfg.k,
        persist_dir=ctx.cfg.index_dir,
        where={"sdk_version": api_version},
        hybrid=ctx.cfg.hybrid,
    )
    return {
        "api_version": api_version,
        "count": len(hits),
        "passages": [
            {
                "rank": rank,
                "chunk_id": hit.chunk_id,
                "heading_path": str(hit.metadata.get("heading_path") or ""),
                "score": round(hit.score, 4),
                "text": hit.text[: ctx.cfg.snippet_chars],
            }
            for rank, hit in enumerate(hits, start=1)
        ],
    }


def get_openapi_spec(
    ctx: ToolContext,
    *,
    api_version: str,
    sdk_method: str | None = None,
    path: str | None = None,
) -> dict[str, Any]:
    """The wire operation. Retirement fields are deliberately projected out."""
    if api_version not in API_VERSIONS:
        return {"error": f"api_version must be one of {API_VERSIONS}, got {api_version!r}"}
    if sdk_method is None and path is None:
        return {"error": "give either sdk_method or path"}
    if sdk_method is not None and sdk_method not in SDK_METHODS:
        return {"error": f"sdk_method must be one of {SDK_METHODS}"}

    entry = _find_operation(sdk_method=sdk_method, path=path, api_version=api_version)
    if entry is None:
        # A real, documented non-answer beats an invented endpoint.
        if sdk_method == "Client.configure":
            return {
                "found": False,
                "api_version": api_version,
                "sdk_method": sdk_method,
                "reason": "client-local; sets defaults and does not cross the wire",
            }
        return {
            "found": False,
            "api_version": api_version,
            "sdk_method": sdk_method,
            "path": path,
            "reason": f"no operation for that {'path' if path else 'method'} in {api_version}",
        }

    operation = entry["op"]
    return {
        "found": True,
        "operation_id": operation["operationId"],
        "method": entry["method"],
        "path": entry["path"],
        "sdk_method": operation["x-sdk-method"],
        "api_version": operation["x-api-version"],
        "summary": operation["summary"],
        "request_parameters": operation["parameters"],
        "responses": operation["responses"],
        "docs_anchor": operation.get("x-docs-anchor"),
        "provenance": operation.get("x-provenance", ""),
    }


def check_deprecation(
    ctx: ToolContext,
    *,
    api_version: str,
    sdk_method: str | None = None,
    path: str | None = None,
) -> dict[str, Any]:
    """Retirement status and the replacement pointer. Nothing else."""
    if api_version not in API_VERSIONS:
        return {"error": f"api_version must be one of {API_VERSIONS}, got {api_version!r}"}
    if sdk_method is None and path is None:
        return {"error": "give either sdk_method or path"}
    if sdk_method is not None and sdk_method not in SDK_METHODS:
        return {"error": f"sdk_method must be one of {SDK_METHODS}"}

    entry = _find_operation(sdk_method=sdk_method, path=path, api_version=api_version)
    if entry is None:
        return {
            "found": False,
            "api_version": api_version,
            "sdk_method": sdk_method,
            "path": path,
            "reason": f"no operation for that {'path' if path else 'method'} in {api_version}",
        }

    operation_id = entry["op"]["operationId"]
    for record in _deprecations(str(DEPRECATIONS_JSON))["deprecated"]:
        if record["operation_id"] == operation_id:
            return {
                "found": True,
                "operation_id": operation_id,
                "api_version": record["api_version"],
                "sdk_method": record["sdk_method"],
                "retired": True,
                "retired_in": record["deprecated_in"],
                "retirement_reason": record["reason"],
                "replacement": record["replaced_by"],
            }
    return {
        "found": True,
        "operation_id": operation_id,
        "api_version": entry["op"]["x-api-version"],
        "sdk_method": entry["op"]["x-sdk-method"],
        "retired": False,
        "retired_in": None,
        "retirement_reason": None,
        "replacement": None,
    }


IMPLEMENTATIONS = {
    "search_docs": search_docs,
    "get_openapi_spec": get_openapi_spec,
    "check_deprecation": check_deprecation,
}

# Which files each tool is permitted to read. T3 asserts the partition holds, so
# api/ can never leak into search_docs' retrieval context.
TOOL_FILE_SCOPE = {
    "search_docs": "docs/",
    "get_openapi_spec": "api/",
    "check_deprecation": "api/",
}


def dispatch(ctx: ToolContext, name: str, arguments_json: str) -> dict[str, Any]:
    """Run one tool. Never raises — a tool error is a result both arms can see."""
    seq = len(ctx.records) + 1
    started = time.perf_counter()
    arguments: dict[str, Any] = {}
    try:
        if name not in IMPLEMENTATIONS:
            payload = {"error": f"unknown tool {name!r}; available: {list(IMPLEMENTATIONS)}"}
        else:
            try:
                arguments = json.loads(arguments_json) if arguments_json else {}
            except json.JSONDecodeError as exc:
                arguments = {}
                payload = {"error": f"arguments were not valid JSON: {exc}"}
            else:
                if not isinstance(arguments, dict):
                    payload = {"error": "arguments must be a JSON object"}
                else:
                    payload = IMPLEMENTATIONS[name](ctx, **arguments)
    except TypeError as exc:
        payload = {"error": f"bad arguments for {name}: {exc}"}
    except Exception as exc:  # noqa: BLE001 - a crash here must not end the run
        payload = {"error": f"{type(exc).__name__}: {exc}"}

    ctx.records.append(
        ToolCallRecord(
            seq=seq,
            name=name,
            arguments=arguments,
            ok="error" not in payload,
            error=str(payload.get("error", "")),
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
            result_chunk_ids=[p["chunk_id"] for p in payload.get("passages", [])],
            files_read=[TOOL_FILE_SCOPE.get(name, "?")],
        )
    )
    return payload


def tool_schema_sha() -> str:
    return hashlib.sha256(
        json.dumps(tool_schemas(), sort_keys=True).encode()
    ).hexdigest()[:16]


def tool_impl_sha() -> str:
    source = "".join(
        inspect.getsource(fn) for fn in (*IMPLEMENTATIONS.values(), dispatch)
    )
    return hashlib.sha256(source.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------
# T1 / T3
# --------------------------------------------------------------------------


def payload_keys(name: str) -> set[str]:
    """Every non-identity key a tool can return, over the whole argument space.

    Identity keys (found/error/api_version/sdk_method/operation_id/path/method)
    are addressing, not answers — every tool may carry them and sharing them is
    not overlap. What must be disjoint is the payload: the thing the tool is for.
    """
    ctx = ToolContext()
    keys: set[str] = set()
    combos: list[dict[str, Any]] = []
    if name == "search_docs":
        combos = [{"query": "timeout", "api_version": v} for v in API_VERSIONS]
    else:
        combos = [
            {"sdk_method": m, "api_version": v} for m in SDK_METHODS for v in API_VERSIONS
        ]
        spec = _openapi(str(OPENAPI_JSON))
        combos += [
            {"path": p, "api_version": v} for p in spec["paths"] for v in API_VERSIONS
        ]
    for arguments in combos:
        payload = IMPLEMENTATIONS[name](ctx, **arguments)
        keys |= set(payload)
    return keys - IDENTITY_KEYS


def validate() -> tuple[list[str], list[str]]:
    failures: list[str] = []
    warnings: list[str] = []

    # T1 — payload disjointness across the whole argument space
    keys = {name: payload_keys(name) for name in TOOL_NAMES}
    for first in range(len(TOOL_NAMES)):
        for second in range(first + 1, len(TOOL_NAMES)):
            a, b = TOOL_NAMES[first], TOOL_NAMES[second]
            shared = keys[a] & keys[b]
            if shared:
                failures.append(f"T1: {a} and {b} both answer with {sorted(shared)}")

    # the projection that makes T1 true must actually be in place
    ctx = ToolContext()
    leaked = [
        key
        for key in ("deprecated", "x-replaced-by", "x-deprecated-in", "x-deprecation-reason")
        if key in get_openapi_spec(ctx, sdk_method="Client.send", api_version="v2")
    ]
    if leaked:
        failures.append(
            f"T1: get_openapi_spec leaks retirement fields {leaked}; it must project them out"
        )

    # `reason` is classified as an identity key only because it explains an
    # absence. Enforce that, so T1 cannot be satisfied by relabelling an answer.
    for name in TOOL_NAMES:
        if name == "search_docs":
            continue
        combos: list[dict[str, Any]] = [
            {"sdk_method": m, "api_version": v} for m in SDK_METHODS for v in API_VERSIONS
        ]
        combos += [
            {"path": p, "api_version": v}
            for p in _openapi(str(OPENAPI_JSON))["paths"]
            for v in API_VERSIONS
        ]
        for arguments in combos:
            payload = IMPLEMENTATIONS[name](ToolContext(), **arguments)
            if payload.get("found") and "reason" in payload:
                failures.append(
                    f"T1: {name} returns 'reason' alongside found=True for {arguments}; "
                    "'reason' is an absence explanation, not an answer"
                )
                break

    # T3 — corpus partition
    for name, scope in TOOL_FILE_SCOPE.items():
        source = inspect.getsource(IMPLEMENTATIONS[name])
        if scope == "docs/" and ("OPENAPI_JSON" in source or "DEPRECATIONS_JSON" in source):
            failures.append(f"T3: {name} is scoped to docs/ but reads api/")
        if scope == "api/" and "search(" in source:
            failures.append(f"T3: {name} is scoped to api/ but calls retrieval")

    # every schema must require api_version with no default
    for schema in tool_schemas():
        fn = schema["function"]
        params = fn["parameters"]
        if "api_version" not in params.get("required", []):
            failures.append(f"{fn['name']}: api_version must be required")
        version_schema = params["properties"]["api_version"]
        if version_schema.get("enum") != API_VERSIONS:
            failures.append(f"{fn['name']}: api_version must be an enum of {API_VERSIONS}")
        if "default" in version_schema:
            failures.append(
                f"{fn['name']}: api_version must have no default — a defaulted version is "
                "how wrong-version answers happen silently"
            )

    # description non-overlap: the owned verbs must not appear in a sibling
    owned = {
        "search_docs": ["documentation", "prose", "passages"],
        "get_openapi_spec": ["operationid", "verb", "wire operation"],
        "check_deprecation": ["retired", "replaced", "retirement"],
    }
    described = {
        "search_docs": SEARCH_DOCS_DESCRIPTION.lower(),
        "get_openapi_spec": GET_OPENAPI_SPEC_DESCRIPTION.lower(),
        "check_deprecation": CHECK_DEPRECATION_DESCRIPTION.lower(),
    }
    # A sibling may use an owned term only inside a negated or scoping sentence —
    # "returns no documentation text" is a boundary statement, not a claim on the
    # territory. Scoped per sentence, because a non-goal at the end of a
    # description should not license the term in an earlier sentence.
    negation = ("does not", "knows nothing", "no ", "not ", "never", "returns only")
    for owner, terms in owned.items():
        for other, text in described.items():
            if other == owner:
                continue
            # Split on sentence ends and blank lines only — never on ":", or a
            # scoping lead-in like "Returns only:" is severed from the list it
            # scopes and its own clause reads as an unqualified claim.
            for sentence in re.split(r"(?<=\.)\s+|\n\n+", text):
                if any(marker in sentence for marker in negation):
                    continue
                for term in terms:
                    if term in sentence:
                        warnings.append(
                            f"{other} claims {term!r} (owned by {owner}) in a "
                            f"non-negated sentence: {sentence.strip()[:70]!r}"
                        )
    return failures, warnings


def render_description_diff() -> str:
    """The submission artifact: what changed in the third tool, and why."""
    import difflib

    before = CHECK_DEPRECATION_DESCRIPTION_V1.split()
    after = CHECK_DEPRECATION_DESCRIPTION.split()
    lines = [
        "check_deprecation — description and parameter diff",
        "=" * 62,
        "",
        "BEFORE (v1)",
        "-" * 62,
        CHECK_DEPRECATION_DESCRIPTION_V1,
        "",
        f"parameters: {json.dumps(CHECK_DEPRECATION_PARAMS_V1, indent=2)}",
        "",
        "AFTER (v2, shipped)",
        "-" * 62,
        CHECK_DEPRECATION_DESCRIPTION,
        "",
        "parameters: "
        + json.dumps(
            tool_schemas()[2]["function"]["parameters"], indent=2, sort_keys=True
        ),
        "",
        "WORD-LEVEL DIFF",
        "-" * 62,
    ]
    for token in difflib.unified_diff(before, after, lineterm="", n=2):
        lines.append(token)
    lines += [
        "",
        "WHAT CHANGED, AND WHY",
        "-" * 62,
        "1. three jobs -> one. v1 promised deprecation status AND cross-version",
        "   differences AND changelog notes. 'what changed between versions' is",
        "   search_docs territory; a tool with three jobs cannot be shown",
        "   non-overlapping with anything.",
        "2. open-ended 'any relevant notes' -> explicit non-goals ('no parameters,",
        "   no defaults, no documentation text'). The open clause invites a call",
        "   for anything version-shaped, inflating call count without inflating",
        "   correctness.",
        "3. version: string -> api_version: enum [v2, v3], required, no default.",
        "   Free text admits '3', 'latest', '3.x', each needing normalisation or",
        "   a runtime error — a silent failure surface.",
        "4. query: string -> structured sdk_method enum OR path. A free-text query",
        "   makes this a second search engine, which is the overlap with",
        "   search_docs.",
        "5. added the hand-off sentence, so the two-hop plan is discoverable",
        "   rather than hoped for.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="The Week 7 tools.")
    parser.add_argument("--list", action="store_true", help="names, one-line jobs, params")
    parser.add_argument("--schema", action="store_true", help="the JSON schemas sent to the model")
    parser.add_argument("--call", nargs=2, metavar=("NAME", "JSON"), help="run one tool")
    parser.add_argument("--diff", action="store_true", help="the third tool's description diff")
    parser.add_argument("--validate", action="store_true", help="T1 + T3 + schema rules")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.schema:
        print(json.dumps(tool_schemas(), indent=2))
        return 0

    if args.list:
        print(f"schema sha: {tool_schema_sha()}   impl sha: {tool_impl_sha()}")
        print()
        for schema in tool_schemas():
            fn = schema["function"]
            params = fn["parameters"]
            print(f"{fn['name']}  [reads {TOOL_FILE_SCOPE[fn['name']]}]")
            print(f"  {fn['description'].splitlines()[0]}")
            for key, spec in params["properties"].items():
                mark = "*" if key in params.get("required", []) else " "
                enum = f" enum={spec['enum']}" if "enum" in spec else ""
                print(f"    {mark} {key}: {spec['type']}{enum}")
            print(f"  payload keys: {sorted(payload_keys(fn['name']))}")
            print()
        print("* = required")
        return 0

    if args.diff:
        rendered = render_description_diff()
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(rendered, encoding="utf-8")
            print(f"wrote {args.out}")
        else:
            print(rendered)
        return 0

    if args.validate:
        failures, warnings = validate()
        for warning in warnings:
            print(f"WARN  {warning}")
        for failure in failures:
            print(f"FAIL  {failure}")
        print()
        for name in TOOL_NAMES:
            print(f"  {name:20} payload keys: {sorted(payload_keys(name))}")
        print()
        if failures:
            print(f"VALIDATION FAILED — {len(failures)} problem(s)")
            return 1
        print(f"VALIDATION PASSED — {len(warnings)} warning(s)")
        return 0

    if args.call:
        name, arguments_json = args.call
        ctx = ToolContext()
        payload = dispatch(ctx, name, arguments_json)
        print(json.dumps(payload, indent=2)[:4000])
        record = ctx.records[-1]
        print(f"\n-- ok={record.ok} {record.latency_ms:.1f} ms  reads {record.files_read}")
        return 0 if record.ok else 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
