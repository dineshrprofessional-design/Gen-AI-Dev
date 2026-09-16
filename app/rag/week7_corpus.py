"""Generate the Week 7 API layer under `api/`, derived from `docs/`.

    python -m app.rag.week7_corpus --reconcile   # the 8-vs-9 divergence table
    python -m app.rag.week7_corpus               # write api/openapi.json, changelog.md, deprecations.json
    python -m app.rag.week7_corpus --check       # re-derive and byte-diff the committed files
    python -m app.rag.week7_corpus --validate    # regression gates + contradiction detectors

Week 7 races a tool-calling agent against a fixed workflow over migration
questions. The brief assumes an OpenAPI spec and a changelog exist. They do not:
`docs/` is ten Python-SDK reference pages with no HTTP path, no changelog and no
deprecation vocabulary anywhere. So this module manufactures that layer — but
manufactures as little of it as possible, and puts none of it in `docs/`.

## Why `api/` and never `docs/`

`DEFAULT_DOCS = Path("docs")` is the only corpus root in the codebase; every
consumer takes a `docs_root` argument and defaults to it. A sibling directory is
therefore invisible to the Week 3-6 pipeline by construction rather than by care,
which matters because three committed Week 6 negative controls would flip if the
indexed corpus grew an API surface:

    w6-030  "is there a changelog for v3.2.0"           -> a changelog in docs/
    w6-031  "rate limit on the /v1/embeddings endpoint" -> a spec with /v1/ or a rate limit
    w6-032  "how do I authenticate with OAuth2"         -> a spec with an OAuth2 scheme

All three currently PASS in report/judge_v1.txt and judge_v2.txt. `--validate`
re-asserts that `docs/` is byte-identical and that none of the forbidden tokens
reached `api/`, so breaking one of them fails the build instead of silently
moving a published number.

## Derived vs authored

Everything here is generated from the real parameter tables except
`api/endpoints.yaml`, which is ~70 lines of authored routing table plus four
deprecation records. That file is the entire fiction, and it is disclosed in
`api/README.md` and echoed into every generated operation as `x-provenance`, so
the disclosure survives being read one operation at a time through a tool.

## The divergence trap this module exists to avoid

`week6_assertions.divergent_defaults()` is **name-scoped**: it keys on the
parameter name alone and reports 9 symbols. Only 8 are genuine version changes.
`base_url` is not one - v3 `send`'s per-call override defaults to None while v3
`configure`'s defaults to the URL, which are two different parameters that share
a name. A changelog generated from the name-scoped function would assert a change
that the corpus flatly contradicts. `divergent_by_method()` below keys on
(page_id, parameter) instead, and `--reconcile` prints the difference with its
reason so the exclusion is visible rather than silent.
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from app.rag.extractors.markdown import parse_front_matter
from app.rag.index import DEFAULT_DOCS
from app.rag.markdown import BlockKind, parse_blocks

API_DIR = Path("api")
ENDPOINTS_YAML = API_DIR / "endpoints.yaml"
OPENAPI_JSON = API_DIR / "openapi.json"
CHANGELOG_MD = API_DIR / "changelog.md"
DEPRECATIONS_JSON = API_DIR / "deprecations.json"

# The corpus tree as it stood when Week 7 began, pinned while `git diff --quiet
# HEAD -- docs/` independently confirmed it was identical to the committed tree.
# G1 compares against this, so a docs/ edit during Week 7 fails loudly instead of
# quietly invalidating both the generated layer and every Week 4/5/6 number
# measured against it. Computed by docs_tree_sha() below — contents only, sorted
# paths; do not paste a value produced by a different method.
DOCS_TREE_SHA256 = "8019fd04535820fe75865d17611a6cf0"

# The five Python types the corpus actually uses. Asserted exhaustive at build
# time: an unmapped type means the corpus grew a shape this generator has never
# seen, and guessing at it would put an invented type in the spec.
TYPE_MAP = {
    "int": "integer",
    "bool": "boolean",
    "str": "string",
    "dict": "object",
    "list": "array",
}

# G2. Each pattern is the mechanical form of "do not resurrect a Week 6 control".
# The gate scans every byte under api/ with no per-file carve-outs, which is why
# these strings are not spelled out in api/endpoints.yaml's own comments.
FORBIDDEN_IN_API: list[tuple[str, str, str]] = [
    (r"/v1/", "w6-031", "a version-1 path would make w6-031's question answerable"),
    (r"embeddings", "w6-031", "that operation name would make w6-031 answerable"),
    (r"(?i)oauth", "w6-032", "w6-032 expects 'only api_key auth is documented'"),
    (r"(?i)\bbearer\b", "w6-032", "a second auth scheme has the same effect"),
    (r"\bv?\d+\.\d+\.\d+\b", "w6-030", "a release version would make w6-030 answerable"),
    (r"(?i)rate.?limit(?!_rps)", "w6-031", "a documented throughput ceiling would make w6-031 answerable"),
]

# The single documented exemption to the release-version ban.
#
# An OpenAPI document must declare which version of the OpenAPI *format* it uses,
# and that field is a semver by specification. It is a different namespace from a
# product release: no agent can answer "is there a changelog for v3.2.0" from the
# document-format version. The exemption is one exact string, not a pattern class,
# so it cannot quietly widen. Everything else under api/ is scanned unmodified.
OPENAPI_FORMAT_VERSION = "3.1.0"
SEMVER_EXEMPT_LINE = f'"openapi": "{OPENAPI_FORMAT_VERSION}"'

NO_ENDPOINT_METHODS = {"Client.configure"}


# --------------------------------------------------------------------------
# the corpus: header-driven table parsing
# --------------------------------------------------------------------------


class ParamRow(BaseModel):
    """One row of a `## Parameters` table, read by column name."""

    name: str
    type: str
    default: str | None
    required: bool
    description: str = ""


class Page(BaseModel):
    sdk_version: str
    page_id: str
    title: str
    params: list[ParamRow] = Field(default_factory=list)
    example: str = ""
    anchor: str = ""

    @property
    def sdk_method(self) -> str:
        """`Client.send_batch` from the front-matter title `Client.send_batch()`."""
        return self.title.replace("()", "").strip()


def _split_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def parse_parameter_table(table_text: str) -> list[ParamRow]:
    """Parse a markdown table by its HEADER, never by column position.

    Eight of the ten tables carry a `description` column and two do not
    (`docs/v2/client-send.md`, `docs/v3/client-stream.md`). Reading column 5
    positionally would silently produce empty descriptions on some pages and
    IndexError on others, so the header row decides the layout per table.
    """
    lines = [ln for ln in table_text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return []
    header = [h.lower() for h in _split_row(lines[0])]
    if "name" not in header:
        return []
    index = {column: position for position, column in enumerate(header)}

    rows: list[ParamRow] = []
    for line in lines[2:]:  # 0 header, 1 separator
        cells = _split_row(line)
        if len(cells) < len(header):
            continue

        def cell(column: str) -> str:
            position = index.get(column)
            return cells[position] if position is not None else ""

        name = cell("name")
        if not name or name.lower() == "name":
            continue
        default = cell("default")
        rows.append(
            ParamRow(
                name=name,
                type=cell("type"),
                default=None if default in ("—", "-", "") else default,
                required=cell("required").lower() in ("yes", "true", "required"),
                description=cell("description"),
            )
        )
    return rows


def load_pages(docs_root: Path = DEFAULT_DOCS) -> dict[tuple[str, str], Page]:
    """(sdk_version, page_id) -> Page, parsed from the real corpus."""
    pages: dict[tuple[str, str], Page] = {}
    for path in sorted(docs_root.rglob("*.md")):
        raw = path.read_text(encoding="utf-8")
        front, body = parse_front_matter(raw)
        version = front.get("sdk_version") or path.parent.name
        page_id = front.get("page_id") or path.stem

        params: list[ParamRow] = []
        example = ""
        section = ""
        for block in parse_blocks(body):
            if block.kind is BlockKind.heading:
                section = block.text.lstrip("#").strip().lower()
            elif block.kind is BlockKind.table and section.startswith("parameters"):
                params.extend(parse_parameter_table(block.text))
            elif block.kind is BlockKind.code_fence and section.startswith("example"):
                if not example:
                    example = block.text

        pages[(version, page_id)] = Page(
            sdk_version=version,
            page_id=page_id,
            title=front.get("title") or page_id,
            params=params,
            example=example,
            anchor=f"{version}/{page_id}#parameters",
        )
    return pages


def divergent_by_method(
    docs_root: Path = DEFAULT_DOCS,
) -> dict[tuple[str, str], tuple[str, str]]:
    """(page_id, parameter) -> (v2_default, v3_default) for GENUINE changes only.

    Method-scoped on purpose. See the module docstring: the name-scoped version
    in week6_assertions reports `base_url` as changed, which it is not.
    """
    pages = load_pages(docs_root)
    by_key: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for (version, page_id), page in pages.items():
        for row in page.params:
            if row.default is not None:
                by_key[(page_id, row.name)][version] = row.default
    return {
        key: (versions["v2"], versions["v3"])
        for key, versions in sorted(by_key.items())
        if "v2" in versions and "v3" in versions and versions["v2"] != versions["v3"]
    }


def added_parameters(docs_root: Path = DEFAULT_DOCS) -> dict[str, list[str]]:
    """page_id -> parameters present in v3 and absent in v2, on SHARED pages."""
    pages = load_pages(docs_root)
    shared = {p for (v, p) in pages if (("v2", p) in pages and ("v3", p) in pages)}
    out: dict[str, list[str]] = {}
    for page_id in sorted(shared):
        v2 = {r.name for r in pages[("v2", page_id)].params}
        v3 = {r.name for r in pages[("v3", page_id)].params}
        if v3 - v2:
            out[page_id] = sorted(v3 - v2)
    return out


def removed_parameters(docs_root: Path = DEFAULT_DOCS) -> dict[str, list[str]]:
    """page_id -> parameters present in v2 and absent in v3. Expected: empty."""
    pages = load_pages(docs_root)
    shared = {p for (v, p) in pages if (("v2", p) in pages and ("v3", p) in pages)}
    out: dict[str, list[str]] = {}
    for page_id in sorted(shared):
        v2 = {r.name for r in pages[("v2", page_id)].params}
        v3 = {r.name for r in pages[("v3", page_id)].params}
        if v2 - v3:
            out[page_id] = sorted(v2 - v3)
    return out


def added_methods(docs_root: Path = DEFAULT_DOCS) -> list[str]:
    """page_ids present in v3 and absent in v2."""
    pages = load_pages(docs_root)
    v2 = {p for (v, p) in pages if v == "v2"}
    v3 = {p for (v, p) in pages if v == "v3"}
    return sorted(v3 - v2)


def reconcile(docs_root: Path = DEFAULT_DOCS) -> dict[str, Any]:
    """The 8-vs-9 comparison against the name-scoped Week 6 function."""
    from app.rag.week6_assertions import divergent_defaults

    name_scoped = set(divergent_defaults(docs_root))
    method_scoped = divergent_by_method(docs_root)
    method_names = {param for (_page, param) in method_scoped}
    return {
        "name_scoped": sorted(name_scoped),
        "method_scoped": method_scoped,
        "method_names": sorted(method_names),
        "excluded": sorted(name_scoped - method_names),
        "subset_ok": method_names <= name_scoped,
    }


def docs_tree_sha(docs_root: Path = DEFAULT_DOCS) -> str:
    digest = hashlib.sha256()
    for path in sorted(docs_root.rglob("*")):
        if path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()[:32]


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def print_reconcile(report: dict[str, Any]) -> None:
    name_scoped = report["name_scoped"]
    method_scoped = report["method_scoped"]
    print("Version-divergent defaults: name-scoped (Week 6) vs method-scoped (Week 7)")
    print()
    print(f"  method-scoped genuine changes : {len(method_scoped)}")
    print(f"  name-scoped (week6_assertions): {len(name_scoped)}")
    print()
    header = f"  {'page_id':18} {'parameter':20} {'v2':>10} -> {'v3'}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for (page_id, param), (v2, v3) in sorted(method_scoped.items()):
        print(f"  {page_id:18} {param:20} {v2:>10} -> {v3}")
    print()
    for name in report["excluded"]:
        print(f"  EXCLUDED  {name}")
        print("            v3 send's per-call override defaults to None while v3")
        print("            configure's defaults to the URL. Two different parameters")
        print("            that share a name, not a version change.")
    print()
    print(f"  subset check (method names in name-scoped): {'OK' if report['subset_ok'] else 'FAILED'}")


def print_inventory(docs_root: Path = DEFAULT_DOCS) -> None:
    pages = load_pages(docs_root)
    added_p = added_parameters(docs_root)
    removed_p = removed_parameters(docs_root)
    print(f"pages parsed : {len(pages)}")
    header = f"  {'version/page':28} {'params':>6}  {'example':>7}  title"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for (version, page_id), page in sorted(pages.items()):
        print(
            f"  {version + '/' + page_id:28} {len(page.params):6}  "
            f"{'yes' if page.example else 'no':>7}  {page.title}"
        )
    print()
    print(f"added parameters (v3 only, shared pages): {sum(len(v) for v in added_p.values())}")
    for page_id, names in added_p.items():
        print(f"  {page_id:18} +{len(names):<3} {', '.join(names)}")
    print(f"added methods (v3-only pages)           : {added_methods(docs_root)}")
    print(f"REMOVED parameters                      : "
          f"{removed_p if removed_p else 'none  <- why the deprecations are authored'}")


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------


def load_endpoints(path: Path = ENDPOINTS_YAML) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"authored endpoint table not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _normalise_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


PROVENANCE = (
    "Generated by app/rag/week7_corpus.py from docs/. "
    "Paths, verbs, operationIds and deprecations are AUTHORED for Week 7 - see api/README.md."
)


def build_openapi(
    docs_root: Path = DEFAULT_DOCS, endpoints: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Compose the spec: authored routing, derived everything-else."""
    endpoints = endpoints or load_endpoints()
    pages = load_pages(docs_root)
    deprecations = {d["operation_id"]: d for d in endpoints["deprecations"]}

    unmapped = {
        row.type
        for page in pages.values()
        for row in page.params
        if row.type not in TYPE_MAP
    }
    if unmapped:
        raise ValueError(
            f"corpus uses Python types this generator has no mapping for: {sorted(unmapped)}. "
            "Guessing would put an invented type in the spec; extend TYPE_MAP deliberately."
        )

    # DERIVED: the one documented credential becomes the one security scheme.
    # This is what keeps w6-032 ('only api_key auth is documented') a refusal.
    api_key_rows = [
        row for page in pages.values() for row in page.params if row.name == "api_key"
    ]
    if not api_key_rows:
        raise ValueError("no api_key parameter found in the corpus; refusing to invent a scheme")

    paths: dict[str, Any] = {}
    for op in endpoints["operations"]:
        page = pages.get((op["api_version"], op["page_id"])) if op["page_id"] else None
        parameters = [
            {
                "name": row.name,
                "in": "body",
                "required": row.required,
                "schema": {"type": TYPE_MAP[row.type]}
                | ({"default": row.default} if row.default is not None else {}),
                "description": row.description,
            }
            for row in (page.params if page else [])
        ]
        operation: dict[str, Any] = {
            "operationId": op["operation_id"],
            "summary": (
                f"{op['sdk_method']}() on {op['api_version']}"
                if op["sdk_method"]
                else f"{op['operation_id']} ({op['api_version']}); no SDK binding in this version"
            ),
            "x-sdk-method": op["sdk_method"],
            "x-api-version": op["api_version"],
            "x-docs-anchor": page.anchor if page else None,
            "x-provenance": (
                f"parameters derived from docs/{op['api_version']}/{op['page_id']}.md; "
                "path, verb and operationId authored"
                if page
                else "no docs page for this operation; path, verb and operationId authored"
            ),
            "parameters": parameters,
            "responses": {"200": {"description": "Success."}},
        }
        if page and page.example:
            operation["x-sdk-example"] = page.example
        if op["operation_id"] in deprecations:
            record = deprecations[op["operation_id"]]
            operation["deprecated"] = True
            operation["x-deprecated-in"] = record["deprecated_in"]
            operation["x-replaced-by"] = record["replaced_by"]
            operation["x-deprecation-reason"] = record["reason"]
        paths.setdefault(op["path"], {})[op["method"].lower()] = operation

    return {
        "openapi": OPENAPI_FORMAT_VERSION,
        "info": {
            "title": endpoints["service"]["title"],
            "summary": "The HTTP surface behind the documented Python SDK.",
            "description": PROVENANCE,
            "x-provenance": {
                "generator": "app/rag/week7_corpus.py",
                "derived_from": str(docs_root),
                "authored_fields": [
                    "paths", "http methods", "operationId",
                    "deprecated", "x-deprecated-in", "x-replaced-by",
                ],
                "disclosure": "api/README.md",
            },
        },
        "servers": [{"url": endpoints["service"]["server"]}],
        "components": {
            "securitySchemes": {
                "apiKey": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "api_key",
                    "description": api_key_rows[0].description,
                }
            }
        },
        "security": [{"apiKey": []}],
        "paths": paths,
    }


def build_deprecations(openapi: dict[str, Any]) -> dict[str, Any]:
    """A projection of openapi.json, so the two files cannot drift."""
    records = []
    index: dict[str, dict[str, Any]] = {}
    for path, verbs in openapi["paths"].items():
        for verb, op in verbs.items():
            index[op["operationId"]] = {"path": path, "method": verb.upper(), "op": op}
    for operation_id, entry in sorted(index.items()):
        op = entry["op"]
        if not op.get("deprecated"):
            continue
        replacement = index.get(op["x-replaced-by"], {})
        replacement_op = replacement.get("op", {})
        records.append(
            {
                "operation_id": operation_id,
                "path": entry["path"],
                "method": entry["method"],
                "api_version": op["x-api-version"],
                "sdk_method": op["x-sdk-method"],
                "deprecated": True,
                "deprecated_in": op["x-deprecated-in"],
                "reason": op["x-deprecation-reason"],
                "replaced_by": {
                    "operation_id": op["x-replaced-by"],
                    "path": replacement.get("path"),
                    "method": replacement.get("method"),
                    "sdk_method": replacement_op.get("x-sdk-method"),
                    "api_version": replacement_op.get("x-api-version"),
                },
            }
        )
    current = sorted(oid for oid, e in index.items() if not e["op"].get("deprecated"))
    return {
        "x-provenance": PROVENANCE + " This file is a projection of api/openapi.json.",
        "deprecated": records,
        "current": current,
    }


def build_changelog(
    docs_root: Path = DEFAULT_DOCS, endpoints: dict[str, Any] | None = None
) -> str:
    """Every line derived. No release-version vocabulary, by design."""
    endpoints = endpoints or load_endpoints()
    pages = load_pages(docs_root)
    changed = divergent_by_method(docs_root)
    added_p = added_parameters(docs_root)
    removed_p = removed_parameters(docs_root)
    new_methods = added_methods(docs_root)

    lines: list[str] = [
        "# Changelog — v2 to v3",
        "",
        "GENERATED — do not edit. Reproduce with:",
        "",
        "    python -m app.rag.week7_corpus",
        "",
        "Every entry below is computed from the parameter tables in `docs/`. The HTTP",
        "operation bindings and deprecations referenced here are authored — see",
        "`api/README.md`.",
        "",
        "## Changed (default value)",
        "",
        "| method | parameter | v2 | v3 |",
        "| --- | --- | --- | --- |",
    ]
    for (page_id, param), (v2, v3) in sorted(changed.items()):
        method = pages[("v3", page_id)].sdk_method
        lines.append(f"| `{method}()` | `{param}` | `{v2}` | `{v3}` |")

    lines += ["", "## Added (parameter)", ""]
    for page_id, names in sorted(added_p.items()):
        method = pages[("v3", page_id)].sdk_method
        lines.append(f"- `{method}()` — {', '.join(f'`{n}`' for n in names)}")

    lines += ["", "## Added (method)", ""]
    for page_id in new_methods:
        page = pages[("v3", page_id)]
        lines.append(f"- `{page.sdk_method}()` — new in v3, {len(page.params)} parameters")

    total_removed = sum(len(v) for v in removed_p.values())
    lines += [
        "",
        "## Removed",
        "",
        (
            "**Removed: none.** v3 adds to v2 and takes nothing away; every v2 parameter is "
            "present in v3 with the same type, and no v2 method was dropped. That is computed, "
            "not asserted — and it is why the deprecations in `api/deprecations.json` are all "
            "at the HTTP layer and all authored. See `api/README.md`."
            if total_removed == 0
            else f"{total_removed} parameters removed: {removed_p}"
        ),
        "",
    ]
    return "\n".join(lines) + "\n"


def render_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def generate(docs_root: Path = DEFAULT_DOCS) -> dict[Path, str]:
    endpoints = load_endpoints()
    openapi = build_openapi(docs_root, endpoints)
    return {
        OPENAPI_JSON: render_json(openapi),
        DEPRECATIONS_JSON: render_json(build_deprecations(openapi)),
        CHANGELOG_MD: build_changelog(docs_root, endpoints),
    }


# --------------------------------------------------------------------------
# gates
# --------------------------------------------------------------------------


def scan_forbidden(api_dir: Path = API_DIR) -> list[str]:
    """G2 — no byte under api/ may resurrect a Week 6 negative control."""
    problems: list[str] = []
    for path in sorted(api_dir.rglob("*")):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        scanned = text.replace(SEMVER_EXEMPT_LINE, "")
        for pattern, case, why in FORBIDDEN_IN_API:
            for match in re.finditer(pattern, scanned):
                problems.append(f"{path}: {match.group(0)!r} would break {case} — {why}")
    return problems


def check_anchors(endpoints: dict[str, Any]) -> list[str]:
    """Every authored deprecation must be pinned to real docs prose."""
    problems: list[str] = []
    for record in endpoints["deprecations"]:
        anchor = record["anchor"]
        source = Path(anchor["file"])
        if not source.is_file():
            problems.append(f"{record['operation_id']}: anchor file {source} does not exist")
            continue
        body = _normalise_ws(source.read_text(encoding="utf-8"))
        if _normalise_ws(anchor["quote"]) not in body:
            problems.append(
                f"{record['operation_id']}: anchor quote is not a literal substring of {source}"
            )
    return problems


def validate(docs_root: Path = DEFAULT_DOCS) -> tuple[list[str], list[str]]:
    failures: list[str] = []
    warnings: list[str] = []

    # G1 — docs/ untouched
    actual = docs_tree_sha(docs_root)
    if actual != DOCS_TREE_SHA256:
        failures.append(
            f"G1: docs/ tree sha is {actual}, expected {DOCS_TREE_SHA256}. "
            "The corpus moved; every Week 4/5/6 number measured against it is now suspect."
        )
    try:
        dirty = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", str(docs_root)], timeout=10
        ).returncode
        if dirty:
            failures.append("G1: docs/ has uncommitted changes")
    except Exception as exc:  # noqa: BLE001 - git absence must not mask the sha check
        warnings.append(f"G1: could not run git diff ({exc}); relying on the tree sha")

    # G2 — forbidden tokens
    failures.extend(f"G2: {problem}" for problem in scan_forbidden())

    # the two claims week6_assertions.py:19-27 makes about docs/, re-asserted
    corpus = " ".join(p.read_text(encoding="utf-8") for p in sorted(docs_root.rglob("*.md")))
    if re.search(r"(?i)deprecat|superseded|removed in|migration", corpus):
        failures.append(
            "docs/ now contains deprecation vocabulary; week6_assertions.py:19-27 says it does not"
        )
    if re.search(r"(?i)\b(GET|POST|PUT|DELETE|PATCH)\s+/", corpus):
        failures.append(
            "docs/ now contains an HTTP path; week6_assertions.py:19-27 says it does not"
        )

    # anchors + reconciliation
    try:
        endpoints = load_endpoints()
        failures.extend(f"anchor: {p}" for p in check_anchors(endpoints))
        ids = [op["operation_id"] for op in endpoints["operations"]]
        if len(ids) != len(set(ids)):
            failures.append("duplicate operation_id in endpoints.yaml")
        known = set(ids)
        for record in endpoints["deprecations"]:
            if record["operation_id"] not in known:
                failures.append(f"deprecation names unknown operation {record['operation_id']}")
            if record["replaced_by"] not in known:
                failures.append(f"replacement {record['replaced_by']} is not an operation")
        for method in NO_ENDPOINT_METHODS:
            if any(op["sdk_method"] == method for op in endpoints["operations"]):
                failures.append(f"{method} is client-local and must have no endpoint")
    except (FileNotFoundError, KeyError, TypeError) as exc:
        failures.append(f"endpoints.yaml unusable: {exc}")

    report = reconcile(docs_root)
    if not report["subset_ok"]:
        failures.append("method-scoped divergences are not a subset of the name-scoped set")
    if removed_parameters(docs_root):
        warnings.append(
            "docs/ now has removed parameters — genuine deprecations exist and the authored "
            "HTTP-layer fiction may no longer be necessary"
        )

    # derived files must be reproducible
    try:
        for path, rendered in generate(docs_root).items():
            if not path.is_file():
                warnings.append(f"{path} not generated yet")
            elif path.read_text(encoding="utf-8") != rendered:
                failures.append(f"{path} is stale — a fresh build differs. Re-run without --check.")
    except (FileNotFoundError, ValueError) as exc:
        failures.append(f"generation failed: {exc}")

    return failures, warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the Week 7 api/ layer from docs/.")
    parser.add_argument("--docs", type=Path, default=DEFAULT_DOCS)
    parser.add_argument("--reconcile", action="store_true", help="the 8-vs-9 divergence table")
    parser.add_argument("--inventory", action="store_true", help="what the parser sees in docs/")
    parser.add_argument("--check", action="store_true", help="re-derive and byte-diff the committed files")
    parser.add_argument("--validate", action="store_true", help="regression gates, no writes")
    args = parser.parse_args(argv)

    try:
        if args.reconcile:
            report = reconcile(args.docs)
            print_reconcile(report)
            return 0 if report["subset_ok"] else 1

        if args.inventory:
            print_inventory(args.docs)
            return 0

        if args.validate:
            failures, warnings = validate(args.docs)
            for warning in warnings:
                print(f"WARN  {warning}")
            for failure in failures:
                print(f"FAIL  {failure}")
            print()
            if failures:
                print(f"VALIDATION FAILED — {len(failures)} problem(s)")
                return 1
            print(f"VALIDATION PASSED — {len(warnings)} warning(s)")
            return 0

        rendered = generate(args.docs)

        if args.check:
            stale = [
                str(path)
                for path, text in rendered.items()
                if not path.is_file() or path.read_text(encoding="utf-8") != text
            ]
            if stale:
                print(f"stale or missing: {', '.join(stale)}", file=sys.stderr)
                return 1
            print(f"api/ matches a fresh build ({len(rendered)} files)")
            return 0

        API_DIR.mkdir(parents=True, exist_ok=True)
        for path, text in rendered.items():
            path.write_text(text, encoding="utf-8")
            print(f"wrote {path}  ({len(text.splitlines())} lines)")
        return 0

    except (FileNotFoundError, ValueError) as exc:
        print(f"week7 corpus failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
