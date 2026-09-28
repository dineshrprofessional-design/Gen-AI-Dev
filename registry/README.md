# `registry/`: stand-in package registry data for Week 9

## 0. What this is, disclosed first

**This is locally authored stand-in package registry data for the Week 9
experiment.** Every package name, release number, date and deprecation notice in
`packages.yaml` was written by hand for Week 9 Task Set E. None of it was
retrieved from a real package registry, and no real package is described. The
server that serves it, `app/mcp/servers/package_server.py`, is a local stand-in
built in this repository. It is **not** an official or third-party server.

Why it has to exist: Task Set E describes a package-registry MCP server that
DevRel runs, serving "version metadata, release dates, and deprecation notices
for any package name". This repository has no such server and no such data, so
both were authored here, and every result the server returns carries this
file's `provenance` line. The disclosure therefore reaches the model along with
the data.

## 1. Everything invented, in one table

| what | value | why this value |
|---|---|---|
| package name | `example-sdk` | `docs/` imports the SDK as `sdk` and never names a distribution. The name follows `api/`'s "Example SDK Service" and `api.example.com`. |
| releases | 2.0.0, 2.1.0, 3.0.0, 3.1.0, 3.2.0, with dates | The 2.x and 3.x lines are the SDK versions `docs/` calls v2 and v3. |
| deprecation notice | 2.0.0 and 2.1.0: the 2.x line reached end of life | A whole-line deprecation. |
| deprecation notice | 3.0.0: wheel published without `py.typed` | A single-release deprecation inside a line that is still active. It is a packaging defect, so it contradicts nothing `docs/` says about behaviour. |
| support lines | 3.x active; 2.x end-of-life | Served as a resource, not a tool (section 3). |

Five releases and two notices is the whole dataset. It is sized to exercise
each path of the server once: a current release, a release deprecated as part of
its line, a release deprecated on its own, a release line passed where a release
was expected, a release that does not exist, and a package that does not exist.

## 2. What it deliberately does not contain

- **No changelog, release notes or per-release change list.** Week 6 case
  `w6-030` asks for a v3.2.0 changelog and must still find none; a release date
  is not a changelog.
- **No HTTP paths, operation ids or API-operation deprecations.** Those are
  Week 7's `api/` layer.
- **No SDK behaviour, parameters or defaults.** Those are `docs/`.
- **No `/v1/`, embeddings, OAuth or rate limits**, which Week 6 negative
  controls `w6-031` and `w6-032` depend on staying absent.

It lives in `registry/` because `docs/` is the RAG corpus, which Week 6 and
Week 7 gates assert is unchanged, and `api/` is scanned by Week 7's G2 gate.
That gate forbids semver strings, so release numbers placed there would fail
it by design. Nothing in Weeks 6 to 8 reads this directory.

## 3. Package deprecation is not API-operation deprecation

| | package deprecation (here) | API-operation deprecation (Week 7 `api/`) |
|---|---|---|
| what is deprecated | one published **release** of a package, e.g. `example-sdk 2.1.0` | one **HTTP operation**, e.g. `sendMessageV2` (`POST /v2/messages`) |
| the question it answers | which build should I install or pin? | which endpoint should my code call? |
| the replacement | a newer release to install | another operation |
| where it is served | `get_package_deprecation`, package-registry server | Week 7's `check_deprecation`, not exposed over MCP |

A release being deprecated says nothing about any one operation. `3.0.0` is
deprecated for a packaging defect while every v3 operation is current.

Support status, meaning which release lines are maintained, is a third thing,
and it is served as an MCP **resource** (`registry://example-sdk/support`).
Task Set E names exposing the version-support matrix as a tool as a mistake:
it is a small table that is the same for every question, so the host should
attach it as context rather than spend a model turn fetching it.
