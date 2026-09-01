"""Generate the trace population by driving the real app.

    python -m app.rag.make_traces            # ~150 organic + 10 demo questions
    python -m app.rag.make_traces --dry-run  # print the pool, call nothing

Week 5 analyses traces, and this repo had none — so the population is produced
here, honestly: every trace comes from a real request through the actual
/api/v1/chat/ask route (FastAPI TestClient, the same code path the UI uses),
with real retrieval and real generation. Nothing is synthesised.

## Why the pool is a recipe, not a list someone curated

The task's second listed mistake is sampling the questions you already know are
broken — a curated sample gives fictional frequencies. This pool is therefore
built mechanically:

- one question per parameter-table row in the corpus (the row is parsed out of
  the markdown, the phrasing template is chosen by a stable hash of the
  parameter name — no human picked which parameter gets which phrasing);
- two questions per page about the method itself;
- one cross-version question per parameter that appears in both the v2 and v3
  table of the same page;
- a no-underscore casual transform of the first dozen underscored parameters;
- three hand-written lists (prose/notes questions, multi-part questions, and
  out-of-corpus questions), written from the pages before any of this ran and
  committed as code so they can be judged.

The order is shuffled with a fixed seed. The out-of-corpus share (~20%) is
there because real users ask about things the docs don't cover, and a trace
file with no refusals in it would misrepresent the app.

## The demo set

Ten questions the team "always shows at the DX review": the UI's sample chips
plus Week 3's polished, fully-disambiguated gold questions. Traced separately
and NEVER mixed into the random population — comparing failure frequency
between the two sets is the bonus exercise.
"""

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

from app.rag.index import DEFAULT_DOCS
from app.rag.loader import load_documents

POOL_SEED = 20260831
OUT_DIR = Path("traces")
MANIFEST = OUT_DIR / "pool_manifest.json"

# Gap between generating requests. Groq's free tier rate-limits per minute;
# losing a trace to a 429 would quietly bias the population toward
# retrieval-only records.
THROTTLE_S = 2.2

_ROW = re.compile(r"^\|\s*([a-z_][a-z0-9_]*)\s*\|", re.MULTILINE)

PARAM_TEMPLATES = [
    "what is the default for {p}",
    "{p} default",
    "what type is {p}",
    "is {p} required",
    "how do I set {p}",
]

METHOD_TEMPLATES = [
    "what does {m} do",
    "show me an example of {m}",
]

# Written from the pages, before any trace was generated. Prose and Notes
# content, phrased the way a developer types.
PROSE_QUESTIONS = [
    "is 503 retried automatically",
    "what happens when force=True on close",
    "does close block forever if requests never finish",
    "how is jitter applied to retries",
    "are 4xx responses retried",
    "what does the client do when the endpoint returns 5xx",
    "does stream reconnect after a drop",
    "how long is an upload handle valid",
    "what happens if I delete the upload state file",
    "why is my batch slower than concurrency suggests",
    "does raising concurrency increase throughput",
    "is the connection pool created eagerly",
    "when are idle connections reaped",
    "does checksum slow down uploads",
    "can I call close twice",
]

MULTIPART_QUESTIONS = [
    "difference between timeout_ms and default_timeout_ms",
    "concurrency vs rate_limit_rps which one wins",
    "retry_backoff_ms vs max_retries how do they interact",
    "chunk_bytes on upload vs send_batch are they the same thing",
    "force vs drain on close",
    "pool_size vs pool_reap_seconds",
    "compare v2 and v3 close behaviour",
    "stream vs send for large responses",
]

# Things a real developer asks that the corpus does not cover. Verified absent
# from docs/ by grep before this file was committed.
OUT_OF_CORPUS_QUESTIONS = [
    "how do I authenticate with OAuth2",
    "what is the rate limit on the /v1/embeddings endpoint",
    "how much does the SDK cost per million tokens",
    "which python versions are supported",
    "how do I install the sdk",
    "is there a javascript version",
    "how do I configure a proxy with authentication",
    "can I pin the TLS certificate",
    "how do I turn on debug logging to a file",
    "does the client support HTTP/2",
    "how do I paginate results",
    "what error codes can send return",
    "is there a changelog for v3.2.0",
    "how do I set a global user agent for my org",
    "can I use the client from multiple threads",
    "does send_batch support streaming responses",
    "how do I mock the client in tests",
    "what is the maximum payload size for send",
    "how do I rotate my api key without downtime",
    "is there a sandbox environment",
    "how do I report a bug in the sdk",
    "can I disable telemetry for GDPR compliance",
    "what regions are the api servers in",
    "how do I retry only idempotent requests",
    "is there an async version of the client",
    "how do I cancel a single request in a batch",
    "what does error E1042 mean",
    "how do I get webhook callbacks when a batch finishes",
    "can the client cache responses",
    "what is the SLA for the api",
]

DEMO_QUESTIONS = [
    # the UI's sample chips
    "What is the default value of retry_backoff_ms?",
    "Which parameter is required when calling Client.send()?",
    "What is the default chunk_size for Client.stream()?",
    "How do I close a client without cancelling in-flight requests?",
    "What is the rate limit on the /v1/embeddings endpoint?",
    "How do I authenticate with OAuth2?",
    # Week 3's polished, fully-disambiguated gold questions
    "What is the default value of retry_backoff_ms on Client.send() in v3?",
    "What type is the max_redirects parameter on Client.send()?",
    "What status codes trigger an automatic retry by default?",
    "How is jitter applied to the retry backoff delay?",
]


def _stable_pick(options: list[str], key: str) -> str:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return options[int(digest[:8], 16) % len(options)]


def build_pool(docs_root: Path = DEFAULT_DOCS) -> dict:
    """The recipe. Deterministic given the corpus and POOL_SEED."""
    documents = load_documents(docs_root).documents

    param_rows: list[tuple[str, str, str]] = []  # (sdk_version, page_id, param)
    by_page: dict[str, dict[str, set[str]]] = {}
    for doc in documents:
        names = [n for n in _ROW.findall(doc.text) if n != "name"]
        for name in names:
            param_rows.append((doc.sdk_version, doc.page_id, name))
        by_page.setdefault(doc.page_id, {}).setdefault(doc.sdk_version, set()).update(names)

    questions: list[dict] = []

    for version, page_id, param in sorted(set(param_rows)):
        template = _stable_pick(PARAM_TEMPLATES, f"{version}/{page_id}/{param}")
        questions.append({"q": template.format(p=param), "kind": "param"})

    methods = {
        "client-send": "Client.send()",
        "client-stream": "Client.stream()",
        "client-close": "Client.close()",
        "client-configure": "Client.configure()",
        "client-batch": "Client.send_batch()",
        "client-upload": "Client.upload()",
    }
    for page_id, method in sorted(methods.items()):
        for template in METHOD_TEMPLATES:
            questions.append({"q": template.format(m=method), "kind": "method"})

    for page_id, versions in sorted(by_page.items()):
        if "v2" in versions and "v3" in versions:
            for param in sorted(versions["v2"] & versions["v3"]):
                questions.append(
                    {"q": f"did {param} change between v2 and v3", "kind": "xversion"}
                )

    underscored = sorted({p for _, _, p in param_rows if "_" in p})[:12]
    for param in underscored:
        questions.append({"q": f"{param.replace('_', ' ')} default", "kind": "casual"})

    questions += [{"q": q, "kind": "prose"} for q in PROSE_QUESTIONS]
    questions += [{"q": q, "kind": "multipart"} for q in MULTIPART_QUESTIONS]
    questions += [{"q": q, "kind": "out_of_corpus"} for q in OUT_OF_CORPUS_QUESTIONS]

    # dedupe (a casual transform can collide with a param question), then a
    # seeded shuffle so generation order carries no information.
    seen: set[str] = set()
    unique = []
    for item in questions:
        if item["q"].lower() not in seen:
            seen.add(item["q"].lower())
            unique.append(item)

    import random

    random.Random(POOL_SEED).shuffle(unique)
    return {
        "seed": POOL_SEED,
        "recipe": "see module docstring of app/rag/make_traces.py",
        "counts": {},
        "questions": unique,
    }


def run(pool: dict, demo: list[str], hybrid: bool = True) -> dict:
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    manifest = {
        "seed": pool["seed"],
        "hybrid": hybrid,
        "organic": [],
        "demo": [],
    }

    def ask(question: str) -> dict:
        for attempt in range(3):
            response = client.post(
                "/api/v1/chat/ask",
                json={"question": question, "k": 5, "hybrid": hybrid, "generate": True},
            )
            body = response.json()
            error = body.get("error") or ""
            if "429" in error or "rate limit" in error.lower():
                wait = 30 * (attempt + 1)
                print(f"    rate limited, waiting {wait}s", file=sys.stderr)
                time.sleep(wait)
                continue
            return body
        return body

    total = len(pool["questions"]) + len(demo)
    done = 0
    for item in pool["questions"]:
        body = ask(item["q"])
        manifest["organic"].append(
            {"trace_id": body.get("trace_id"), "kind": item["kind"], "q": item["q"]}
        )
        done += 1
        if done % 10 == 0:
            print(f"  {done}/{total}", file=sys.stderr)
        time.sleep(THROTTLE_S if body.get("answer") is not None else 0.3)

    for question in demo:
        body = ask(question)
        manifest["demo"].append({"trace_id": body.get("trace_id"), "q": question})
        done += 1
        time.sleep(THROTTLE_S if body.get("answer") is not None else 0.3)

    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the Week 5 trace population.")
    parser.add_argument("--docs", type=Path, default=DEFAULT_DOCS)
    parser.add_argument("--dry-run", action="store_true", help="print the pool only")
    parser.add_argument("--dense", action="store_true", help="trace dense instead of hybrid")
    args = parser.parse_args(argv)

    pool = build_pool(args.docs)
    kinds: dict[str, int] = {}
    for item in pool["questions"]:
        kinds[item["kind"]] = kinds.get(item["kind"], 0) + 1
    pool["counts"] = kinds

    print(f"pool: {len(pool['questions'])} questions, seed={pool['seed']}")
    for kind, count in sorted(kinds.items()):
        print(f"  {kind:<14} {count}")
    print(f"demo: {len(DEMO_QUESTIONS)} questions (traced separately)")

    if args.dry_run:
        for item in pool["questions"]:
            print(f"  [{item['kind']:<14}] {item['q']}")
        return 0

    manifest = run(pool, DEMO_QUESTIONS, hybrid=not args.dense)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {MANIFEST}")
    print(f"organic traces: {len(manifest['organic'])}   demo traces: {len(manifest['demo'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
