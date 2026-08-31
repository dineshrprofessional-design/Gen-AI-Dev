"""Okapi BM25 over the same chunks the vector index holds.

Sparse lexical retrieval, pure stdlib. This exists because dense embeddings are
structurally bad at exact tokens: a query naming `retry_backoff_ms` needs the
chunk that literally contains that string, and no amount of semantic similarity
encodes literal containment. BM25 does exactly that and nothing else.

## Why the corpus is re-chunked rather than read out of Chroma

`build_index` chunks `load_documents(docs_root)` with `CHUNKERS[strategy]()` and
default constructor arguments. Re-running that here is byte-identical to what
was embedded, by construction, and needs no new store surface. Reading the text
back out of the collection instead would couple BM25 to a possibly-stale index
and lean on `collection.get()` ordering, which is not contractual — and
deterministic tie-breaking depends on ordering being stable.

## Tokenisation: underscores kept, no subword splitting

`[a-z0-9_]+` keeps `retry_backoff_ms` as one token, which is the entire point.
Emitting `retry`/`backoff`/`ms` as extra terms was considered and rejected: it
collapses the IDF of `retry` (which appears in `retry_on`, `max_retries`, and
prose on nearly every v3 page), it inflates every document length and so
silently shifts the `b` length-normalisation, and it is a second tuning knob
that cannot be isolated under a one-change rule. The cost is that a query typed
"retry backoff" with a space gets no BM25 signal — which is what the dense arm
is for. Fusion is supposed to make the two arms cover for each other.

## The IDF variant matters on a corpus this small

`log(1 + (N - df + 0.5) / (df + 0.5))` is used rather than the classic
`log((N - df + 0.5) / (df + 0.5))`. The classic form goes **negative** for any
term appearing in more than half the documents. With 38 chunks where "client",
"parameters" and "default" appear almost everywhere, negative IDF *penalises* a
chunk for containing a common query word and inverts the ranking — quietly, and
in exactly the queries this exercise cares about.
"""

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.rag.models import Chunk

K1 = 1.5
B = 0.75

# Underscores are word characters here. See the module docstring.
_TOKEN = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass(frozen=True)
class BM25Hit:
    chunk_id: str
    score: float


def _matches(metadata: dict[str, Any], where: dict[str, Any] | None) -> bool:
    return where is None or all(metadata.get(k) == v for k, v in where.items())


class BM25Index:
    """An in-memory BM25 index over a chunk list."""

    def __init__(self, chunks: list[Chunk], k1: float = K1, b: float = B) -> None:
        self.k1 = k1
        self.b = b
        self.chunk_ids: list[str] = [c.chunk_id for c in chunks]
        self._chunks: dict[str, Chunk] = {c.chunk_id: c for c in chunks}
        self._tf: list[dict[str, int]] = []
        self._len: list[int] = []
        self._df: dict[str, int] = {}

        for chunk in chunks:
            tokens = tokenize(chunk.text)
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            self._tf.append(counts)
            self._len.append(len(tokens))
            for term in counts:
                self._df[term] = self._df.get(term, 0) + 1

        self._n = len(chunks)
        self._avgdl = (sum(self._len) / self._n) if self._n else 0.0

    def __len__(self) -> int:
        return self._n

    def chunk(self, chunk_id: str) -> Chunk:
        return self._chunks[chunk_id]

    def idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        if df == 0:
            return 0.0
        return math.log(1 + (self._n - df + 0.5) / (df + 0.5))

    def score_doc(self, query_tokens: list[str], doc_index: int) -> float:
        tf = self._tf[doc_index]
        length = self._len[doc_index]
        norm = self.k1 * (1 - self.b + self.b * (length / self._avgdl))
        total = 0.0
        for term in query_tokens:
            freq = tf.get(term)
            if not freq:
                continue
            total += self.idf(term) * (freq * (self.k1 + 1)) / (freq + norm)
        return total

    def search(
        self, query: str, k: int = 5, where: dict[str, Any] | None = None
    ) -> list[BM25Hit]:
        tokens = tokenize(query)
        if not tokens or not self._n:
            return []
        hits: list[BM25Hit] = []
        for index, chunk_id in enumerate(self.chunk_ids):
            if where is not None and not _matches(
                self._chunks[chunk_id].to_metadata(), where
            ):
                continue
            score = self.score_doc(tokens, index)
            if score > 0.0:
                hits.append(BM25Hit(chunk_id, score))
        # Lexicographic id as the tie-break, never dict order: a tie decided by
        # insertion order is not reproducible across runs or platforms.
        hits.sort(key=lambda h: (-h.score, h.chunk_id))
        return hits[:k]


@lru_cache(maxsize=None)
def get_bm25_index(strategy: str = "structure", docs_root: Path | None = None) -> BM25Index:
    """Build once per process, mirroring how `get_embedder` is cached.

    Not persisted: building costs a handful of file reads and a markdown parse,
    with no model load. Persisting would buy milliseconds and cost a file
    format plus a staleness bug surface.
    """
    from app.rag.chunkers import CHUNKERS
    from app.rag.index import DEFAULT_DOCS
    from app.rag.loader import load_documents

    root = docs_root or DEFAULT_DOCS
    documents = load_documents(root).documents
    chunker = CHUNKERS[strategy]()
    chunks = [c for d in documents for c in chunker.chunk(d)]
    return BM25Index(chunks)
