"""Embedder seam.

Everything downstream talks to the `Embedder` protocol, so the model is one
config value rather than a rewrite.

**The model must stay frozen across both chunker runs.** Changing the chunker
and the embedder together is the classic mistake: you get two numbers and no
way to know which change moved them. The store puts the embedder name in the
collection name to make an accidental mix impossible rather than merely
discouraged.
"""

import hashlib
import math
import re
from functools import lru_cache
from typing import Protocol

_TOKEN = re.compile(r"[a-z0-9_]+")

# BAAI/bge-m3 — 1024-dim, 8192-token context. The long context matters here:
# the structure-aware chunker deliberately emits oversized chunks rather than
# cutting a table, and those must embed without truncation.
BGE_M3 = "BAAI/bge-m3"


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class HashEmbedder:
    """Deterministic offline embedder — a test double, never a real result.

    Hashes tokens into fixed buckets with sublinear term weighting. It captures
    literal token overlap and nothing else: no synonyms, no semantics. Useful
    for proving the pipeline stores and filters correctly, useless for
    measuring retrieval quality. Any hit-rate computed from this is meaningless.
    """

    name = "hash"

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _bucket(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self.dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)

    def _embed_one(self, text: str) -> list[float]:
        counts: dict[int, int] = {}
        for token in _TOKEN.findall(text.lower()):
            bucket = self._bucket(token)
            counts[bucket] = counts.get(bucket, 0) + 1

        vector = [0.0] * self.dim
        for bucket, count in counts.items():
            vector[bucket] = 1.0 + math.log(count)

        norm = math.sqrt(sum(v * v for v in vector))
        return [v / norm for v in vector] if norm else vector


class BgeM3Embedder:
    """BAAI/bge-m3 via sentence-transformers. Dense vectors, CPU by default.

    Loaded lazily and cached on the instance: the weights are ~2.2 GB, so
    importing this module must not pull them in.
    """

    name = "bge-m3"
    dim = 1024

    def __init__(self, model_name: str = BGE_M3, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        # bge-m3 needs no query prefix, unlike the bge-v1.5 family which
        # required "Represent this sentence...". Queries and passages go
        # through the same encoder.
        return self.embed([text])[0]


EMBEDDERS = {"bge-m3": BgeM3Embedder, "hash": HashEmbedder}


@lru_cache(maxsize=None)
def get_embedder(name: str = "bge-m3") -> Embedder:
    """Cached: bge-m3's weights are ~2.2 GB and must load once per process.

    Indexing and searching both call this, and the evaluation runs it across
    two chunkers — without the cache that is several model loads per run.
    """
    if name not in EMBEDDERS:
        raise ValueError(
            f"unknown embedder {name!r}. Available: {', '.join(sorted(EMBEDDERS))}"
        )
    return EMBEDDERS[name]()
