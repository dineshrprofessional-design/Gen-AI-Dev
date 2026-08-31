"""Hybrid retrieval: dense + BM25, fused by Reciprocal Rank Fusion.

This is the Week 4 change, and it is the *only* one. It is reachable solely
through `index.search(..., hybrid=True)`; with the flag off, not a line of this
module executes.

## RRF fuses ranks, never scores

    score(d) = sum over arms of  1 / (FUSION_K + rank_arm(d))

Cosine similarity lives on roughly 0.4-0.75 here; BM25 term weights are
unbounded sums that happen to land around 1-3. Adding or averaging them is
meaningless, and any weighting that made it look reasonable would be a second
tuned parameter. RRF only needs each arm's *ordering*, which is why
`rrf_fuse` takes ranked id lists — scores are discarded at the boundary, so
mixing scales is not merely discouraged, it is unrepresentable.

A document missing from an arm contributes nothing, rather than an imputed
`pool_size + 1`. Imputation is a different algorithm needing its own defence.

## Why the returned score is still a cosine

`generate.answer()` refuses when `hits[0].score` falls below `SCORE_FLOOR`
(0.45). A raw RRF score maxes out at `2/61 = 0.033`, so putting fused values on
`SearchHit.score` would refuse *every* question while the retrieval metrics
looked fine — a retrieval experiment silently turning into a generation
regression. So `.score` remains the dense cosine on exactly the scale it had
before, and RRF rides along in `metadata` for reporting.

For a chunk BM25 surfaced but the dense arm did not return, the cosine is
recovered as a dot product against the stored vector. That is valid because
`BgeM3Embedder` normalises embeddings, so both vectors are unit length and the
dot product *is* the cosine that Chroma reports as `1.0 - distance`.

## Constants, deliberately not CLI-exposed

`k1`, `b`, `FUSION_K` and `FETCH_K` are frozen module constants. Tuning any of
them on 12 questions over 38 chunks would be overfitting, and — worse — a
second variable in an experiment whose whole claim is that one thing changed.
`FETCH_K` is not that second variable: a fusion that cannot see past rank `k`
is not fusion, and the branch holding it is unreachable when `hybrid=False`,
so the dense run still fetches exactly `k`.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.rag.bm25 import get_bm25_index
from app.rag.store import SearchHit, VectorStore

FUSION_K = 60  # the RRF constant the task specifies
FETCH_K = 25  # candidate pool per arm


@dataclass(frozen=True)
class FusedRank:
    chunk_id: str
    rrf: float
    dense_rank: int | None
    bm25_rank: int | None


def rrf_fuse(
    dense_ids: list[str], bm25_ids: list[str], fusion_k: int = FUSION_K
) -> list[FusedRank]:
    """Fuse two ranked id lists. Scores are not accepted, by design."""
    dense_at = {cid: r for r, cid in enumerate(dense_ids, start=1)}
    bm25_at = {cid: r for r, cid in enumerate(bm25_ids, start=1)}

    fused: list[FusedRank] = []
    for chunk_id in dict.fromkeys([*dense_ids, *bm25_ids]):
        d = dense_at.get(chunk_id)
        s = bm25_at.get(chunk_id)
        score = 0.0
        if d is not None:
            score += 1.0 / (fusion_k + d)
        if s is not None:
            score += 1.0 / (fusion_k + s)
        fused.append(FusedRank(chunk_id, score, d, s))

    # Exact RRF ties are common — a doc at ranks (i, j) ties with one at (j, i)
    # — and a tie across the rank-3 boundary flips hit-rate@3. Break it on the
    # chunk_id so the result is reproducible rather than dict-order dependent.
    fused.sort(key=lambda f: (-f.rrf, f.chunk_id))
    return fused


def hybrid_search(
    query: str,
    *,
    embedder,
    store: VectorStore,
    strategy: str = "structure",
    k: int = 5,
    where: dict[str, Any] | None = None,
    docs_root: Path | None = None,
    stats: dict[str, float] | None = None,
) -> list[SearchHit]:
    """Dense + BM25, fused by RRF, truncated to `k`.

    `embedder` and `store` arrive already constructed. `get_store` is not
    cached, so building one in here would charge the hybrid run an extra
    PersistentClient construction the dense run never paid — and that asymmetry
    would be misread as the cost of fusion.
    """
    t0 = time.perf_counter()
    vector = embedder.embed_query(query)
    t1 = time.perf_counter()

    dense = store.search(vector, k=FETCH_K, where=where)
    t2 = time.perf_counter()

    bm25 = get_bm25_index(strategy, docs_root).search(query, k=FETCH_K, where=where)
    t3 = time.perf_counter()

    fused = rrf_fuse([h.chunk_id for h in dense], [h.chunk_id for h in bm25])[:k]

    by_id = {h.chunk_id: h for h in dense}
    missing = [f.chunk_id for f in fused if f.chunk_id not in by_id]
    stored_vectors = store.vectors(missing) if missing else {}
    bm25_index = get_bm25_index(strategy, docs_root)

    hits: list[SearchHit] = []
    for rank in fused:
        dense_hit = by_id.get(rank.chunk_id)
        if dense_hit is not None:
            text, metadata, score = dense_hit.text, dict(dense_hit.metadata), dense_hit.score
        else:
            chunk = bm25_index.chunk(rank.chunk_id)
            text, metadata = chunk.text, chunk.to_metadata()
            vec = stored_vectors.get(rank.chunk_id)
            score = sum(a * b for a, b in zip(vector, vec)) if vec else 0.0
        metadata["rrf_score"] = rank.rrf
        metadata["dense_rank"] = rank.dense_rank
        metadata["bm25_rank"] = rank.bm25_rank
        hits.append(SearchHit(rank.chunk_id, score, text, metadata))
    t4 = time.perf_counter()

    if stats is not None:
        stats["embed_ms"] = (t1 - t0) * 1000
        stats["dense_ms"] = (t2 - t1) * 1000
        stats["bm25_ms"] = (t3 - t2) * 1000
        stats["fuse_ms"] = (t4 - t3) * 1000

    return hits
