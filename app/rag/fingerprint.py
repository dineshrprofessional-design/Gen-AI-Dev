"""A fingerprint of everything that must NOT change between two measured runs.

The Week 4 claim worth the most marks is "exactly one variable changed". That is
a claim about things that stayed still, so it should be checked mechanically
rather than asserted in prose. Each run records this fingerprint; `assert_same`
then names any field that moved.

The highest-value field is the pair `chunk_id_sha256` / `index_id_sha256`. The
first hashes the chunk ids a fresh chunking of `docs/` produces; the second
hashes the ids actually sitting in the index. If they disagree, the corpus was
edited without re-indexing — and a stale index does not error, it quietly
returns plausible numbers for a corpus that no longer exists.

Embedding vectors are deliberately not hashed. Floats are not bit-reproducible
across torch versions or thread counts, so a vector hash would false-alarm
constantly and teach everyone to ignore it.
"""

import hashlib
from pathlib import Path

from pydantic import BaseModel


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class IndexFingerprint(BaseModel):
    corpus_sha256: str
    doc_count: int
    chunk_count: int
    chunk_id_sha256: str
    index_chunk_count: int
    index_id_sha256: str
    strategy: str
    chunker_params: dict[str, int]
    embedder: str
    embedder_model: str
    collection: str
    golden_set_sha256: str

    @property
    def index_matches_corpus(self) -> bool:
        return self.chunk_id_sha256 == self.index_id_sha256


def fingerprint(
    strategy: str,
    embedder_name: str,
    docs_root: Path,
    persist_dir: Path,
    golden_path: Path,
) -> IndexFingerprint:
    from app.rag.chunkers import CHUNKERS
    from app.rag.embeddings import get_embedder
    from app.rag.loader import load_documents
    from app.rag.store import collection_name, get_store

    documents = load_documents(docs_root).documents
    chunker = CHUNKERS[strategy]()
    chunks = [c for d in documents for c in chunker.chunk(d)]

    corpus = "\n".join(
        f"{d.source_file}:{_sha256(d.text)}" for d in sorted(documents, key=lambda d: d.source_file)
    )
    chunk_ids = sorted(c.chunk_id for c in chunks)
    store = get_store(strategy, embedder_name, persist_dir)
    index_ids = sorted(store.ids())

    params = {
        k: v for k, v in vars(chunker).items() if isinstance(v, int) and not k.startswith("_")
    }
    # Construction does not load weights (they are lazy on first embed), so this
    # is cheap - and it catches a model swap that `name == "bge-m3"` would hide.
    embedder = get_embedder(embedder_name)
    model = getattr(embedder, "model_name", None) or embedder_name

    return IndexFingerprint(
        corpus_sha256=_sha256(corpus),
        doc_count=len(documents),
        chunk_count=len(chunks),
        chunk_id_sha256=_sha256("\n".join(chunk_ids)),
        index_chunk_count=len(index_ids),
        index_id_sha256=_sha256("\n".join(index_ids)),
        strategy=strategy,
        chunker_params=params,
        embedder=embedder_name,
        embedder_model=str(model),
        collection=collection_name(strategy, embedder_name),
        golden_set_sha256=_sha256(golden_path.read_text(encoding="utf-8")),
    )


def assert_same(a: IndexFingerprint, b: IndexFingerprint) -> list[str]:
    """Field names that differ. Empty means nothing about the setup moved."""
    return [f for f in a.model_fields if getattr(a, f) != getattr(b, f)]
