"""Vector store seam.

Two implementations behind one protocol: Chroma for a persistent index, and an
in-memory store that needs no dependencies. Both support metadata filtering,
which is what lets a query be scoped to one sdk_version later.
"""

from pathlib import Path
from typing import Any, Protocol

from app.rag.models import Chunk


def collection_name(strategy: str, embedder: str) -> str:
    """Both variables in the name, so a mix is impossible rather than unlikely.

    Vectors from two different embedders are not comparable, and chunks from
    two different chunkers must not share a collection — that would silently
    destroy the comparison the whole exercise is built on.
    """
    return f"{strategy}__{embedder}"


class SearchHit:
    """One retrieved chunk with its score and metadata."""

    __slots__ = ("chunk_id", "score", "text", "metadata")

    def __init__(
        self, chunk_id: str, score: float, text: str, metadata: dict[str, Any]
    ) -> None:
        self.chunk_id = chunk_id
        self.score = score
        self.text = text
        self.metadata = metadata

    @property
    def target(self) -> str:
        """`version/page#anchor` — the form a gold target is written in."""
        anchor = self.metadata.get("anchor") or ""
        base = f"{self.metadata.get('sdk_version')}/{self.metadata.get('page_id')}"
        return f"{base}#{anchor}" if anchor else base

    def __repr__(self) -> str:
        return f"SearchHit({self.chunk_id!r}, score={self.score:.4f})"


def _matches(metadata: dict[str, Any], where: dict[str, Any] | None) -> bool:
    return where is None or all(metadata.get(k) == v for k, v in where.items())


class VectorStore(Protocol):
    name: str

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...

    def search(
        self, vector: list[float], k: int = 5, where: dict[str, Any] | None = None
    ) -> list[SearchHit]: ...

    def count(self) -> int: ...

    def reset(self) -> None: ...


class InMemoryStore:
    """Pure-python cosine search. No dependencies, no I/O, no persistence."""

    def __init__(self, name: str = "memory") -> None:
        self.name = name
        self._rows: dict[str, tuple[list[float], str, dict[str, Any]]] = {}

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must be the same length")
        for chunk, vector in zip(chunks, vectors):
            self._rows[chunk.chunk_id] = (vector, chunk.text, chunk.to_metadata())

    def search(
        self, vector: list[float], k: int = 5, where: dict[str, Any] | None = None
    ) -> list[SearchHit]:
        hits = [
            SearchHit(cid, sum(a * b for a, b in zip(vector, vec)), text, meta)
            for cid, (vec, text, meta) in self._rows.items()
            if _matches(meta, where)
        ]
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]

    def count(self) -> int:
        return len(self._rows)

    def reset(self) -> None:
        self._rows.clear()


class ChromaStore:
    """Persistent store. chromadb is imported lazily so it stays optional."""

    def __init__(self, name: str, persist_dir: Path) -> None:
        import chromadb

        self.name = name
        self._persist_dir = persist_dir
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if not chunks:
            return
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=vectors,
            documents=[c.text for c in chunks],
            metadatas=[c.to_metadata() for c in chunks],
        )

    def search(
        self, vector: list[float], k: int = 5, where: dict[str, Any] | None = None
    ) -> list[SearchHit]:
        result = self._collection.query(
            query_embeddings=[vector], n_results=k, where=where or None
        )
        return [
            # Chroma returns cosine *distance*; convert to similarity so every
            # store reports scores the same way round.
            SearchHit(cid, 1.0 - dist, doc, meta)
            for cid, dist, doc, meta in zip(
                result["ids"][0],
                result["distances"][0],
                result["documents"][0],
                result["metadatas"][0],
            )
        ]

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        """Drop and recreate.

        Re-indexing without this leaves chunks from a previous chunker version
        behind — upsert updates and inserts, it never deletes. Stale chunks then
        pollute every later search and the numbers quietly become wrong.
        """
        self._client.delete_collection(self.name)
        self._collection = self._client.get_or_create_collection(
            name=self.name, metadata={"hnsw:space": "cosine"}
        )


def get_store(
    strategy: str, embedder: str, persist_dir: Path | None = None
) -> VectorStore:
    name = collection_name(strategy, embedder)
    return InMemoryStore(name) if persist_dir is None else ChromaStore(name, persist_dir)
