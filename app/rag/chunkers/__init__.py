"""Chunker registry — both strategies on this branch.

Same documents in, same Chunk shape out. The only difference is where they
choose to cut, which is what makes them comparable.
"""

from app.rag.chunkers.base import Chunker
from app.rag.chunkers.fixed import FixedSizeChunker
from app.rag.chunkers.structure import StructureAwareChunker

CHUNKERS: dict[str, type[Chunker]] = {
    "fixed": FixedSizeChunker,
    "structure": StructureAwareChunker,
}

__all__ = ["CHUNKERS", "Chunker", "FixedSizeChunker", "StructureAwareChunker"]
