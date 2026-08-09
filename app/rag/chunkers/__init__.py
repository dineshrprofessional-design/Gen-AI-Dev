"""Chunker registry — strategy A only on this branch.

A is the baseline: fixed-size windows with overlap, no knowledge of markdown
structure. It exists to be measured against, not to be good.
"""

from app.rag.chunkers.base import Chunker
from app.rag.chunkers.fixed import FixedSizeChunker

CHUNKERS: dict[str, type[Chunker]] = {
    "fixed": FixedSizeChunker,
}

__all__ = ["CHUNKERS", "Chunker", "FixedSizeChunker"]
