"""Chunker registry — strategy B only on this branch.

B splits on markdown headings and keeps tables and code fences whole. It uses
no overlap: the heading breadcrumb carries context instead.
"""

from app.rag.chunkers.base import Chunker
from app.rag.chunkers.structure import StructureAwareChunker

CHUNKERS: dict[str, type[Chunker]] = {
    "structure": StructureAwareChunker,
}

__all__ = ["CHUNKERS", "Chunker", "StructureAwareChunker"]
