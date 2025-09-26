"""MemPack: A portable, fast knowledge pack with two-file ANN memory."""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "MemPack Contributors"

# Public API imports
from .api import MemPackEncoder, MemPackRetriever, MemPackChat
from .types import SearchHit, ChunkMeta, BuildStats, RetrieverStats

__all__ = [
    "MemPackEncoder",
    "MemPackRetriever", 
    "MemPackChat",
    "SearchHit",
    "ChunkMeta",
    "BuildStats",
    "RetrieverStats",
]
