"""MemPack: A portable, fast knowledge pack with two-file ANN memory."""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "MemPack Contributors"

# Public API imports
from .api import MemPackEncoder, MemPackRetriever, MemPackChat
from .types import SearchHit, ChunkMeta, BuildStats, RetrieverStats

# CLI function
def cli():
    """Run the MemPack CLI."""
    from .cli import app
    app()

# Alias for easier access
def mempack_cli():
    """Run the MemPack CLI (alias for cli)."""
    cli()

__all__ = [
    "MemPackEncoder",
    "MemPackRetriever", 
    "MemPackChat",
    "SearchHit",
    "ChunkMeta",
    "BuildStats",
    "RetrieverStats",
    "cli",
    "mempack_cli",
]
