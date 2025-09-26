"""Embedding backends for MemPack."""

from .base import EmbeddingBackend, EmbeddingResult
from .sentence_tfm import SentenceTransformerBackend
from .external import ExternalEmbeddingBackend

__all__ = [
    "EmbeddingBackend",
    "EmbeddingResult", 
    "SentenceTransformerBackend",
    "ExternalEmbeddingBackend",
]
