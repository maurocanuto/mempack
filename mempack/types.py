"""Core data types and DTOs for MemPack."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np


@dataclass
class SearchHit:
    """A search result containing score, chunk ID, text, and metadata."""
    
    score: float
    """Similarity score (higher is more similar)."""
    
    id: int
    """Chunk ID in the knowledge pack."""
    
    text: str
    """The text content of the chunk."""
    
    meta: Dict[str, Any]
    """Metadata associated with the chunk."""


@dataclass
class ChunkMeta:
    """Metadata for a text chunk."""
    
    source: Optional[str] = None
    """Source file or identifier."""
    
    timestamp: Optional[float] = None
    """Creation timestamp."""
    
    tags: List[str] = field(default_factory=list)
    """Tags for filtering and organization."""
    
    custom: Dict[str, Any] = field(default_factory=dict)
    """Custom metadata fields."""


@dataclass
class BuildStats:
    """Statistics from building a knowledge pack."""
    
    chunks: int
    """Number of text chunks processed."""
    
    blocks: int
    """Number of compressed blocks created."""
    
    vectors: int
    """Number of embedding vectors generated."""
    
    bytes_written: int
    """Total bytes written to pack file."""
    
    build_time_ms: float
    """Total build time in milliseconds."""
    
    embedding_time_ms: float
    """Time spent on embedding generation in milliseconds."""
    
    compression_ratio: float
    """Compression ratio (uncompressed / compressed)."""


@dataclass
class RetrieverStats:
    """Statistics from retriever operations."""
    
    cache_hits: int = 0
    """Number of cache hits."""
    
    cache_misses: int = 0
    """Number of cache misses."""
    
    avg_fetch_ms: float = 0.0
    """Average block fetch time in milliseconds."""
    
    total_searches: int = 0
    """Total number of searches performed."""
    
    avg_search_ms: float = 0.0
    """Average search time in milliseconds."""


@dataclass
class Chunk:
    """A text chunk with metadata and optional embedding."""
    
    id: int
    """Unique chunk identifier."""
    
    text: str
    """The text content."""
    
    meta: ChunkMeta
    """Chunk metadata."""
    
    embedding: Optional[np.ndarray] = None
    """Optional embedding vector (float32)."""
    
    block_id: Optional[int] = None
    """ID of the block containing this chunk."""
    
    offset_in_block: Optional[int] = None
    """Offset within the block."""


@dataclass
class BlockInfo:
    """Information about a compressed block."""
    
    id: int
    """Unique block identifier."""
    
    uncompressed_size: int
    """Uncompressed size in bytes."""
    
    compressed_size: int
    """Compressed size in bytes."""
    
    first_chunk_id: int
    """ID of first chunk in this block."""
    
    last_chunk_id: int
    """ID of last chunk in this block."""
    
    checksum: int
    """XXH3 checksum of uncompressed data."""
    
    offset: int
    """Offset in the pack file."""


@dataclass
class PackConfig:
    """Configuration for a MemPack file."""
    
    version: int = 1
    """File format version."""
    
    compressor: str = "zstd"
    """Compression algorithm (zstd, deflate, none)."""
    
    chunk_size: int = 300
    """Target chunk size in characters."""
    
    chunk_overlap: int = 50
    """Overlap between chunks in characters."""
    
    embedding_model: str = "all-MiniLM-L6-v2"
    """Embedding model identifier."""
    
    embedding_dim: int = 384
    """Embedding vector dimension."""
    
    index_type: str = "hnsw"
    """Index type (hnsw, ivfpq)."""
    
    index_params: Dict[str, Any] = field(default_factory=dict)
    """Index-specific parameters."""
    
    ecc_enabled: bool = False
    """Whether error correction is enabled."""
    
    ecc_params: Optional[Dict[str, Any]] = None
    """Error correction parameters."""


@dataclass
class IndexConfig:
    """Configuration for an ANN index."""
    
    algorithm: str = "hnsw"
    """Index algorithm (hnsw, ivfpq)."""
    
    dimensions: int = 384
    """Vector dimensions."""
    
    vector_count: int = 0
    """Number of vectors in the index."""
    
    id_width: int = 4
    """Width of ID field in bytes."""
    
    params: Dict[str, Any] = field(default_factory=dict)
    """Algorithm-specific parameters."""


@dataclass
class HNSWParams:
    """HNSW index parameters."""
    
    M: int = 32
    """Number of bi-directional links."""
    
    ef_construction: int = 200
    """Size of dynamic candidate list during construction."""
    
    ef_search: int = 64
    """Size of dynamic candidate list during search."""
    
    max_elements: int = 0
    """Maximum number of elements (0 = auto)."""


@dataclass
class CompressionStats:
    """Compression statistics."""
    
    algorithm: str
    """Compression algorithm used."""
    
    original_size: int
    """Original uncompressed size."""
    
    compressed_size: int
    """Compressed size."""
    
    ratio: float
    """Compression ratio (original / compressed)."""
    
    time_ms: float
    """Compression time in milliseconds."""


@dataclass
class FileHeader:
    """Header information for MemPack files."""
    
    magic: bytes
    """Magic bytes for file identification."""
    
    version: int
    """File format version."""
    
    flags: int
    """File flags (compression, ECC, etc.)."""
    
    created_at: float
    """Creation timestamp."""
    
    section_offsets: Dict[str, int]
    """Offsets of major sections."""
    
    section_lengths: Dict[str, int]
    """Lengths of major sections."""


# Type aliases for common use cases
ChunkList = List[Chunk]
SearchResult = List[SearchHit]
MetadataFilter = Dict[str, Any]
EmbeddingVector = np.ndarray
