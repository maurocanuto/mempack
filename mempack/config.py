"""Configuration management for MemPack."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


class EmbeddingConfig(BaseModel):
    """Configuration for embedding generation."""
    
    model: str = "all-MiniLM-L6-v2"
    """Embedding model identifier."""
    
    dimensions: int = Field(default=384, ge=1, le=4096)
    """Embedding dimensions."""
    
    batch_size: int = Field(default=64, ge=1, le=1024)
    """Batch size for embedding generation."""
    
    max_length: int = Field(default=512, ge=1, le=8192)
    """Maximum sequence length for the model."""
    
    normalize: bool = True
    """Whether to normalize embeddings."""
    
    device: Optional[str] = None
    """Device to use (cpu, cuda, auto)."""
    
    @field_validator('device')
    @classmethod
    def validate_device(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in ['cpu', 'cuda', 'auto']:
            raise ValueError("Device must be 'cpu', 'cuda', or 'auto'")
        return v


class ChunkingConfig(BaseModel):
    """Configuration for text chunking."""
    
    chunk_size: int = Field(default=300, ge=50, le=2000)
    """Target chunk size in characters."""
    
    chunk_overlap: int = Field(default=50, ge=0, le=200)
    """Overlap between chunks in characters."""
    
    min_chunk_size: int = Field(default=50, ge=10, le=500)
    """Minimum chunk size to keep."""
    
    split_on_sentences: bool = True
    """Whether to split on sentence boundaries."""
    
    sentence_endings: List[str] = Field(default_factory=lambda: ['.', '!', '?', '\n\n'])
    """Characters that indicate sentence endings."""
    
    @field_validator('chunk_overlap')
    @classmethod
    def validate_overlap(cls, v: int, info) -> int:
        if hasattr(info, 'data'):
            chunk_size = info.data.get('chunk_size', 300)
            if v >= chunk_size:
                raise ValueError("chunk_overlap must be less than chunk_size")
        return v


class CompressionConfig(BaseModel):
    """Configuration for compression."""
    
    algorithm: str = Field(default="zstd", pattern="^(zstd|deflate|none)$")
    """Compression algorithm."""
    
    level: int = Field(default=3, ge=1, le=22)
    """Compression level (1=fastest, 22=best)."""
    
    threads: int = Field(default=0, ge=0, le=32)
    """Number of threads for compression (0=auto)."""
    
    @field_validator('level')
    @classmethod
    def validate_level(cls, v: int, info) -> int:
        if hasattr(info, 'data'):
            algorithm = info.data.get('algorithm', 'zstd')
            if algorithm == 'zstd' and v > 22:
                raise ValueError("zstd compression level must be <= 22")
            elif algorithm == 'deflate' and v > 9:
                raise ValueError("deflate compression level must be <= 9")
        return v


class HNSWConfig(BaseModel):
    """Configuration for HNSW index."""
    
    M: int = Field(default=32, ge=4, le=128)
    """Number of bi-directional links."""
    
    ef_construction: int = Field(default=200, ge=50, le=2000)
    """Size of dynamic candidate list during construction."""
    
    ef_search: int = Field(default=64, ge=10, le=1000)
    """Size of dynamic candidate list during search."""
    
    max_elements: int = Field(default=0, ge=0)
    """Maximum number of elements (0=auto)."""
    
    allow_replace_deleted: bool = True
    """Whether to allow replacing deleted elements."""
    
    @field_validator('ef_search')
    @classmethod
    def validate_ef_search(cls, v: int, info) -> int:
        if hasattr(info, 'data'):
            ef_construction = info.data.get('ef_construction', 200)
            if v > ef_construction:
                raise ValueError("ef_search should not exceed ef_construction")
        return v


class ECCConfig(BaseModel):
    """Configuration for error correction."""
    
    enabled: bool = False
    """Whether ECC is enabled."""
    
    k: int = Field(default=10, ge=1, le=255)
    """Number of data blocks."""
    
    m: int = Field(default=2, ge=1, le=255)
    """Number of parity blocks."""
    
    block_size: int = Field(default=4096, ge=512, le=65536)
    """ECC block size in bytes."""
    
    @field_validator('m')
    @classmethod
    def validate_m(cls, v: int, info) -> int:
        if hasattr(info, 'data'):
            k = info.data.get('k', 10)
            if v > k:
                raise ValueError("Number of parity blocks (m) should not exceed data blocks (k)")
        return v


class IndexConfig(BaseModel):
    """Configuration for ANN index."""
    
    type: str = Field(default="hnsw", pattern="^(hnsw|ivfpq)$")
    """Index type."""
    
    dimensions: int = Field(default=384, ge=1, le=4096)
    """Vector dimensions."""
    
    hnsw: Optional[HNSWConfig] = Field(default=None)
    """HNSW-specific configuration."""
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.type == 'hnsw' and self.hnsw is None:
            self.hnsw = HNSWConfig()


class RetrieverConfig(BaseModel):
    """Configuration for retriever."""
    
    mmap: bool = True
    """Whether to use memory mapping."""
    
    block_cache_size: int = Field(default=1024, ge=1, le=10000)
    """Number of blocks to cache."""
    
    io_batch_size: int = Field(default=64, ge=1, le=1000)
    """Batch size for I/O operations."""
    
    ef_search: int = Field(default=64, ge=10, le=1000)
    """HNSW search parameter."""
    
    prefetch: bool = True
    """Whether to prefetch blocks."""
    
    max_results: int = Field(default=1000, ge=1, le=10000)
    """Maximum number of results to return."""


class MemPackConfig(BaseModel):
    """Main configuration for MemPack."""
    
    version: int = Field(default=1, ge=1)
    """Configuration version."""
    
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    """Embedding configuration."""
    
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    """Chunking configuration."""
    
    compression: CompressionConfig = Field(default_factory=CompressionConfig)
    """Compression configuration."""
    
    index: IndexConfig = Field(default_factory=IndexConfig)
    """Index configuration."""
    
    ecc: ECCConfig = Field(default_factory=ECCConfig)
    """Error correction configuration."""
    
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)
    """Retriever configuration."""
    
    # Build-time settings
    workers: int = Field(default=0, ge=0, le=32)
    """Number of worker threads (0=auto)."""
    
    progress: bool = True
    """Whether to show progress bars."""
    
    temp_dir: Optional[str] = None
    """Temporary directory for build operations."""
    
    # File settings
    pack_path: Optional[str] = None
    """Path to the .mpack file."""
    
    ann_path: Optional[str] = None
    """Path to the .ann file."""
    
    @field_validator('workers')
    @classmethod
    def validate_workers(cls, v: int) -> int:
        if v == 0:
            import os
            v = min(32, max(1, os.cpu_count() or 1))
        return v


def get_default_config() -> MemPackConfig:
    """Get the default MemPack configuration.
    
    Returns:
        Default configuration instance
    """
    return MemPackConfig()


def load_config(config_dict: Dict[str, Any]) -> MemPackConfig:
    """Load configuration from a dictionary.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        Configuration instance
        
    Raises:
        ValidationError: If configuration is invalid
    """
    return MemPackConfig(**config_dict)


def save_config(config: MemPackConfig, file_path: str) -> None:
    """Save configuration to a file.
    
    Args:
        config: Configuration to save
        file_path: Path to save the configuration
    """
    import json
    with open(file_path, 'w') as f:
        json.dump(config.dict(), f, indent=2)


def load_config_from_file(file_path: str) -> MemPackConfig:
    """Load configuration from a file.
    
    Args:
        file_path: Path to the configuration file
        
    Returns:
        Configuration instance
        
    Raises:
        ValidationError: If configuration is invalid
        FileNotFoundError: If file doesn't exist
    """
    import json
    with open(file_path, 'r') as f:
        config_dict = json.load(f)
    return load_config(config_dict)
