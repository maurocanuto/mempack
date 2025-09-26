"""Tests for types module."""

import pytest
import numpy as np

from mempack.types import (
    SearchHit, ChunkMeta, BuildStats, RetrieverStats,
    Chunk, BlockInfo, PackConfig, IndexConfig, HNSWParams
)


def test_search_hit():
    """Test SearchHit creation."""
    hit = SearchHit(
        score=0.95,
        id=1,
        text="Test text",
        meta={"source": "test.md"}
    )
    
    assert hit.score == 0.95
    assert hit.id == 1
    assert hit.text == "Test text"
    assert hit.meta["source"] == "test.md"


def test_chunk_meta():
    """Test ChunkMeta creation."""
    meta = ChunkMeta(
        source="test.md",
        timestamp=1234567890.0,
        tags=["test", "example"],
        custom={"key": "value"}
    )
    
    assert meta.source == "test.md"
    assert meta.timestamp == 1234567890.0
    assert meta.tags == ["test", "example"]
    assert meta.custom["key"] == "value"


def test_build_stats():
    """Test BuildStats creation."""
    stats = BuildStats(
        chunks=100,
        blocks=10,
        vectors=100,
        bytes_written=1024,
        build_time_ms=500.0,
        embedding_time_ms=200.0,
        compression_ratio=2.5
    )
    
    assert stats.chunks == 100
    assert stats.blocks == 10
    assert stats.vectors == 100
    assert stats.bytes_written == 1024
    assert stats.build_time_ms == 500.0
    assert stats.embedding_time_ms == 200.0
    assert stats.compression_ratio == 2.5


def test_retriever_stats():
    """Test RetrieverStats creation."""
    stats = RetrieverStats(
        cache_hits=50,
        cache_misses=10,
        avg_fetch_ms=1.5,
        total_searches=60,
        avg_search_ms=25.0
    )
    
    assert stats.cache_hits == 50
    assert stats.cache_misses == 10
    assert stats.avg_fetch_ms == 1.5
    assert stats.total_searches == 60
    assert stats.avg_search_ms == 25.0


def test_chunk():
    """Test Chunk creation."""
    meta = ChunkMeta(source="test.md")
    embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    
    chunk = Chunk(
        id=1,
        text="Test text",
        meta=meta,
        embedding=embedding,
        block_id=0,
        offset_in_block=100
    )
    
    assert chunk.id == 1
    assert chunk.text == "Test text"
    assert chunk.meta.source == "test.md"
    assert np.array_equal(chunk.embedding, embedding)
    assert chunk.block_id == 0
    assert chunk.offset_in_block == 100


def test_block_info():
    """Test BlockInfo creation."""
    block = BlockInfo(
        id=1,
        uncompressed_size=1024,
        compressed_size=512,
        first_chunk_id=0,
        last_chunk_id=9,
        checksum=0x12345678,
        offset=1000
    )
    
    assert block.id == 1
    assert block.uncompressed_size == 1024
    assert block.compressed_size == 512
    assert block.first_chunk_id == 0
    assert block.last_chunk_id == 9
    assert block.checksum == 0x12345678
    assert block.offset == 1000


def test_pack_config():
    """Test PackConfig creation."""
    config = PackConfig(
        version=1,
        compressor="zstd",
        chunk_size=300,
        chunk_overlap=50,
        embedding_model="all-MiniLM-L6-v2",
        embedding_dim=384,
        index_type="hnsw"
    )
    
    assert config.version == 1
    assert config.compressor == "zstd"
    assert config.chunk_size == 300
    assert config.chunk_overlap == 50
    assert config.embedding_model == "all-MiniLM-L6-v2"
    assert config.embedding_dim == 384
    assert config.index_type == "hnsw"


def test_hnsw_params():
    """Test HNSWParams creation."""
    params = HNSWParams(
        M=32,
        ef_construction=200,
        ef_search=64,
        max_elements=1000,
        allow_replace_deleted=True
    )
    
    assert params.M == 32
    assert params.ef_construction == 200
    assert params.ef_search == 64
    assert params.max_elements == 1000
    assert params.allow_replace_deleted is True
