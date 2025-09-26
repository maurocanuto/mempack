"""End-to-end tests for MemPack."""

import pytest
import tempfile
from pathlib import Path

from mempack import MemPackEncoder, MemPackRetriever
from mempack.config import MemPackConfig


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_texts():
    """Sample texts for testing."""
    return [
        {
            "text": "Artificial intelligence is a branch of computer science that aims to create intelligent machines.",
            "meta": {"topic": "AI", "source": "ai_intro.txt"}
        },
        {
            "text": "Machine learning is a subset of AI that enables computers to learn without being explicitly programmed.",
            "meta": {"topic": "ML", "source": "ml_intro.txt"}
        },
        {
            "text": "Deep learning uses neural networks with multiple layers to model complex patterns in data.",
            "meta": {"topic": "DL", "source": "dl_intro.txt"}
        }
    ]


def test_build_and_search(temp_dir, sample_texts):
    """Test building a knowledge pack and searching it."""
    # Create configuration
    config = MemPackConfig()
    config.chunking.chunk_size = 100
    config.chunking.chunk_overlap = 20
    config.embedding.model = "all-MiniLM-L6-v2"
    config.index.hnsw.M = 16
    config.index.hnsw.ef_construction = 100
    
    # Set paths
    pack_path = temp_dir / "test.mpack"
    ann_path = temp_dir / "test.ann"
    
    # Build knowledge pack
    encoder = MemPackEncoder(config=config)
    
    for text_data in sample_texts:
        encoder.add_text(text_data["text"], text_data["meta"])
    
    stats = encoder.build(pack_path=pack_path, ann_path=ann_path)
    
    # Verify build results
    assert stats.chunks > 0
    assert stats.vectors > 0
    assert pack_path.exists()
    assert ann_path.exists()
    
    # Search knowledge pack
    with MemPackRetriever(pack_path=pack_path, ann_path=ann_path) as retriever:
        # Test search
        hits = retriever.search("artificial intelligence", top_k=3)
        
        assert len(hits) > 0
        assert all(hit.score > 0 for hit in hits)
        assert all(hit.text for hit in hits)
        assert all(hit.meta for hit in hits)
        
        # Test get_chunk_by_id
        if hits:
            chunk_id = hits[0].id
            chunk = retriever.get_chunk_by_id(chunk_id)
            
            assert chunk is not None
            assert chunk["id"] == chunk_id
            assert chunk["text"]
            assert chunk["meta"]
        
        # Test verification
        assert retriever.verify()
        
        # Test statistics
        stats = retriever.get_stats()
        assert stats.total_searches >= 1


def test_build_from_directory(temp_dir):
    """Test building from a directory of files."""
    # Create test files
    test_dir = temp_dir / "test_docs"
    test_dir.mkdir()
    
    (test_dir / "doc1.txt").write_text("This is document 1 about machine learning.")
    (test_dir / "doc2.txt").write_text("This is document 2 about artificial intelligence.")
    (test_dir / "doc3.md").write_text("# Deep Learning\n\nDeep learning is a subset of machine learning.")
    
    # Create configuration
    config = MemPackConfig()
    config.chunking.chunk_size = 50
    config.embedding.model = "all-MiniLM-L6-v2"
    
    # Set paths
    pack_path = temp_dir / "test.mpack"
    ann_path = temp_dir / "test.ann"
    
    # Build from directory
    encoder = MemPackEncoder(config=config)
    encoder.add_directory(test_dir, pattern="*.{txt,md}")
    
    stats = encoder.build(pack_path=pack_path, ann_path=ann_path)
    
    # Verify results
    assert stats.chunks > 0
    assert pack_path.exists()
    assert ann_path.exists()
    
    # Search
    with MemPackRetriever(pack_path=pack_path, ann_path=ann_path) as retriever:
        hits = retriever.search("machine learning", top_k=5)
        assert len(hits) > 0


def test_metadata_filtering(temp_dir, sample_texts):
    """Test metadata filtering in search."""
    # Create configuration
    config = MemPackConfig()
    config.chunking.chunk_size = 100
    config.embedding.model = "all-MiniLM-L6-v2"
    
    # Set paths
    pack_path = temp_dir / "test.mpack"
    ann_path = temp_dir / "test.ann"
    
    # Build knowledge pack
    encoder = MemPackEncoder(config=config)
    
    for text_data in sample_texts:
        encoder.add_text(text_data["text"], text_data["meta"])
    
    encoder.build(pack_path=pack_path, ann_path=ann_path)
    
    # Search with metadata filter
    with MemPackRetriever(pack_path=pack_path, ann_path=ann_path) as retriever:
        # Search for AI topic
        hits = retriever.search("intelligence", top_k=10, filter_meta={"topic": "AI"})
        
        # All hits should have topic="AI"
        for hit in hits:
            assert hit.meta.get("topic") == "AI"
        
        # Search for ML topic
        hits = retriever.search("learning", top_k=10, filter_meta={"topic": "ML"})
        
        # All hits should have topic="ML"
        for hit in hits:
            assert hit.meta.get("topic") == "ML"


def test_batch_search(temp_dir, sample_texts):
    """Test batch search functionality."""
    # Create configuration
    config = MemPackConfig()
    config.chunking.chunk_size = 100
    config.embedding.model = "all-MiniLM-L6-v2"
    
    # Set paths
    pack_path = temp_dir / "test.mpack"
    ann_path = temp_dir / "test.ann"
    
    # Build knowledge pack
    encoder = MemPackEncoder(config=config)
    
    for text_data in sample_texts:
        encoder.add_text(text_data["text"], text_data["meta"])
    
    encoder.build(pack_path=pack_path, ann_path=ann_path)
    
    # Batch search
    with MemPackRetriever(pack_path=pack_path, ann_path=ann_path) as retriever:
        queries = ["artificial intelligence", "machine learning", "deep learning"]
        results = retriever.search_batch(queries, top_k=3)
        
        assert len(results) == len(queries)
        assert all(len(hits) > 0 for hits in results)


def test_error_handling(temp_dir):
    """Test error handling."""
    # Test with non-existent files
    pack_path = temp_dir / "nonexistent.mpack"
    ann_path = temp_dir / "nonexistent.ann"
    
    with pytest.raises(Exception):  # Should raise IOError
        MemPackRetriever(pack_path=pack_path, ann_path=ann_path)
    
    # Test with empty encoder
    config = MemPackConfig()
    encoder = MemPackEncoder(config=config)
    
    with pytest.raises(Exception):  # Should raise ValidationError
        encoder.build(pack_path=pack_path, ann_path=ann_path)
