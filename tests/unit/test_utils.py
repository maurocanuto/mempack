"""Tests for utils module."""

import pytest
import tempfile
from pathlib import Path

from mempack.utils import (
    atomic_write, compute_xxh3, compute_crc32, verify_checksum,
    chunk_text, normalize_text, count_tokens, Timer
)


def test_atomic_write():
    """Test atomic file writing."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = Path(tmp.name)
    
    try:
        # Write data
        test_data = b"Hello, World!"
        atomic_write(tmp_path, test_data)
        
        # Verify content
        assert tmp_path.exists()
        with open(tmp_path, 'rb') as f:
            assert f.read() == test_data
        
    finally:
        # Cleanup
        if tmp_path.exists():
            tmp_path.unlink()


def test_compute_xxh3():
    """Test XXH3 hash computation."""
    data = b"Hello, World!"
    hash1 = compute_xxh3(data)
    hash2 = compute_xxh3(data)
    
    # Same input should produce same hash
    assert hash1 == hash2
    
    # Different input should produce different hash
    hash3 = compute_xxh3(b"Different data")
    assert hash1 != hash3


def test_compute_crc32():
    """Test CRC32 hash computation."""
    data = b"Hello, World!"
    hash1 = compute_crc32(data)
    hash2 = compute_crc32(data)
    
    # Same input should produce same hash
    assert hash1 == hash2
    
    # Different input should produce different hash
    hash3 = compute_crc32(b"Different data")
    assert hash1 != hash3


def test_verify_checksum():
    """Test checksum verification."""
    data = b"Hello, World!"
    checksum = compute_xxh3(data)
    
    # Valid checksum should pass
    assert verify_checksum(data, checksum, "xxh3")
    
    # Invalid checksum should fail
    assert not verify_checksum(data, checksum + 1, "xxh3")


def test_chunk_text():
    """Test text chunking."""
    text = "This is a test sentence. This is another sentence. And one more sentence for testing."
    
    chunks = chunk_text(
        text=text,
        chunk_size=30,
        chunk_overlap=10,
        min_chunk_size=10
    )
    
    assert len(chunks) > 0
    assert all(len(chunk) >= 10 for chunk in chunks)
    
    # Check that chunks don't exceed size limit (with some tolerance)
    assert all(len(chunk) <= 40 for chunk in chunks)


def test_normalize_text():
    """Test text normalization."""
    text = "  Hello,   World!  \n\n  "
    normalized = normalize_text(text)
    
    assert normalized == "Hello, World!"
    
    # Test with excessive whitespace
    text2 = "Multiple    spaces   and\n\n\nnewlines"
    normalized2 = normalize_text(text2)
    
    assert "  " not in normalized2  # No double spaces
    assert "\n\n" not in normalized2  # No double newlines


def test_count_tokens():
    """Test token counting."""
    text = "Hello, World! This is a test."
    
    # Character count
    char_count = count_tokens(text, "char")
    assert char_count == len(text)
    
    # Word count
    word_count = count_tokens(text, "word")
    assert word_count == 7  # "Hello,", "World!", "This", "is", "a", "test."
    
    # Sentence count
    sentence_count = count_tokens(text, "sentence")
    assert sentence_count == 2


def test_timer():
    """Test Timer functionality."""
    timer = Timer()
    
    # Initially not running
    assert not timer.running
    assert timer.elapsed == 0.0
    
    # Start timer
    timer.start()
    assert timer.running
    
    # Stop timer
    elapsed = timer.stop()
    assert not timer.running
    assert elapsed >= 0.0
    assert timer.elapsed == elapsed
    
    # Reset timer
    timer.reset()
    assert not timer.running
    assert timer.elapsed == 0.0


def test_timer_context_manager():
    """Test Timer as context manager."""
    with Timer() as timer:
        assert timer.running
    
    assert not timer.running
    assert timer.elapsed >= 0.0
