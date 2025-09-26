"""Text processing utilities for MemPack."""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from ..errors import ChunkingError


def chunk_text(
    text: str,
    chunk_size: int = 300,
    chunk_overlap: int = 50,
    min_chunk_size: int = 50,
    split_on_sentences: bool = True,
    sentence_endings: Optional[List[str]] = None,
) -> List[str]:
    """Split text into overlapping chunks.
    
    Args:
        text: Text to chunk
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks
        min_chunk_size: Minimum chunk size to keep
        split_on_sentences: Whether to split on sentence boundaries
        sentence_endings: Characters that indicate sentence endings
        
    Returns:
        List of text chunks
        
    Raises:
        ChunkingError: If chunking parameters are invalid
    """
    if chunk_overlap >= chunk_size:
        raise ChunkingError("chunk_overlap must be less than chunk_size")
    
    if chunk_size < min_chunk_size:
        raise ChunkingError("chunk_size must be at least min_chunk_size")
    
    if sentence_endings is None:
        sentence_endings = ['.', '!', '?', '\n\n']
    
    # Normalize text
    text = normalize_text(text)
    
    if len(text) <= chunk_size:
        return [text] if len(text) >= min_chunk_size else []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        if end >= len(text):
            # Last chunk
            chunk = text[start:]
            if len(chunk) >= min_chunk_size:
                chunks.append(chunk)
            break
        
        if split_on_sentences:
            # Try to split at sentence boundary
            chunk = text[start:end]
            best_split = find_sentence_split(chunk, sentence_endings)
            if best_split > 0:
                end = start + best_split
        
        chunk = text[start:end]
        if len(chunk) >= min_chunk_size:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = end - chunk_overlap
        if start >= len(text):
            break
    
    return chunks


def find_sentence_split(
    text: str,
    sentence_endings: List[str],
    min_split: int = 50,
) -> int:
    """Find the best sentence split point in text.
    
    Args:
        text: Text to analyze
        sentence_endings: Characters that indicate sentence endings
        min_split: Minimum split position
        
    Returns:
        Best split position (0 if no good split found)
    """
    if len(text) < min_split:
        return 0
    
    # Look for sentence endings from the end
    for i in range(len(text) - 1, min_split - 1, -1):
        if text[i] in sentence_endings:
            # Check if next character is whitespace or end of text
            if i + 1 >= len(text) or text[i + 1].isspace():
                return i + 1
    
    return 0


def normalize_text(text: str) -> str:
    """Normalize text for processing.
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text
    """
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    return text


def count_tokens(text: str, method: str = "char") -> int:
    """Count tokens in text using various methods.
    
    Args:
        text: Text to count
        method: Counting method ('char', 'word', 'sentence')
        
    Returns:
        Token count
    """
    if method == "char":
        return len(text)
    elif method == "word":
        return len(text.split())
    elif method == "sentence":
        # Simple sentence counting based on punctuation
        sentences = re.split(r'[.!?]+', text)
        return len([s for s in sentences if s.strip()])
    else:
        raise ValueError(f"Unknown counting method: {method}")


def extract_sentences(text: str) -> List[str]:
    """Extract sentences from text.
    
    Args:
        text: Text to extract sentences from
        
    Returns:
        List of sentences
    """
    # Split on sentence endings
    sentences = re.split(r'[.!?]+', text)
    
    # Clean up and filter empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add when truncating
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    if len(suffix) >= max_length:
        return suffix[:max_length]
    
    return text[:max_length - len(suffix)] + suffix


def clean_text(text: str) -> str:
    """Clean text by removing unwanted characters and normalizing.
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    # Remove control characters except newlines and tabs
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def detect_language(text: str) -> str:
    """Detect language of text (simple heuristic).
    
    Args:
        text: Text to analyze
        
    Returns:
        Detected language code ('en', 'unknown')
    """
    # Simple heuristic based on common English words
    english_words = {
        'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
        'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at'
    }
    
    words = set(text.lower().split())
    english_count = len(words.intersection(english_words))
    
    if english_count > 0:
        return 'en'
    else:
        return 'unknown'


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract keywords from text.
    
    Args:
        text: Text to extract keywords from
        max_keywords: Maximum number of keywords to return
        
    Returns:
        List of keywords
    """
    # Simple keyword extraction based on word frequency
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    
    # Count word frequencies
    word_counts = {}
    for word in words:
        if len(word) > 2:  # Skip very short words
            word_counts[word] = word_counts.get(word, 0) + 1
    
    # Sort by frequency and return top keywords
    keywords = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    return [word for word, count in keywords[:max_keywords]]
