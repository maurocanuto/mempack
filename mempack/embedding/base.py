"""Base classes for embedding backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np

from ..errors import EmbeddingError


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    
    embeddings: np.ndarray
    """Generated embeddings (float32, shape: [batch_size, dim])"""
    
    model_name: str
    """Name of the model used"""
    
    model_hash: str
    """Hash of the model for verification"""
    
    processing_time_ms: float
    """Processing time in milliseconds"""
    
    batch_size: int
    """Number of texts processed"""
    
    dimensions: int
    """Embedding dimensions"""


class EmbeddingBackend(ABC):
    """Abstract base class for embedding backends."""
    
    def __init__(
        self,
        model_name: str,
        max_length: int = 512,
        normalize: bool = True,
        device: Optional[str] = None,
    ) -> None:
        """Initialize the embedding backend.
        
        Args:
            model_name: Name of the embedding model
            max_length: Maximum sequence length
            normalize: Whether to normalize embeddings
            device: Device to use (cpu, cuda, auto)
        """
        self.model_name = model_name
        self.max_length = max_length
        self.normalize = normalize
        self.device = device
        self._model = None
        self._model_hash = None
    
    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Get the embedding dimensions."""
        pass
    
    @property
    @abstractmethod
    def model_hash(self) -> str:
        """Get the model hash for verification."""
        pass
    
    @abstractmethod
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> EmbeddingResult:
        """Encode texts into embeddings.
        
        Args:
            texts: Text or list of texts to encode
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            Embedding result
            
        Raises:
            EmbeddingError: If encoding fails
        """
        pass
    
    @abstractmethod
    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text into an embedding.
        
        Args:
            text: Text to encode
            
        Returns:
            Embedding vector
            
        Raises:
            EmbeddingError: If encoding fails
        """
        pass
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text before encoding.
        
        Args:
            text: Text to preprocess
            
        Returns:
            Preprocessed text
        """
        # Basic preprocessing
        text = text.strip()
        
        # Truncate if too long
        if len(text) > self.max_length * 4:  # Rough character to token ratio
            text = text[:self.max_length * 4]
        
        return text
    
    def preprocess_texts(self, texts: List[str]) -> List[str]:
        """Preprocess multiple texts.
        
        Args:
            texts: Texts to preprocess
            
        Returns:
            Preprocessed texts
        """
        return [self.preprocess_text(text) for text in texts]
    
    def normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings if enabled.
        
        Args:
            embeddings: Raw embeddings
            
        Returns:
            Normalized embeddings
        """
        if not self.normalize:
            return embeddings
        
        # L2 normalization
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        return embeddings / norms
    
    def validate_texts(self, texts: Union[str, List[str]]) -> List[str]:
        """Validate and prepare texts for encoding.
        
        Args:
            texts: Text or list of texts
            
        Returns:
            List of validated texts
            
        Raises:
            EmbeddingError: If validation fails
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            raise EmbeddingError("No texts provided", self.model_name)
        
        # Check for empty texts
        empty_texts = [i for i, text in enumerate(texts) if not text.strip()]
        if empty_texts:
            raise EmbeddingError(f"Empty texts at indices: {empty_texts}", self.model_name)
        
        return texts
    
    def get_model_info(self) -> dict:
        """Get model information.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "model_hash": self.model_hash,
            "dimensions": self.dimensions,
            "max_length": self.max_length,
            "normalize": self.normalize,
            "device": self.device,
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(model='{self.model_name}', dims={self.dimensions})"
