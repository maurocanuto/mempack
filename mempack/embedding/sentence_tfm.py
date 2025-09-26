"""SentenceTransformers embedding backend."""

from __future__ import annotations

import hashlib
import time
from typing import List, Optional, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from .base import EmbeddingBackend, EmbeddingResult
from ..errors import EmbeddingError
from ..logging import embedding_logger


class SentenceTransformerBackend(EmbeddingBackend):
    """SentenceTransformers-based embedding backend."""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        max_length: int = 512,
        normalize: bool = True,
        device: Optional[str] = None,
        cache_folder: Optional[str] = None,
    ) -> None:
        """Initialize the SentenceTransformers backend.
        
        Args:
            model_name: Name of the SentenceTransformers model
            max_length: Maximum sequence length
            normalize: Whether to normalize embeddings
            device: Device to use (cpu, cuda, auto)
            cache_folder: Cache folder for models
        """
        super().__init__(model_name, max_length, normalize, device)
        self.cache_folder = cache_folder
        self._model = None
        self._model_hash = None
        self._dimensions = None
    
    @property
    def dimensions(self) -> int:
        """Get the embedding dimensions."""
        if self._dimensions is None:
            self._load_model()
        return self._dimensions
    
    @property
    def model_hash(self) -> str:
        """Get the model hash for verification."""
        if self._model_hash is None:
            self._load_model()
        return self._model_hash
    
    def _load_model(self) -> None:
        """Load the SentenceTransformers model."""
        if self._model is not None:
            return
        
        try:
            embedding_logger.info(f"Loading SentenceTransformers model: {self.model_name}")
            
            # Determine device
            device = self.device
            if device == "auto":
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Load model
            self._model = SentenceTransformer(
                self.model_name,
                cache_folder=self.cache_folder,
                device=device,
            )
            
            # Get dimensions
            self._dimensions = self._model.get_sentence_embedding_dimension()
            
            # Compute model hash
            self._model_hash = self._compute_model_hash()
            
            embedding_logger.info(f"Model loaded: {self.model_name}, dims={self._dimensions}")
            
        except Exception as e:
            raise EmbeddingError(f"Failed to load model {self.model_name}: {e}", self.model_name)
    
    def _compute_model_hash(self) -> str:
        """Compute hash of the model for verification."""
        try:
            # Get model configuration
            config = self._model._modules['0'].auto_model.config
            
            # Create hash from model name and config
            hash_input = f"{self.model_name}_{config.to_dict()}"
            return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
            
        except Exception:
            # Fallback to model name hash
            return hashlib.sha256(self.model_name.encode()).hexdigest()[:16]
    
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
        start_time = time.perf_counter()
        
        try:
            # Load model if needed
            self._load_model()
            
            # Validate texts
            texts = self.validate_texts(texts)
            
            # Preprocess texts
            texts = self.preprocess_texts(texts)
            
            embedding_logger.debug(f"Encoding {len(texts)} texts with batch_size={batch_size}")
            
            # Encode texts
            embeddings = self._model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize,
            )
            
            # Ensure float32
            embeddings = embeddings.astype(np.float32)
            
            # Normalize if not done by the model
            if not self.normalize:
                embeddings = self.normalize_embeddings(embeddings)
            
            processing_time = (time.perf_counter() - start_time) * 1000
            
            return EmbeddingResult(
                embeddings=embeddings,
                model_name=self.model_name,
                model_hash=self.model_hash,
                processing_time_ms=processing_time,
                batch_size=len(texts),
                dimensions=self.dimensions,
            )
            
        except Exception as e:
            raise EmbeddingError(f"Encoding failed: {e}", self.model_name)
    
    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text into an embedding.
        
        Args:
            text: Text to encode
            
        Returns:
            Embedding vector
            
        Raises:
            EmbeddingError: If encoding fails
        """
        result = self.encode([text], batch_size=1, show_progress=False)
        return result.embeddings[0]
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text before encoding.
        
        Args:
            text: Text to preprocess
            
        Returns:
            Preprocessed text
        """
        # Basic preprocessing
        text = text.strip()
        
        # Truncate if too long (rough estimate)
        if len(text) > self.max_length * 4:
            text = text[:self.max_length * 4]
        
        return text
    
    def get_model_info(self) -> dict:
        """Get model information.
        
        Returns:
            Dictionary with model information
        """
        info = super().get_model_info()
        info.update({
            "backend": "sentence_transformers",
            "cache_folder": self.cache_folder,
        })
        return info
