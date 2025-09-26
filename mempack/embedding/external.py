"""External embedding backend for HTTP APIs."""

from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
import requests

from .base import EmbeddingBackend, EmbeddingResult
from ..errors import EmbeddingError
from ..logging import embedding_logger


class ExternalEmbeddingBackend(EmbeddingBackend):
    """External HTTP API embedding backend."""
    
    def __init__(
        self,
        model_name: str,
        api_url: str,
        api_key: Optional[str] = None,
        max_length: int = 512,
        normalize: bool = True,
        device: Optional[str] = None,
        dimensions: int = 384,
        timeout: int = 30,
        max_retries: int = 3,
    ) -> None:
        """Initialize the external embedding backend.
        
        Args:
            model_name: Name of the embedding model
            api_url: API endpoint URL
            api_key: API key for authentication
            max_length: Maximum sequence length
            normalize: Whether to normalize embeddings
            device: Device to use (ignored for external APIs)
            dimensions: Embedding dimensions
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
        """
        super().__init__(model_name, max_length, normalize, device)
        self.api_url = api_url
        self.api_key = api_key
        self.dimensions = dimensions
        self.timeout = timeout
        self.max_retries = max_retries
        self._model_hash = None
    
    @property
    def model_hash(self) -> str:
        """Get the model hash for verification."""
        if self._model_hash is None:
            self._model_hash = self._compute_model_hash()
        return self._model_hash
    
    def _compute_model_hash(self) -> str:
        """Compute hash of the model for verification."""
        hash_input = f"{self.model_name}_{self.api_url}_{self.dimensions}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    
    def _make_request(
        self,
        texts: List[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Make API request for embeddings.
        
        Args:
            texts: Texts to embed
            batch_size: Batch size (may be ignored by API)
            
        Returns:
            Embeddings array
            
        Raises:
            EmbeddingError: If request fails
        """
        headers = {
            "Content-Type": "application/json",
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Prepare request data
        data = {
            "model": self.model_name,
            "input": texts,
            "encoding_format": "float",
        }
        
        # Make request with retries
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    json=data,
                    headers=headers,
                    timeout=self.timeout,
                )
                
                if response.status_code == 200:
                    result = response.json()
                    embeddings = self._parse_response(result)
                    return embeddings
                else:
                    error_msg = f"API request failed: {response.status_code} - {response.text}"
                    if attempt < self.max_retries - 1:
                        embedding_logger.warning(f"Attempt {attempt + 1} failed: {error_msg}")
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    else:
                        raise EmbeddingError(error_msg, self.model_name)
                        
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    embedding_logger.warning(f"Request attempt {attempt + 1} failed: {e}")
                    time.sleep(2 ** attempt)
                    continue
                else:
                    raise EmbeddingError(f"Request failed: {e}", self.model_name)
        
        raise EmbeddingError("All retry attempts failed", self.model_name)
    
    def _parse_response(self, response: Dict[str, Any]) -> np.ndarray:
        """Parse API response to extract embeddings.
        
        Args:
            response: API response dictionary
            
        Returns:
            Embeddings array
            
        Raises:
            EmbeddingError: If response format is invalid
        """
        try:
            # OpenAI format
            if "data" in response:
                embeddings = []
                for item in response["data"]:
                    if "embedding" in item:
                        embeddings.append(item["embedding"])
                    else:
                        raise EmbeddingError("Invalid response format: missing embedding", self.model_name)
                
                embeddings = np.array(embeddings, dtype=np.float32)
                
                # Verify dimensions
                if embeddings.shape[1] != self.dimensions:
                    embedding_logger.warning(
                        f"Expected {self.dimensions} dimensions, got {embeddings.shape[1]}"
                    )
                
                return embeddings
            
            # Direct array format
            elif "embeddings" in response:
                embeddings = np.array(response["embeddings"], dtype=np.float32)
                return embeddings
            
            else:
                raise EmbeddingError("Invalid response format: no embeddings found", self.model_name)
                
        except (KeyError, ValueError, TypeError) as e:
            raise EmbeddingError(f"Failed to parse response: {e}", self.model_name)
    
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
            # Validate texts
            texts = self.validate_texts(texts)
            
            # Preprocess texts
            texts = self.preprocess_texts(texts)
            
            embedding_logger.debug(f"Encoding {len(texts)} texts via API")
            
            # Process in batches
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = self._make_request(batch_texts, batch_size)
                all_embeddings.append(batch_embeddings)
            
            # Combine all embeddings
            embeddings = np.vstack(all_embeddings)
            
            # Normalize if needed
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
    
    def get_model_info(self) -> dict:
        """Get model information.
        
        Returns:
            Dictionary with model information
        """
        info = super().get_model_info()
        info.update({
            "backend": "external_api",
            "api_url": self.api_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        })
        return info


class OpenAIEmbeddingBackend(ExternalEmbeddingBackend):
    """OpenAI embedding backend."""
    
    def __init__(
        self,
        model_name: str = "text-embedding-ada-002",
        api_key: Optional[str] = None,
        max_length: int = 8191,
        normalize: bool = True,
        dimensions: int = 1536,
        timeout: int = 30,
        max_retries: int = 3,
    ) -> None:
        """Initialize the OpenAI embedding backend.
        
        Args:
            model_name: OpenAI model name
            api_key: OpenAI API key
            max_length: Maximum sequence length
            normalize: Whether to normalize embeddings
            dimensions: Embedding dimensions
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
        """
        super().__init__(
            model_name=model_name,
            api_url="https://api.openai.com/v1/embeddings",
            api_key=api_key,
            max_length=max_length,
            normalize=normalize,
            dimensions=dimensions,
            timeout=timeout,
            max_retries=max_retries,
        )
    
    def get_model_info(self) -> dict:
        """Get model information.
        
        Returns:
            Dictionary with model information
        """
        info = super().get_model_info()
        info.update({
            "provider": "openai",
        })
        return info
