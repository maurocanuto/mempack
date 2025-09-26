"""HNSW index implementation."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Union

import hnswlib
import numpy as np

from ..errors import IndexError
from ..logging import index_logger
from ..types import HNSWParams


class HNSWIndex:
    """HNSW (Hierarchical Navigable Small World) index."""
    
    def __init__(
        self,
        dimensions: int,
        max_elements: int = 0,
        M: int = 32,
        ef_construction: int = 200,
        ef_search: int = 64,
        allow_replace_deleted: bool = True,
    ) -> None:
        """Initialize the HNSW index.
        
        Args:
            dimensions: Vector dimensions
            max_elements: Maximum number of elements (0 = auto)
            M: Number of bi-directional links
            ef_construction: Size of dynamic candidate list during construction
            ef_search: Size of dynamic candidate list during search
            allow_replace_deleted: Whether to allow replacing deleted elements
        """
        self.dimensions = dimensions
        self.max_elements = max_elements
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.allow_replace_deleted = allow_replace_deleted
        
        self._index = None
        self._id_to_label = {}
        self._label_to_id = {}
        self._next_label = 0
        self._is_built = False
    
    def add_items(
        self,
        vectors: np.ndarray,
        ids: Optional[List[int]] = None,
    ) -> None:
        """Add vectors to the index.
        
        Args:
            vectors: Array of vectors (shape: [n_vectors, dimensions])
            ids: Optional list of IDs for the vectors
            
        Raises:
            IndexError: If adding items fails
        """
        if self._index is None:
            self._init_index()
        
        n_vectors = vectors.shape[0]
        
        if ids is None:
            ids = list(range(self._next_label, self._next_label + n_vectors))
        
        if len(ids) != n_vectors:
            raise IndexError(f"Number of IDs ({len(ids)}) must match number of vectors ({n_vectors})")
        
        try:
            # Convert IDs to labels
            labels = []
            for vector_id in ids:
                if vector_id in self._id_to_label:
                    # Replace existing
                    label = self._id_to_label[vector_id]
                else:
                    # Add new
                    label = self._next_label
                    self._next_label += 1
                    self._id_to_label[vector_id] = label
                    self._label_to_id[label] = vector_id
                labels.append(label)
            
            # Add to index
            self._index.add_items(vectors, labels)
            self._is_built = True
            
            index_logger.debug(f"Added {n_vectors} vectors to HNSW index")
            
        except Exception as e:
            raise IndexError(f"Failed to add items to HNSW index: {e}", "hnsw")
    
    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        ef_search: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search for similar vectors.
        
        Args:
            query_vector: Query vector (shape: [dimensions])
            k: Number of results to return
            ef_search: Search parameter (uses default if None)
            
        Returns:
            Tuple of (distances, labels)
            
        Raises:
            IndexError: If search fails
        """
        if self._index is None or not self._is_built:
            raise IndexError("Index not built or empty", "hnsw")
        
        if query_vector.shape[0] != self.dimensions:
            raise IndexError(f"Query vector dimension ({query_vector.shape[0]}) must match index dimension ({self.dimensions})", "hnsw")
        
        try:
            # Set search parameter
            if ef_search is not None:
                self._index.set_ef(ef_search)
            
            # Search
            labels, distances = self._index.knn_query(query_vector, k=k)
            
            # Convert labels to IDs
            ids = np.array([self._label_to_id[label] for label in labels[0]])
            
            return distances[0], ids
            
        except Exception as e:
            raise IndexError(f"Search failed: {e}", "hnsw")
    
    def search_batch(
        self,
        query_vectors: np.ndarray,
        k: int = 10,
        ef_search: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search for similar vectors in batch.
        
        Args:
            query_vectors: Array of query vectors (shape: [n_queries, dimensions])
            k: Number of results to return per query
            ef_search: Search parameter (uses default if None)
            
        Returns:
            Tuple of (distances, labels) where each has shape [n_queries, k]
            
        Raises:
            IndexError: If search fails
        """
        if self._index is None or not self._is_built:
            raise IndexError("Index not built or empty", "hnsw")
        
        if query_vectors.shape[1] != self.dimensions:
            raise IndexError(f"Query vector dimension ({query_vectors.shape[1]}) must match index dimension ({self.dimensions})", "hnsw")
        
        try:
            # Set search parameter
            if ef_search is not None:
                self._index.set_ef(ef_search)
            
            # Search
            labels, distances = self._index.knn_query(query_vectors, k=k)
            
            # Convert labels to IDs
            ids = np.array([[self._label_to_id[label] for label in query_labels] for query_labels in labels])
            
            return distances, ids
            
        except Exception as e:
            raise IndexError(f"Batch search failed: {e}", "hnsw")
    
    def get_item(self, vector_id: int) -> Optional[np.ndarray]:
        """Get a vector by ID.
        
        Args:
            vector_id: Vector ID
            
        Returns:
            Vector or None if not found
        """
        if self._index is None or not self._is_built:
            return None
        
        if vector_id not in self._id_to_label:
            return None
        
        try:
            label = self._id_to_label[vector_id]
            return self._index.get_items([label])[0]
        except Exception:
            return None
    
    def remove_item(self, vector_id: int) -> bool:
        """Remove a vector by ID.
        
        Args:
            vector_id: Vector ID
            
        Returns:
            True if removed, False if not found
        """
        if self._index is None or not self._is_built:
            return False
        
        if vector_id not in self._id_to_label:
            return False
        
        try:
            label = self._id_to_label[vector_id]
            self._index.mark_deleted(label)
            
            # Remove from mappings
            del self._id_to_label[vector_id]
            del self._label_to_id[label]
            
            return True
            
        except Exception:
            return False
    
    def get_stats(self) -> dict:
        """Get index statistics.
        
        Returns:
            Statistics dictionary
        """
        if self._index is None:
            return {
                "dimensions": self.dimensions,
                "max_elements": self.max_elements,
                "current_elements": 0,
                "is_built": False,
            }
        
        return {
            "dimensions": self.dimensions,
            "max_elements": self.max_elements,
            "current_elements": self._index.get_current_count(),
            "is_built": self._is_built,
            "M": self.M,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
        }
    
    def save(self, file_path: Union[str, Path]) -> None:
        """Save index to file.
        
        Args:
            file_path: Path to save the index
            
        Raises:
            IndexError: If saving fails
        """
        if self._index is None or not self._is_built:
            raise IndexError("Index not built or empty", "hnsw")
        
        try:
            self._index.save_index(str(file_path))
            index_logger.info(f"HNSW index saved to {file_path}")
        except Exception as e:
            raise IndexError(f"Failed to save HNSW index: {e}", "hnsw")
    
    def load(self, file_path: Union[str, Path]) -> None:
        """Load index from file.
        
        Args:
            file_path: Path to load the index from
            
        Raises:
            IndexError: If loading fails
        """
        try:
            # Create index without initializing
            self._index = hnswlib.Index(
                space='cosine',
                dim=self.dimensions,
            )
            
            # Load the index directly
            self._index.load_index(str(file_path))
            
            # Set ef_search parameter
            self._index.set_ef(self.ef_search)
            
            print(f"[DEBUG HNSW] Index loaded successfully")
            print(f"[DEBUG HNSW] Index element count: {self._index.get_current_count()}")
            print(f"[DEBUG HNSW] Index max elements: {self._index.get_max_elements()}")
            
            # Rebuild ID mappings (this is a limitation of hnswlib)
            # Since hnswlib doesn't save ID mappings, we need to reconstruct them
            # Assume labels are the same as IDs for now (simple case)
            element_count = self._index.get_current_count()
            if element_count > 0:
                # Reconstruct the mappings assuming labels 0..n-1 map to IDs 0..n-1
                self._label_to_id = {i: i for i in range(element_count)}
                self._id_to_label = {i: i for i in range(element_count)}
                print(f"[DEBUG HNSW] Reconstructed ID mappings for {element_count} elements")
            
            self._is_built = True
            
            index_logger.info(f"HNSW index loaded from {file_path}")
            
        except Exception as e:
            raise IndexError(f"Failed to load HNSW index: {e}", "hnsw")
    
    def _init_index(self) -> None:
        """Initialize the HNSW index."""
        if self._index is not None:
            return
        
        try:
            self._index = hnswlib.Index(
                space='cosine',
                dim=self.dimensions,
            )
            
            self._index.init_index(
                max_elements=self.max_elements,
                ef_construction=self.ef_construction,
                M=self.M,
                allow_replace_deleted=self.allow_replace_deleted,
            )
            
            self._index.set_ef(self.ef_search)
            
        except Exception as e:
            raise IndexError(f"Failed to initialize HNSW index: {e}", "hnsw")
    
    def clear(self) -> None:
        """Clear the index."""
        self._index = None
        self._id_to_label.clear()
        self._label_to_id.clear()
        self._next_label = 0
        self._is_built = False
    
    def __len__(self) -> int:
        """Get number of elements in the index."""
        if self._index is None:
            return 0
        return self._index.get_current_count()
    
    def __repr__(self) -> str:
        """String representation."""
        return f"HNSWIndex(dim={self.dimensions}, elements={len(self)}, built={self._is_built})"
