"""MemPack retriever for searching knowledge packs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from tqdm import tqdm

from .config import MemPackConfig, get_default_config
from .embedding import EmbeddingBackend, SentenceTransformerBackend
from .errors import EmbeddingError, IOError, ValidationError
from .index import ANNFile, HNSWIndex
from .logging import retriever_logger
from .pack import MemPackReader
from .types import SearchHit, RetrieverStats
from .utils import time_ms


class MemPackRetriever:
    """Retriever for searching MemPack knowledge packs."""
    
    def __init__(
        self,
        pack_path: Union[str, Path],
        ann_path: Union[str, Path],
        embedding_backend: Optional[EmbeddingBackend] = None,
        mmap: bool = True,
        block_cache_size: int = 1024,
        io_batch_size: int = 64,
        ef_search: int = 64,
        prefetch: bool = True,
    ) -> None:
        """Initialize the retriever.
        
        Args:
            pack_path: Path to the .mpack file
            ann_path: Path to the .ann file
            embedding_backend: Embedding backend for queries
            mmap: Whether to use memory mapping
            block_cache_size: Size of block cache
            io_batch_size: Batch size for I/O operations
            ef_search: HNSW search parameter
            prefetch: Whether to prefetch blocks
        """
        self.pack_path = Path(pack_path)
        self.ann_path = Path(ann_path)
        self.mmap = mmap
        self.block_cache_size = block_cache_size
        self.io_batch_size = io_batch_size
        self.ef_search = ef_search
        self.prefetch = prefetch
        
        # Initialize embedding backend
        self.embedding_backend = embedding_backend or self._create_default_embedding_backend()
        
        # Load files
        self._load_files()
        
        # Statistics
        self.stats = RetrieverStats()
    
    def _create_default_embedding_backend(self) -> EmbeddingBackend:
        """Create the default embedding backend.
        
        Returns:
            Default embedding backend
        """
        return SentenceTransformerBackend(
            model_name="all-MiniLM-L6-v2",  # Default model
            max_length=512,
            normalize=True,
        )
    
    def _load_files(self) -> None:
        """Load the pack and ANN files.
        
        Raises:
            IOError: If files cannot be loaded
        """
        try:
            # Load pack file
            self.pack_reader = MemPackReader(
                pack_path=self.pack_path,
                mmap_enabled=self.mmap,
                block_cache_size=self.block_cache_size,
            )
            
            # Load ANN file
            self.ann_file = ANNFile(self.ann_path, mmap_enabled=self.mmap)
            self.hnsw_index = self.ann_file.read()
            
            # Verify compatibility
            self._verify_compatibility()
            
            retriever_logger.info(f"MemPack loaded: {self.pack_path}")
            retriever_logger.info(f"Chunks: {len(self.pack_reader._toc.chunks)}")
            retriever_logger.info(f"Vectors: {len(self.hnsw_index)}")
            
        except Exception as e:
            raise IOError(f"Failed to load MemPack: {e}", str(self.pack_path))
    
    def _verify_compatibility(self) -> None:
        """Verify compatibility between pack and ANN files.
        
        Raises:
            ValidationError: If files are incompatible
        """
        # Check embedding dimensions
        pack_config = self.pack_reader.get_config()
        pack_dims = pack_config.get("embedding_dim", 0)
        ann_dims = self.hnsw_index.dimensions
        
        if pack_dims != ann_dims:
            raise ValidationError(
                f"Dimension mismatch: pack={pack_dims}, ann={ann_dims}",
                "dimensions"
            )
        
        # Check model compatibility
        pack_model = pack_config.get("embedding_model", "")
        if pack_model != self.embedding_backend.model_name:
            retriever_logger.warning(
                f"Model mismatch: pack={pack_model}, backend={self.embedding_backend.model_name}"
            )
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_meta: Optional[Dict[str, Any]] = None,
        ef_search: Optional[int] = None,
    ) -> List[SearchHit]:
        """Search for similar chunks.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_meta: Optional metadata filter
            ef_search: HNSW search parameter
            
        Returns:
            List of search hits
            
        Raises:
            EmbeddingError: If query embedding fails
            ValidationError: If parameters are invalid
        """
        if not query.strip():
            return []
        
        if top_k <= 0:
            raise ValidationError("top_k must be positive", "top_k")
        
        try:
            with time_ms() as search_timer:
                # Generate query embedding
                query_embedding = self.embedding_backend.encode_single(query)
                
                # Search HNSW index
                distances, chunk_ids = self.hnsw_index.search(
                    query_vector=query_embedding,
                    k=min(top_k, len(self.hnsw_index)),
                    ef_search=ef_search or self.ef_search,
                )
                
                # Get chunks
                chunks = self.pack_reader.get_chunks(list(chunk_ids))
                
                # Apply metadata filter
                if filter_meta:
                    chunks = self._filter_chunks_by_meta(chunks, filter_meta)
                
                # Create search hits
                hits = []
                for i, chunk in enumerate(chunks):
                    if chunk is not None:
                        # Convert distance to similarity score (higher is better)
                        score = 1.0 / (1.0 + distances[i])
                        
                        hit = SearchHit(
                            score=score,
                            id=chunk.id,
                            text=chunk.text,
                            meta=chunk.meta.__dict__,
                        )
                        hits.append(hit)
                
                # Sort by score (descending)
                hits.sort(key=lambda x: x.score, reverse=True)
                
                # Update statistics
                self.stats.total_searches += 1
                self.stats.avg_search_ms = (
                    (self.stats.avg_search_ms * (self.stats.total_searches - 1) + search_timer.elapsed * 1000) /
                    self.stats.total_searches
                )
                
                retriever_logger.debug(f"Search returned {len(hits)} hits in {search_timer.elapsed * 1000:.2f}ms")
                
                return hits[:top_k]
                
        except Exception as e:
            raise EmbeddingError(f"Search failed: {e}", self.embedding_backend.model_name)
    
    def search_batch(
        self,
        queries: List[str],
        top_k: int = 5,
        filter_meta: Optional[Dict[str, Any]] = None,
        ef_search: Optional[int] = None,
        show_progress: bool = False,
    ) -> List[List[SearchHit]]:
        """Search for multiple queries.
        
        Args:
            queries: List of search queries
            top_k: Number of results per query
            filter_meta: Optional metadata filter
            ef_search: HNSW search parameter
            show_progress: Whether to show progress bar
            
        Returns:
            List of search hit lists (one per query)
        """
        if not queries:
            return []
        
        results = []
        
        # Process queries in batches
        batch_size = self.io_batch_size
        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i:i + batch_size]
            
            # Generate embeddings for batch
            try:
                embedding_result = self.embedding_backend.encode(
                    texts=batch_queries,
                    batch_size=len(batch_queries),
                    show_progress=False,
                )
                
                # Search HNSW index
                distances, chunk_ids = self.hnsw_index.search_batch(
                    query_vectors=embedding_result.embeddings,
                    k=min(top_k, len(self.hnsw_index)),
                    ef_search=ef_search or self.ef_search,
                )
                
                # Process results
                for j, query in enumerate(batch_queries):
                    query_hits = []
                    
                    for k in range(len(chunk_ids[j])):
                        chunk_id = chunk_ids[j][k]
                        chunk = self.pack_reader.get_chunk(chunk_id)
                        
                        if chunk is not None:
                            # Apply metadata filter
                            if filter_meta and not self._matches_meta_filter(chunk.meta.__dict__, filter_meta):
                                continue
                            
                            # Convert distance to similarity score
                            score = 1.0 / (1.0 + distances[j][k])
                            
                            hit = SearchHit(
                                score=score,
                                id=chunk.id,
                                text=chunk.text,
                                meta=chunk.meta.__dict__,
                            )
                            query_hits.append(hit)
                    
                    # Sort by score
                    query_hits.sort(key=lambda x: x.score, reverse=True)
                    results.append(query_hits[:top_k])
                
            except Exception as e:
                retriever_logger.warning(f"Batch search failed for queries {i}-{i+len(batch_queries)-1}: {e}")
                # Add empty results for failed batch
                results.extend([[] for _ in batch_queries])
        
        return results
    
    def get_chunk_by_id(self, chunk_id: int) -> Optional[Dict[str, Any]]:
        """Get a chunk by ID.
        
        Args:
            chunk_id: Chunk ID
            
        Returns:
            Chunk data or None if not found
        """
        chunk = self.pack_reader.get_chunk(chunk_id)
        if chunk is None:
            return None
        
        return {
            "id": chunk.id,
            "text": chunk.text,
            "meta": chunk.meta.__dict__,
        }
    
    def get_chunks_by_ids(self, chunk_ids: List[int]) -> List[Dict[str, Any]]:
        """Get multiple chunks by IDs.
        
        Args:
            chunk_ids: List of chunk IDs
            
        Returns:
            List of chunk data
        """
        chunks = self.pack_reader.get_chunks(chunk_ids)
        return [
            {
                "id": chunk.id,
                "text": chunk.text,
                "meta": chunk.meta.__dict__,
            }
            for chunk in chunks
        ]
    
    def get_chunks_by_meta(
        self,
        meta_filter: Dict[str, Any],
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get chunks by metadata filter.
        
        Args:
            meta_filter: Metadata filter
            limit: Maximum number of chunks to return
            
        Returns:
            List of chunk data
        """
        chunks = self.pack_reader.search_chunks(meta_filter=meta_filter)
        
        if limit is not None:
            chunks = chunks[:limit]
        
        return [
            {
                "id": chunk.id,
                "text": chunk.text,
                "meta": chunk.meta.__dict__,
            }
            for chunk in chunks
        ]
    
    def _filter_chunks_by_meta(
        self,
        chunks: List[Any],
        meta_filter: Dict[str, Any],
    ) -> List[Any]:
        """Filter chunks by metadata.
        
        Args:
            chunks: List of chunks
            meta_filter: Metadata filter
            
        Returns:
            Filtered chunks
        """
        if not meta_filter:
            return chunks
        
        filtered = []
        for chunk in chunks:
            if chunk and self._matches_meta_filter(chunk.meta.__dict__, meta_filter):
                filtered.append(chunk)
        
        return filtered
    
    def _matches_meta_filter(self, meta: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """Check if metadata matches filter.
        
        Args:
            meta: Chunk metadata
            filter_dict: Filter criteria
            
        Returns:
            True if metadata matches filter
        """
        for key, value in filter_dict.items():
            if key not in meta:
                return False
            if meta[key] != value:
                return False
        return True
    
    def verify(self) -> bool:
        """Verify the integrity of the knowledge pack.
        
        Returns:
            True if pack is valid
        """
        try:
            # Verify pack file
            if not self.pack_reader.verify():
                return False
            
            # Verify ANN file
            ann_stats = self.ann_file.get_stats()
            if ann_stats["vector_count"] != len(self.hnsw_index):
                return False
            
            return True
            
        except Exception:
            return False
    
    def get_stats(self) -> RetrieverStats:
        """Get retriever statistics.
        
        Returns:
            Statistics object
        """
        # Update cache statistics
        pack_stats = self.pack_reader.get_stats()
        self.stats.cache_hits = pack_stats.get("cache_hits", 0)
        self.stats.cache_misses = pack_stats.get("cache_misses", 0)
        self.stats.avg_fetch_ms = pack_stats.get("avg_fetch_ms", 0.0)
        
        return self.stats
    
    def get_pack_stats(self) -> Dict[str, Any]:
        """Get pack file statistics.
        
        Returns:
            Statistics dictionary
        """
        return self.pack_reader.get_stats()
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics.
        
        Returns:
            Statistics dictionary
        """
        return self.hnsw_index.get_stats()
    
    def close(self) -> None:
        """Close the retriever and cleanup resources."""
        if hasattr(self, 'pack_reader'):
            self.pack_reader.close()
        
        if hasattr(self, 'ann_file'):
            self.ann_file.close()
    
    def __enter__(self) -> MemPackRetriever:
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
    
    def __del__(self) -> None:
        """Destructor."""
        self.close()
