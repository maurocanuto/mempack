"""MemPack encoder for building knowledge packs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from tqdm import tqdm

from .config import MemPackConfig, get_default_config
from .embedding import EmbeddingBackend, SentenceTransformerBackend
from .errors import EmbeddingError, IOError, ValidationError
from .index import HNSWIndex, ANNFile
from .logging import builder_logger
from .pack import MemPackWriter
from .types import Chunk, ChunkMeta, BuildStats
from .utils import chunk_text, time_ms


class MemPackEncoder:
    """Encoder for building MemPack knowledge packs."""
    
    def __init__(
        self,
        config: Optional[MemPackConfig] = None,
        embedding_backend: Optional[EmbeddingBackend] = None,
    ) -> None:
        """Initialize the encoder.
        
        Args:
            config: MemPack configuration
            embedding_backend: Embedding backend to use
        """
        self.config = config or get_default_config()
        self.embedding_backend = embedding_backend or self._create_default_embedding_backend()
        
        # State
        self.chunks: List[Chunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self.chunk_id_counter = 0
    
    def _create_default_embedding_backend(self) -> EmbeddingBackend:
        """Create the default embedding backend.
        
        Returns:
            Default embedding backend
        """
        return SentenceTransformerBackend(
            model_name=self.config.embedding.model,
            max_length=self.config.embedding.max_length,
            normalize=self.config.embedding.normalize,
            device=self.config.embedding.device,
        )
    
    def add_text(
        self,
        text: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add text to be chunked and embedded.
        
        Args:
            text: Text content
            meta: Optional metadata
        """
        if not text.strip():
            return
        
        # Create chunk metadata
        chunk_meta = ChunkMeta()
        if meta:
            chunk_meta.source = meta.get("source")
            chunk_meta.timestamp = meta.get("timestamp")
            chunk_meta.tags = meta.get("tags", [])
            chunk_meta.custom = {k: v for k, v in meta.items() if k not in ["source", "timestamp", "tags"]}
        
        # Chunk the text
        text_chunks = chunk_text(
            text=text,
            chunk_size=self.config.chunking.chunk_size,
            chunk_overlap=self.config.chunking.chunk_overlap,
            min_chunk_size=self.config.chunking.min_chunk_size,
            split_on_sentences=self.config.chunking.split_on_sentences,
            sentence_endings=self.config.chunking.sentence_endings,
        )
        
        # Create chunks
        for i, chunk_text_content in enumerate(text_chunks):
            chunk = Chunk(
                id=self.chunk_id_counter,
                text=chunk_text_content,
                meta=chunk_meta,
            )
            
            # Add chunk index to metadata
            chunk.meta.custom["chunk_index"] = i
            chunk.meta.custom["total_chunks"] = len(text_chunks)
            
            self.chunks.append(chunk)
            self.chunk_id_counter += 1
        
        builder_logger.debug(f"Added {len(text_chunks)} chunks from text")
    
    def add_chunks(
        self,
        chunks: Union[List[str], List[Dict[str, Any]]],
    ) -> None:
        """Add multiple chunks.
        
        Args:
            chunks: List of text strings or chunk dictionaries
        """
        for chunk_data in chunks:
            if isinstance(chunk_data, str):
                self.add_text(chunk_data)
            elif isinstance(chunk_data, dict):
                text = chunk_data.get("text", "")
                meta = {k: v for k, v in chunk_data.items() if k != "text"}
                self.add_text(text, meta)
            else:
                raise ValidationError(f"Invalid chunk data type: {type(chunk_data)}", "chunks")
    
    def add_file(
        self,
        file_path: Union[str, Path],
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add text from a file.
        
        Args:
            file_path: Path to the file
            meta: Optional metadata
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise IOError(f"File not found: {file_path}", str(file_path))
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Add file path to metadata
            if meta is None:
                meta = {}
            meta["source"] = str(file_path)
            
            self.add_text(text, meta)
            
        except Exception as e:
            raise IOError(f"Failed to read file {file_path}: {e}", str(file_path))
    
    def add_directory(
        self,
        dir_path: Union[str, Path],
        pattern: str = "*.{md,txt}",
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add all text files from a directory.
        
        Args:
            dir_path: Path to the directory
            pattern: File pattern to match
            meta: Optional metadata
        """
        dir_path = Path(dir_path)
        
        if not dir_path.exists():
            raise IOError(f"Directory not found: {dir_path}", str(dir_path))
        
        if not dir_path.is_dir():
            raise IOError(f"Path is not a directory: {dir_path}", str(dir_path))
        
        # Find matching files
        import glob
        files = glob.glob(str(dir_path / pattern))
        
        builder_logger.info(f"Found {len(files)} files in {dir_path}")
        
        for file_path in files:
            try:
                self.add_file(file_path, meta)
            except Exception as e:
                builder_logger.warning(f"Failed to process {file_path}: {e}")
    
    def generate_embeddings(
        self,
        batch_size: Optional[int] = None,
        show_progress: bool = True,
    ) -> None:
        """Generate embeddings for all chunks.
        
        Args:
            batch_size: Batch size for embedding generation
            show_progress: Whether to show progress bar
        """
        if not self.chunks:
            raise ValidationError("No chunks to embed", "chunks")
        
        if batch_size is None:
            batch_size = self.config.embedding.batch_size
        
        builder_logger.info(f"Generating embeddings for {len(self.chunks)} chunks")
        
        try:
            with time_ms() as timer:
                # Extract texts
                texts = [chunk.text for chunk in self.chunks]
                
                # Generate embeddings
                result = self.embedding_backend.encode(
                    texts=texts,
                    batch_size=batch_size,
                    show_progress=show_progress,
                )
                
                self.embeddings = result.embeddings
                
                # Add embeddings to chunks
                for i, chunk in enumerate(self.chunks):
                    chunk.embedding = self.embeddings[i]
                
                builder_logger.info(f"Embeddings generated in {timer.elapsed * 1000:.2f}ms")
                
        except Exception as e:
            raise EmbeddingError(f"Failed to generate embeddings: {e}", self.embedding_backend.model_name)
    
    def build(
        self,
        pack_path: Union[str, Path],
        ann_path: Union[str, Path],
        embed_batch_size: Optional[int] = None,
        workers: Optional[int] = None,
    ) -> BuildStats:
        """Build the complete MemPack knowledge pack.
        
        Args:
            pack_path: Path for the .mpack file
            ann_path: Path for the .ann file
            embed_batch_size: Batch size for embedding generation
            workers: Number of worker threads
            
        Returns:
            Build statistics
            
        Raises:
            ValidationError: If build parameters are invalid
            IOError: If file operations fail
        """
        if not self.chunks:
            raise ValidationError("No chunks to build", "chunks")
        
        pack_path = Path(pack_path)
        ann_path = Path(ann_path)
        
        # Update config paths
        self.config.pack_path = str(pack_path)
        self.config.ann_path = str(ann_path)
        
        if workers is not None:
            self.config.workers = workers
        
        builder_logger.info(f"Building MemPack: {pack_path} + {ann_path}")
        
        try:
            with time_ms() as build_timer:
                # Generate embeddings if not already done
                if self.embeddings is None:
                    self.generate_embeddings(
                        batch_size=embed_batch_size,
                        show_progress=self.config.progress,
                    )
                
                # Build HNSW index
                with time_ms() as index_timer:
                    self._build_index()
                    index_time_ms = index_timer.elapsed * 1000
                
                # Write pack file
                with time_ms() as pack_timer:
                    self._write_pack_file()
                    pack_time_ms = pack_timer.elapsed * 1000
                
                # Write ANN file
                with time_ms() as ann_timer:
                    self._write_ann_file()
                    ann_time_ms = ann_timer.elapsed * 1000
                
                total_time_ms = build_timer.elapsed * 1000
                
                # Calculate statistics
                stats = BuildStats(
                    chunks=len(self.chunks),
                    blocks=0,  # Will be updated by writer
                    vectors=len(self.embeddings),
                    bytes_written=pack_path.stat().st_size + ann_path.stat().st_size,
                    build_time_ms=total_time_ms,
                    embedding_time_ms=0,  # Will be updated if we tracked it
                    compression_ratio=0,  # Will be updated by writer
                )
                
                builder_logger.info(f"Build completed in {total_time_ms:.2f}ms")
                builder_logger.info(f"Index: {index_time_ms:.2f}ms, Pack: {pack_time_ms:.2f}ms, ANN: {ann_time_ms:.2f}ms")
                
                return stats
                
        except Exception as e:
            # Clean up partial files
            for path in [pack_path, ann_path]:
                if path.exists():
                    path.unlink()
            raise IOError(f"Build failed: {e}", str(pack_path))
    
    def _build_index(self) -> None:
        """Build the HNSW index."""
        builder_logger.info("Building HNSW index")
        
        # Create HNSW index
        self.hnsw_index = HNSWIndex(
            dimensions=self.embedding_backend.dimensions,
            max_elements=len(self.chunks),
            M=self.config.index.hnsw.M,
            ef_construction=self.config.index.hnsw.ef_construction,
            ef_search=self.config.index.hnsw.ef_search,
            allow_replace_deleted=self.config.index.hnsw.allow_replace_deleted,
        )
        
        # Add vectors to index
        chunk_ids = [chunk.id for chunk in self.chunks]
        self.hnsw_index.add_items(self.embeddings, chunk_ids)
        
        builder_logger.info(f"HNSW index built with {len(self.hnsw_index)} vectors")
    
    def _write_pack_file(self) -> None:
        """Write the .mpack file."""
        builder_logger.info("Writing pack file")
        
        writer = MemPackWriter(self.config, self.config.pack_path)
        writer.add_chunks(self.chunks)
        writer.write()
        
        builder_logger.info("Pack file written")
    
    def _write_ann_file(self) -> None:
        """Write the .ann file."""
        builder_logger.info("Writing ANN file")
        
        ann_file = ANNFile(self.config.ann_path)
        ann_file.write(
            index=self.hnsw_index,
            algorithm="hnsw",
            params=self.config.index.hnsw.dict(),
        )
        
        builder_logger.info("ANN file written")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get encoder statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            "chunks": len(self.chunks),
            "embeddings_generated": self.embeddings is not None,
            "embedding_dimensions": self.embedding_backend.dimensions if self.embeddings is not None else 0,
            "model_name": self.embedding_backend.model_name,
            "model_hash": self.embedding_backend.model_hash,
        }
    
    def clear(self) -> None:
        """Clear all data."""
        self.chunks.clear()
        self.embeddings = None
        self.chunk_id_counter = 0
        if hasattr(self, 'hnsw_index'):
            self.hnsw_index.clear()
