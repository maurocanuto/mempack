"""MemPack file writer."""

from __future__ import annotations

import os
import time
import zstandard
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cbor2

from ..config import MemPackConfig
from ..errors import CompressionError, IOError, ValidationError
from ..logging import pack_logger
from ..types import Chunk, ChunkMeta, PackConfig
from ..utils import atomic_write, compute_xxh3, align_offset
from .spec import PackSpec, FileHeader, SectionOffsets, MPACK_HEADER_SIZE, FORMAT_VERSION, MPACK_MAGIC
from .toc import TableOfContents, ChunkInfo, BlockInfo, TagsIndex


class MemPackWriter:
    """Writer for MemPack files."""
    
    def __init__(
        self,
        config: MemPackConfig,
        pack_path: Union[str, Path],
    ) -> None:
        """Initialize the writer.
        
        Args:
            config: MemPack configuration
            pack_path: Path to the .mpack file
        """
        self.config = config
        self.pack_path = Path(pack_path)
        self.toc = TableOfContents()
        self.chunks: List[Chunk] = []
        self.blocks: List[bytes] = []
        self.checksums: List[int] = []
        self.tags_index = TagsIndex()
        
        # Initialize compressor
        self._init_compressor()
        
        # Ensure output directory exists
        self.pack_path.parent.mkdir(parents=True, exist_ok=True)
    
    def _init_compressor(self) -> None:
        """Initialize the compressor."""
        compressor = self.config.compression.algorithm
        
        if compressor == "zstd":
            self.compressor = zstandard.ZstdCompressor(
                level=self.config.compression.level,
                threads=self.config.compression.threads,
            )
        elif compressor == "deflate":
            import zlib
            self.compressor = zlib
        elif compressor == "none":
            self.compressor = None
        else:
            raise ValidationError(f"Unsupported compressor: {compressor}", "compressor")
    
    def add_chunk(self, chunk: Chunk) -> None:
        """Add a chunk to be written.
        
        Args:
            chunk: Chunk to add
        """
        self.chunks.append(chunk)
        
        # Add to tags index if metadata has tags
        if chunk.meta.tags:
            for tag in chunk.meta.tags:
                self.tags_index.add_tag(tag, chunk.id)
    
    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Add multiple chunks.
        
        Args:
            chunks: Chunks to add
        """
        for chunk in chunks:
            self.add_chunk(chunk)
    
    def _compress_data(self, data: bytes) -> bytes:
        """Compress data using the configured compressor.
        
        Args:
            data: Data to compress
            
        Returns:
            Compressed data
            
        Raises:
            CompressionError: If compression fails
        """
        if self.compressor is None:
            return data
        
        try:
            if self.config.compression.algorithm == "zstd":
                return self.compressor.compress(data)
            elif self.config.compression.algorithm == "deflate":
                return self.compressor.compress(data)
            else:
                return data
        except Exception as e:
            raise CompressionError(f"Compression failed: {e}", self.config.compression.algorithm)
    
    def _create_blocks(self) -> None:
        """Create compressed blocks from chunks."""
        pack_logger.info("Creating compressed blocks")
        
        block_size = 64 * 1024  # 64KB blocks
        current_block = b""
        current_chunks = []
        block_id = 0
        
        for chunk in self.chunks:
            # Serialize chunk data
            chunk_data = chunk.text.encode('utf-8')
            chunk_header = f"{chunk.id}:{len(chunk_data)}:".encode('utf-8')
            chunk_payload = chunk_header + chunk_data
            
            # Check if chunk fits in current block
            if len(current_block) + len(chunk_payload) > block_size and current_block:
                # Finalize current block
                self._finalize_block(current_block, current_chunks, block_id)
                block_id += 1
                current_block = b""
                current_chunks = []
            
            # Add chunk to current block
            current_block += chunk_payload
            current_chunks.append(chunk)
        
        # Finalize last block
        if current_block:
            self._finalize_block(current_block, current_chunks, block_id)
    
    def _finalize_block(
        self,
        block_data: bytes,
        chunks: List[Chunk],
        block_id: int,
    ) -> None:
        """Finalize a block and add it to the pack.
        
        Args:
            block_data: Uncompressed block data
            chunks: Chunks in this block
            block_id: Block ID
        """
        # Compress block
        compressed_data = self._compress_data(block_data)
        
        # Compute checksum
        checksum = compute_xxh3(block_data)
        
        # Store block
        self.blocks.append(compressed_data)
        self.checksums.append(checksum)
        
        # Create block info
        block_info = BlockInfo(
            id=block_id,
            uncompressed_size=len(block_data),
            compressed_size=len(compressed_data),
            first_chunk_id=chunks[0].id,
            last_chunk_id=chunks[-1].id,
            checksum=checksum,
            offset=0,  # Will be set later
        )
        
        # Add to TOC
        self.toc.add_block(block_info)
        
        # Create chunk infos
        offset = 0
        for chunk in chunks:
            # Find chunk data in block
            chunk_header = f"{chunk.id}:".encode('utf-8')
            chunk_start = block_data.find(chunk_header, offset)
            if chunk_start == -1:
                raise ValidationError(f"Chunk {chunk.id} not found in block", "chunk")
            
            # Find chunk length
            length_start = chunk_start + len(chunk_header)
            length_end = block_data.find(b':', length_start)
            if length_end == -1:
                raise ValidationError(f"Invalid chunk format for {chunk.id}", "chunk")
            
            chunk_length = int(block_data[length_start:length_end].decode('utf-8'))
            chunk_offset = length_end + 1
            
            # Create chunk info
            chunk_info = ChunkInfo(
                id=chunk.id,
                block_id=block_id,
                offset=chunk_offset,
                length=chunk_length,
                meta=chunk.meta.__dict__,
            )
            
            self.toc.add_chunk(chunk_info)
            offset = chunk_offset + chunk_length
    
    def _write_config(self) -> bytes:
        """Write configuration section.
        
        Returns:
            Configuration bytes
        """
        config_data = {
            "version": self.config.version,
            "compressor": self.config.compression.algorithm,
            "chunk_size": self.config.chunking.chunk_size,
            "chunk_overlap": self.config.chunking.chunk_overlap,
            "embedding_model": self.config.embedding.model,
            "embedding_dim": self.config.embedding.dimensions,
            "index_type": self.config.index.type,
            "index_params": self.config.index.hnsw.dict() if self.config.index.hnsw else {},
            "ecc_enabled": self.config.ecc.enabled,
            "ecc_params": self.config.ecc.dict() if self.config.ecc.enabled else None,
        }
        
        return cbor2.dumps(config_data)
    
    def _write_toc(self) -> bytes:
        """Write table of contents.
        
        Returns:
            TOC bytes
        """
        return self.toc.serialize()
    
    def _write_blocks(self) -> bytes:
        """Write compressed blocks.
        
        Returns:
            Blocks bytes
        """
        return b''.join(self.blocks)
    
    def _write_checksums(self) -> bytes:
        """Write checksums.
        
        Returns:
            Checksums bytes
        """
        return b''.join(checksum.to_bytes(8, 'little') for checksum in self.checksums)
    
    def _write_tags_index(self) -> bytes:
        """Write tags index.
        
        Returns:
            Tags index bytes
        """
        if not self.tags_index.tags:
            return b''
        
        return cbor2.dumps(self.tags_index.to_dict())
    
    def _write_ecc(self) -> bytes:
        """Write ECC data.
        
        Returns:
            ECC bytes
        """
        if not self.config.ecc.enabled:
            return b''
        
        # TODO: Implement ECC encoding
        return b''
    
    def write(self) -> None:
        """Write the complete MemPack file.
        
        Raises:
            IOError: If writing fails
        """
        pack_logger.info(f"Writing MemPack file: {self.pack_path}")
        
        try:
            # Create blocks from chunks
            self._create_blocks()
            
            # Write sections
            config_data = self._write_config()
            toc_data = self._write_toc()
            blocks_data = self._write_blocks()
            checksums_data = self._write_checksums()
            tags_index_data = self._write_tags_index()
            ecc_data = self._write_ecc()
            
            # Calculate section offsets
            offset = PackSpec.align_offset(MPACK_HEADER_SIZE)
            
            section_offsets = SectionOffsets(
                config_offset=offset,
                config_length=len(config_data),
            )
            offset = PackSpec.align_offset(offset + len(config_data))
            
            section_offsets.toc_offset = offset
            section_offsets.toc_length = len(toc_data)
            offset = PackSpec.align_offset(offset + len(toc_data))
            
            section_offsets.blocks_offset = offset
            section_offsets.blocks_length = len(blocks_data)
            offset = PackSpec.align_offset(offset + len(blocks_data))
            
            section_offsets.checksums_offset = offset
            section_offsets.checksums_length = len(checksums_data)
            offset = PackSpec.align_offset(offset + len(checksums_data))
            
            section_offsets.tags_index_offset = offset
            section_offsets.tags_index_length = len(tags_index_data)
            offset = PackSpec.align_offset(offset + len(tags_index_data))
            
            section_offsets.ecc_offset = offset
            section_offsets.ecc_length = len(ecc_data)
            
            # Create header
            flags = PackSpec.get_compressor_flag(self.config.compression.algorithm)
            if self.config.ecc.enabled:
                flags |= 0x01  # ECC enabled flag
            
            header = FileHeader(
                magic=MPACK_MAGIC,
                version=FORMAT_VERSION,
                flags=flags,
                created_at=int(time.time()),
                section_offsets=section_offsets,
            )
            
            # Write file
            file_data = (
                header.pack() +
                config_data +
                b'\x00' * (section_offsets.toc_offset - section_offsets.config_offset - len(config_data)) +
                toc_data +
                b'\x00' * (section_offsets.blocks_offset - section_offsets.toc_offset - len(toc_data)) +
                blocks_data +
                b'\x00' * (section_offsets.checksums_offset - section_offsets.blocks_offset - len(blocks_data)) +
                checksums_data +
                b'\x00' * (section_offsets.tags_index_offset - section_offsets.checksums_offset - len(checksums_data)) +
                tags_index_data +
                b'\x00' * (section_offsets.ecc_offset - section_offsets.tags_index_offset - len(tags_index_data)) +
                ecc_data
            )
            
            # Atomic write
            atomic_write(self.pack_path, file_data)
            
            pack_logger.info(f"MemPack file written: {self.pack_path}")
            pack_logger.info(f"Total size: {len(file_data)} bytes")
            pack_logger.info(f"Chunks: {len(self.chunks)}")
            pack_logger.info(f"Blocks: {len(self.blocks)}")
            
        except Exception as e:
            raise IOError(f"Failed to write MemPack file: {e}", str(self.pack_path))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get writing statistics.
        
        Returns:
            Statistics dictionary
        """
        total_uncompressed = sum(block.uncompressed_size for block in self.toc.blocks)
        total_compressed = sum(block.compressed_size for block in self.toc.blocks)
        
        return {
            "chunks": len(self.chunks),
            "blocks": len(self.blocks),
            "total_uncompressed_bytes": total_uncompressed,
            "total_compressed_bytes": total_compressed,
            "compression_ratio": total_uncompressed / total_compressed if total_compressed > 0 else 0,
            "file_size": self.pack_path.stat().st_size if self.pack_path.exists() else 0,
        }
