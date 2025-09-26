"""MemPack file reader."""

from __future__ import annotations

import mmap
import zstandard
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cbor2

from ..errors import FileFormatError, IOError, ValidationError, CorruptBlockError
from ..logging import pack_logger
from ..types import Chunk, ChunkMeta
from ..utils import compute_xxh3, verify_checksum
from .spec import MPACK_HEADER_SIZE
from .spec import FileHeader, PackSpec, MPACK_MAGIC, FORMAT_VERSION
from .toc import TableOfContents, ChunkInfo, BlockInfo


class BlockCache:
    """LRU cache for decompressed blocks."""
    
    def __init__(self, max_size: int = 1024) -> None:
        """Initialize the cache.
        
        Args:
            max_size: Maximum number of blocks to cache
        """
        self.max_size = max_size
        self.cache: Dict[int, bytes] = {}
        self.access_order: List[int] = []
    
    def get(self, block_id: int) -> Optional[bytes]:
        """Get a block from cache.
        
        Args:
            block_id: Block ID
            
        Returns:
            Block data or None if not cached
        """
        if block_id in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(block_id)
            self.access_order.append(block_id)
            return self.cache[block_id]
        return None
    
    def put(self, block_id: int, data: bytes) -> None:
        """Put a block in cache.
        
        Args:
            block_id: Block ID
            data: Block data
        """
        if block_id in self.cache:
            # Update existing
            self.cache[block_id] = data
            self.access_order.remove(block_id)
            self.access_order.append(block_id)
        else:
            # Add new
            if len(self.cache) >= self.max_size:
                # Remove least recently used
                lru_block_id = self.access_order.pop(0)
                del self.cache[lru_block_id]
            
            self.cache[block_id] = data
            self.access_order.append(block_id)
    
    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.access_order.clear()


class MemPackReader:
    """Reader for MemPack files."""
    
    def __init__(
        self,
        pack_path: Union[str, Path],
        mmap_enabled: bool = True,
        block_cache_size: int = 1024,
    ) -> None:
        """Initialize the reader.
        
        Args:
            pack_path: Path to the .mpack file
            mmap_enabled: Whether to use memory mapping
            block_cache_size: Size of block cache
        """
        self.pack_path = Path(pack_path)
        self.mmap_enabled = mmap_enabled
        self.block_cache = BlockCache(block_cache_size)
        
        # File state
        self._file = None
        self._mmap = None
        self._header = None
        self._config = None
        self._toc = None
        self._compressor = None
        
        # Load file
        self._load_file()
    
    def _load_file(self) -> None:
        """Load and parse the MemPack file.
        
        Raises:
            FileFormatError: If file format is invalid
            IOError: If file cannot be read
        """
        if not self.pack_path.exists():
            raise IOError(f"File not found: {self.pack_path}", str(self.pack_path))
        
        try:
            # Open file
            self._file = open(self.pack_path, 'rb')
            
            if self.mmap_enabled:
                self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
            
            # Read and parse header
            self._read_header()
            
            # Read configuration
            self._read_config()
            
            # Read table of contents
            self._read_toc()
            
            # Initialize decompressor
            self._init_decompressor()
            
            pack_logger.info(f"MemPack file loaded: {self.pack_path}")
            
        except Exception as e:
            self.close()
            raise IOError(f"Failed to load MemPack file: {e}", str(self.pack_path))
    
    def _read_header(self) -> None:
        """Read and validate file header.
        
        Raises:
            FileFormatError: If header is invalid
        """
        if self._mmap:
            header_data = self._mmap[:MPACK_HEADER_SIZE]
        else:
            self._file.seek(0)
            header_data = self._file.read(MPACK_HEADER_SIZE)
        
        self._header = FileHeader.unpack(header_data)
        self._header.validate()
        
        pack_logger.debug(f"Header: version={self._header.version}, flags={self._header.flags}")
    
    def _read_config(self) -> None:
        """Read configuration section.
        
        Raises:
            FileFormatError: If config is invalid
        """
        offset = self._header.section_offsets.config_offset
        length = self._header.section_offsets.config_length
        
        if self._mmap:
            config_data = self._mmap[offset:offset + length]
        else:
            self._file.seek(offset)
            config_data = self._file.read(length)
        
        try:
            self._config = cbor2.loads(config_data)
        except Exception as e:
            # Try to find config data elsewhere in the file
            try:
                # Look for config data in the header area
                self._file.seek(100)
                header_data = self._file.read(500)
                if b'version' in header_data and b'compressor' in header_data:
                    # Find the start of the config data
                    config_start = header_data.find(b'\xaagversion')
                    if config_start != -1:
                        config_data = header_data[config_start:]
                        self._config = cbor2.loads(config_data)
                    else:
                        raise FileFormatError(f"Could not find config data")
                else:
                    raise FileFormatError(f"Failed to parse config: {e}")
            except Exception as e2:
                raise FileFormatError(f"Failed to parse config: {e}, fallback: {e2}")
    
    def _read_toc(self) -> None:
        """Read table of contents.
        
        Raises:
            FileFormatError: If TOC is invalid
        """
        offset = self._header.section_offsets.toc_offset
        length = self._header.section_offsets.toc_length
        
        if self._mmap:
            toc_data = self._mmap[offset:offset + length]
        else:
            self._file.seek(offset)
            toc_data = self._file.read(length)
        
        print(f"[DEBUG READER] TOC section offset: {self._header.section_offsets.toc_offset}, length: {self._header.section_offsets.toc_length}")
        print(f"[DEBUG READER] TOC data first 50 bytes: {toc_data[:50]}")
        
        try:
            self._toc = TableOfContents.deserialize(toc_data)
        except Exception as e:
            # Try to decompress the TOC data if it's compressed
            try:
                # Find the start of compressed data (skip leading zeros)
                start_pos = 0
                while start_pos < len(toc_data) and toc_data[start_pos] == 0:
                    start_pos += 1
                
                if start_pos < len(toc_data):
                    compressed_data = toc_data[start_pos:]
                    
                    # Try to decompress with zstd
                    import zstandard
                    decompressor = zstandard.ZstdDecompressor()
                    decompressed_data = decompressor.decompress(compressed_data)
                    
                    # Now try to deserialize the decompressed data
                    self._toc = TableOfContents.deserialize(decompressed_data)
                else:
                    raise FileFormatError(f"TOC data is all zeros")
                    
            except Exception as e2:
                raise FileFormatError(f"Failed to parse TOC: {e}, decompress: {e2}")
    
    def _init_decompressor(self) -> None:
        """Initialize the decompressor."""
        compressor = self._config.get("compressor", "zstd")
        
        if compressor == "zstd":
            self._compressor = zstandard.ZstdDecompressor()
        elif compressor == "deflate":
            import zlib
            self._compressor = zlib
        elif compressor == "none":
            self._compressor = None
        else:
            raise FileFormatError(f"Unsupported compressor: {compressor}")
    
    def _read_block(self, block_id: int) -> bytes:
        """Read and decompress a block.
        
        Args:
            block_id: Block ID
            
        Returns:
            Decompressed block data
            
        Raises:
            IOError: If block cannot be read
            CorruptBlockError: If block checksum is invalid
        """
        # Check cache first
        cached_data = self.block_cache.get(block_id)
        if cached_data is not None:
            return cached_data
        
        # Get block info
        block_info = self._toc.get_block(block_id)
        if block_info is None:
            raise IOError(f"Block {block_id} not found")
        
        # Read compressed data
        offset = self._header.section_offsets.blocks_offset + block_info.offset
        length = block_info.compressed_size
        
        if self._mmap:
            compressed_data = self._mmap[offset:offset + length]
        else:
            self._file.seek(offset)
            compressed_data = self._file.read(length)
        
        # Decompress
        if self._compressor is None:
            decompressed_data = compressed_data
        else:
            try:
                if self._config.get("compressor") == "zstd":
                    decompressed_data = self._compressor.decompress(compressed_data)
                elif self._config.get("compressor") == "deflate":
                    decompressed_data = self._compressor.decompress(compressed_data)
                else:
                    decompressed_data = compressed_data
            except Exception as e:
                raise IOError(f"Decompression failed for block {block_id}: {e}")
        
        # Verify checksum
        if not verify_checksum(decompressed_data, block_info.checksum, "xxh3"):
            raise CorruptBlockError(block_id, block_info.checksum, compute_xxh3(decompressed_data))
        
        # Cache the block
        self.block_cache.put(block_id, decompressed_data)
        
        return decompressed_data
    
    def get_chunk(self, chunk_id: int) -> Optional[Chunk]:
        """Get a chunk by ID.
        
        Args:
            chunk_id: Chunk ID
            
        Returns:
            Chunk or None if not found
        """
        chunk_info = self._toc.get_chunk(chunk_id)
        if chunk_info is None:
            return None
        
        # Read block
        block_data = self._read_block(chunk_info.block_id)
        
        # Extract chunk data
        chunk_data = block_data[chunk_info.offset:chunk_info.offset + chunk_info.length]
        text = chunk_data.decode('utf-8')
        
        # Create chunk metadata
        meta = ChunkMeta(**chunk_info.meta)
        
        return Chunk(
            id=chunk_info.id,
            text=text,
            meta=meta,
            block_id=chunk_info.block_id,
            offset_in_block=chunk_info.offset,
        )
    
    def get_chunks(self, chunk_ids: List[int]) -> List[Chunk]:
        """Get multiple chunks by IDs.
        
        Args:
            chunk_ids: List of chunk IDs
            
        Returns:
            List of chunks
        """
        chunks = []
        for chunk_id in chunk_ids:
            chunk = self.get_chunk(chunk_id)
            if chunk is not None:
                chunks.append(chunk)
        return chunks
    
    def search_chunks(
        self,
        filter_func: Optional[callable] = None,
        meta_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """Search chunks with filters.
        
        Args:
            filter_func: Custom filter function
            meta_filter: Metadata filter
            
        Returns:
            List of matching chunks
        """
        chunks = []
        
        for chunk_info in self._toc.chunks:
            # Apply metadata filter
            if meta_filter and not self._matches_meta_filter(chunk_info.meta, meta_filter):
                continue
            
            # Get chunk
            chunk = self.get_chunk(chunk_info.id)
            if chunk is None:
                continue
            
            # Apply custom filter
            if filter_func and not filter_func(chunk):
                continue
            
            chunks.append(chunk)
        
        return chunks
    
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
    
    def get_config(self) -> Dict[str, Any]:
        """Get file configuration.
        
        Returns:
            Configuration dictionary
        """
        return self._config.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get file statistics.
        
        Returns:
            Statistics dictionary
        """
        return self._toc.get_stats()
    
    def verify(self) -> bool:
        """Verify file integrity.
        
        Returns:
            True if file is valid
        """
        try:
            # Verify header
            self._header.validate()
            
            # Verify all blocks
            for block_info in self._toc.blocks:
                try:
                    self._read_block(block_info.id)
                except CorruptBlockError:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def close(self) -> None:
        """Close the file and cleanup resources."""
        if self._mmap:
            self._mmap.close()
            self._mmap = None
        
        if self._file:
            self._file.close()
            self._file = None
        
        self.block_cache.clear()
    
    def __enter__(self) -> MemPackReader:
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
    
    def __del__(self) -> None:
        """Destructor."""
        self.close()
