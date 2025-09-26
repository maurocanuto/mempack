"""Table of Contents (TOC) for MemPack files."""

from __future__ import annotations

import cbor2
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..errors import FileFormatError


@dataclass
class ChunkInfo:
    """Information about a text chunk."""
    
    id: int
    """Chunk ID"""
    
    block_id: int
    """Block containing this chunk"""
    
    offset: int
    """Offset within the block"""
    
    length: int
    """Length of chunk text"""
    
    meta: Dict[str, Any] = field(default_factory=dict)
    """Chunk metadata"""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "block_id": self.block_id,
            "offset": self.offset,
            "length": self.length,
            "meta": self.meta,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ChunkInfo:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            block_id=data["block_id"],
            offset=data["offset"],
            length=data["length"],
            meta=data.get("meta", {}),
        )


@dataclass
class BlockInfo:
    """Information about a compressed block."""
    
    id: int
    """Block ID"""
    
    uncompressed_size: int
    """Uncompressed size in bytes"""
    
    compressed_size: int
    """Compressed size in bytes"""
    
    first_chunk_id: int
    """First chunk ID in this block"""
    
    last_chunk_id: int
    """Last chunk ID in this block"""
    
    checksum: int
    """XXH3 checksum of uncompressed data"""
    
    offset: int
    """Offset in the pack file"""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "uncompressed_size": self.uncompressed_size,
            "compressed_size": self.compressed_size,
            "first_chunk_id": self.first_chunk_id,
            "last_chunk_id": self.last_chunk_id,
            "checksum": self.checksum,
            "offset": self.offset,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BlockInfo:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            uncompressed_size=data["uncompressed_size"],
            compressed_size=data["compressed_size"],
            first_chunk_id=data["first_chunk_id"],
            last_chunk_id=data["last_chunk_id"],
            checksum=data["checksum"],
            offset=data["offset"],
        )


@dataclass
class TagsIndex:
    """Index for metadata tags."""
    
    tags: Dict[str, List[int]] = field(default_factory=dict)
    """Mapping from tag to chunk IDs"""
    
    def add_tag(self, tag: str, chunk_id: int) -> None:
        """Add a tag for a chunk.
        
        Args:
            tag: Tag name
            chunk_id: Chunk ID
        """
        if tag not in self.tags:
            self.tags[tag] = []
        if chunk_id not in self.tags[tag]:
            self.tags[tag].append(chunk_id)
    
    def get_chunks_by_tag(self, tag: str) -> List[int]:
        """Get chunk IDs for a tag.
        
        Args:
            tag: Tag name
            
        Returns:
            List of chunk IDs
        """
        return self.tags.get(tag, [])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {"tags": self.tags}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TagsIndex:
        """Create from dictionary."""
        return cls(tags=data.get("tags", {}))


@dataclass
class TableOfContents:
    """Table of Contents for a MemPack file."""
    
    chunks: List[ChunkInfo] = field(default_factory=list)
    """List of chunk information"""
    
    blocks: List[BlockInfo] = field(default_factory=list)
    """List of block information"""
    
    tags_index: Optional[TagsIndex] = None
    """Optional tags index"""
    
    def add_chunk(self, chunk: ChunkInfo) -> None:
        """Add a chunk to the TOC.
        
        Args:
            chunk: Chunk information
        """
        self.chunks.append(chunk)
    
    def add_block(self, block: BlockInfo) -> None:
        """Add a block to the TOC.
        
        Args:
            block: Block information
        """
        self.blocks.append(block)
    
    def get_chunk(self, chunk_id: int) -> Optional[ChunkInfo]:
        """Get chunk information by ID.
        
        Args:
            chunk_id: Chunk ID
            
        Returns:
            Chunk information or None
        """
        for chunk in self.chunks:
            if chunk.id == chunk_id:
                return chunk
        return None
    
    def get_block(self, block_id: int) -> Optional[BlockInfo]:
        """Get block information by ID.
        
        Args:
            block_id: Block ID
            
        Returns:
            Block information or None
        """
        for block in self.blocks:
            if block.id == block_id:
                return block
        return None
    
    def get_chunks_in_block(self, block_id: int) -> List[ChunkInfo]:
        """Get all chunks in a block.
        
        Args:
            block_id: Block ID
            
        Returns:
            List of chunk information
        """
        return [chunk for chunk in self.chunks if chunk.block_id == block_id]
    
    def get_chunks_by_tag(self, tag: str) -> List[ChunkInfo]:
        """Get chunks by tag.
        
        Args:
            tag: Tag name
            
        Returns:
            List of chunk information
        """
        if self.tags_index is None:
            return []
        
        chunk_ids = self.tags_index.get_chunks_by_tag(tag)
        return [chunk for chunk in self.chunks if chunk.id in chunk_ids]
    
    def get_chunks_by_meta(self, meta_filter: Dict[str, Any]) -> List[ChunkInfo]:
        """Get chunks by metadata filter.
        
        Args:
            meta_filter: Metadata filter
            
        Returns:
            List of chunk information
        """
        result = []
        for chunk in self.chunks:
            if self._matches_meta_filter(chunk.meta, meta_filter):
                result.append(chunk)
        return result
    
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "blocks": [block.to_dict() for block in self.blocks],
        }
        
        if self.tags_index is not None:
            data["tags_index"] = self.tags_index.to_dict()
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TableOfContents:
        """Create from dictionary."""
        chunks = [ChunkInfo.from_dict(chunk_data) for chunk_data in data.get("chunks", [])]
        blocks = [BlockInfo.from_dict(block_data) for block_data in data.get("blocks", [])]
        
        tags_index = None
        if "tags_index" in data:
            tags_index = TagsIndex.from_dict(data["tags_index"])
        
        return cls(chunks=chunks, blocks=blocks, tags_index=tags_index)
    
    def serialize(self) -> bytes:
        """Serialize TOC to CBOR bytes.
        
        Returns:
            Serialized TOC
        """
        return cbor2.dumps(self.to_dict())
    
    @classmethod
    def deserialize(cls, data: bytes) -> TableOfContents:
        """Deserialize TOC from CBOR bytes.
        
        Args:
            data: Serialized TOC
            
        Returns:
            Deserialized TOC
            
        Raises:
            FileFormatError: If deserialization fails
        """
        try:
            # First try to decode directly
            toc_dict = cbor2.loads(data)
            return cls.from_dict(toc_dict)
        except (ValueError, cbor2.CBORDecodeError) as e:
            # If that fails, try to find the actual CBOR data by skipping leading zeros
            try:
                # Find the first non-zero byte
                start_pos = 0
                while start_pos < len(data) and data[start_pos] == 0:
                    start_pos += 1
                
                if start_pos >= len(data):
                    raise FileFormatError("TOC data is all zeros")
                
                # Try to decode from each position until we find valid CBOR
                for pos in range(start_pos, min(start_pos + 200, len(data))):
                    try:
                        remaining_data = data[pos:]
                        if len(remaining_data) == 0:
                            continue
                        toc_dict = cbor2.loads(remaining_data)
                        if isinstance(toc_dict, dict) and 'chunks' in toc_dict:
                            return cls.from_dict(toc_dict)
                    except:
                        continue
                
                raise FileFormatError("Could not find valid TOC data")
                
            except Exception as inner_e:
                raise FileFormatError(f"Failed to deserialize TOC: {e}, inner: {inner_e}")
        except Exception as e:
            raise FileFormatError(f"Failed to deserialize TOC: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get TOC statistics.
        
        Returns:
            Statistics dictionary
        """
        total_chunks = len(self.chunks)
        total_blocks = len(self.blocks)
        
        total_uncompressed = sum(block.uncompressed_size for block in self.blocks)
        total_compressed = sum(block.compressed_size for block in self.blocks)
        
        compression_ratio = total_uncompressed / total_compressed if total_compressed > 0 else 0
        
        return {
            "total_chunks": total_chunks,
            "total_blocks": total_blocks,
            "total_uncompressed_bytes": total_uncompressed,
            "total_compressed_bytes": total_compressed,
            "compression_ratio": compression_ratio,
            "has_tags_index": self.tags_index is not None,
            "total_tags": len(self.tags_index.tags) if self.tags_index else 0,
        }
