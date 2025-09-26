"""MemPack file format specification."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Dict, Optional

from ..errors import FileFormatError


# File format constants
MPACK_MAGIC = b"MPACK\0"
ANN_MAGIC = b"MPANN\0"
FORMAT_VERSION = 1

# Header sizes
MPACK_HEADER_SIZE = 128
ANN_HEADER_SIZE = 64

# Section alignment
SECTION_ALIGNMENT = 4096

# Compression types
COMPRESSOR_ZSTD = 0x01
COMPRESSOR_DEFLATE = 0x02
COMPRESSOR_NONE = 0x00

# Flags
FLAG_ECC_ENABLED = 0x01
FLAG_LITTLE_ENDIAN = 0x02

# ECC parameters
ECC_DEFAULT_K = 10
ECC_DEFAULT_M = 2
ECC_DEFAULT_BLOCK_SIZE = 4096


@dataclass
class SectionOffsets:
    """Offsets and lengths of file sections."""
    
    config_offset: int = 0
    config_length: int = 0
    
    toc_offset: int = 0
    toc_length: int = 0
    
    blocks_offset: int = 0
    blocks_length: int = 0
    
    checksums_offset: int = 0
    checksums_length: int = 0
    
    tags_index_offset: int = 0
    tags_index_length: int = 0
    
    ecc_offset: int = 0
    ecc_length: int = 0


@dataclass
class FileHeader:
    """MemPack file header."""
    
    magic: bytes = MPACK_MAGIC
    version: int = FORMAT_VERSION
    flags: int = 0
    created_at: int = 0
    section_offsets: SectionOffsets = None
    
    def __post_init__(self) -> None:
        if self.section_offsets is None:
            self.section_offsets = SectionOffsets()
    
    def pack(self) -> bytes:
        """Pack header into bytes.
        
        Returns:
            Packed header bytes
        """
        if len(self.magic) != 6:
            raise FileFormatError("Invalid magic bytes length")
        
        # Pack header
        header = struct.pack(
            '<6sHHIQQQQQQQQQQQQQQQQQQQQ',
            self.magic,
            self.version,
            self.flags,
            self.created_at,
            self.section_offsets.config_offset,
            self.section_offsets.config_length,
            self.section_offsets.toc_offset,
            self.section_offsets.toc_length,
            self.section_offsets.blocks_offset,
            self.section_offsets.blocks_length,
            self.section_offsets.checksums_offset,
            self.section_offsets.checksums_length,
            self.section_offsets.tags_index_offset,
            self.section_offsets.tags_index_length,
            self.section_offsets.ecc_offset,
            self.section_offsets.ecc_length,
            0, 0, 0, 0, 0, 0  # Reserved padding
        )
        
        # Pad to header size
        if len(header) < MPACK_HEADER_SIZE:
            header += b'\x00' * (MPACK_HEADER_SIZE - len(header))
        
        return header
    
    @classmethod
    def unpack(cls, data: bytes) -> FileHeader:
        """Unpack header from bytes.
        
        Args:
            data: Header bytes
            
        Returns:
            Unpacked header
            
        Raises:
            FileFormatError: If header format is invalid
        """
        if len(data) < MPACK_HEADER_SIZE:
            raise FileFormatError("Header too short")
        
        # Unpack header
        fields = struct.unpack('<6sHHIQQQQQQQQQQQQQQQQQQQQ', data[:MPACK_HEADER_SIZE])
        
        magic = fields[0]
        version = fields[1]
        flags = fields[2]
        created_at = fields[3]
        
        # Create section offsets
        section_offsets = SectionOffsets(
            config_offset=fields[4],
            config_length=fields[5],
            toc_offset=fields[6],
            toc_length=fields[7],
            blocks_offset=fields[8],
            blocks_length=fields[9],
            checksums_offset=fields[10],
            checksums_length=fields[11],
            tags_index_offset=fields[12],
            tags_index_length=fields[13],
            ecc_offset=fields[14],
            ecc_length=fields[15],
        )
        
        return cls(
            magic=magic,
            version=version,
            flags=flags,
            created_at=created_at,
            section_offsets=section_offsets,
        )
    
    def validate(self) -> None:
        """Validate header.
        
        Raises:
            FileFormatError: If header is invalid
        """
        if self.magic != MPACK_MAGIC:
            raise FileFormatError(f"Invalid magic bytes: {self.magic}")
        
        if self.version != FORMAT_VERSION:
            raise FileFormatError(f"Unsupported version: {self.version}")
        
        # Validate section offsets are aligned
        for name, offset in [
            ("config", self.section_offsets.config_offset),
            ("toc", self.section_offsets.toc_offset),
            ("blocks", self.section_offsets.blocks_offset),
            ("checksums", self.section_offsets.checksums_offset),
            ("tags_index", self.section_offsets.tags_index_offset),
            ("ecc", self.section_offsets.ecc_offset),
        ]:
            if offset > 0 and offset % SECTION_ALIGNMENT != 0:
                raise FileFormatError(f"Section {name} not aligned: {offset}")


@dataclass
class ANNHeader:
    """ANN index file header."""
    
    magic: bytes = ANN_MAGIC
    version: int = FORMAT_VERSION
    algorithm: int = 0  # 0=HNSW, 1=IVFPQ
    dimensions: int = 0
    vector_count: int = 0
    id_width: int = 4
    params: bytes = b'\x00' * 32  # Algorithm-specific parameters
    
    def pack(self) -> bytes:
        """Pack header into bytes.
        
        Returns:
            Packed header bytes
        """
        if len(self.magic) != 6:
            raise FileFormatError("Invalid ANN magic bytes length")
        
        if len(self.params) != 32:
            raise FileFormatError("Invalid params length")
        
        header = struct.pack(
            '<6sHHIQQI32s',
            self.magic,
            self.version,
            self.algorithm,
            self.dimensions,
            self.vector_count,
            self.id_width,
            self.params,
        )
        
        # Pad to header size
        if len(header) < ANN_HEADER_SIZE:
            header += b'\x00' * (ANN_HEADER_SIZE - len(header))
        
        return header
    
    @classmethod
    def unpack(cls, data: bytes) -> ANNHeader:
        """Unpack header from bytes.
        
        Args:
            data: Header bytes
            
        Returns:
            Unpacked header
            
        Raises:
            FileFormatError: If header format is invalid
        """
        if len(data) < ANN_HEADER_SIZE:
            raise FileFormatError("ANN header too short")
        
        fields = struct.unpack('<6sHHIQQI32s', data[:ANN_HEADER_SIZE])
        
        return cls(
            magic=fields[0],
            version=fields[1],
            algorithm=fields[2],
            dimensions=fields[3],
            vector_count=fields[4],
            id_width=fields[5],
            params=fields[6],
        )
    
    def validate(self) -> None:
        """Validate header.
        
        Raises:
            FileFormatError: If header is invalid
        """
        if self.magic != ANN_MAGIC:
            raise FileFormatError(f"Invalid ANN magic bytes: {self.magic}")
        
        if self.version != FORMAT_VERSION:
            raise FileFormatError(f"Unsupported ANN version: {self.version}")
        
        if self.algorithm not in [0, 1]:
            raise FileFormatError(f"Unsupported algorithm: {self.algorithm}")
        
        if self.dimensions <= 0:
            raise FileFormatError(f"Invalid dimensions: {self.dimensions}")
        
        if self.id_width not in [2, 4, 8]:
            raise FileFormatError(f"Invalid ID width: {self.id_width}")


class PackSpec:
    """MemPack file format specification."""
    
    @staticmethod
    def get_compressor_flag(compressor: str) -> int:
        """Get compressor flag value.
        
        Args:
            compressor: Compressor name
            
        Returns:
            Flag value
            
        Raises:
            FileFormatError: If compressor is unsupported
        """
        compressor_map = {
            "zstd": COMPRESSOR_ZSTD,
            "deflate": COMPRESSOR_DEFLATE,
            "none": COMPRESSOR_NONE,
        }
        
        if compressor not in compressor_map:
            raise FileFormatError(f"Unsupported compressor: {compressor}")
        
        return compressor_map[compressor]
    
    @staticmethod
    def get_compressor_name(flag: int) -> str:
        """Get compressor name from flag.
        
        Args:
            flag: Compressor flag
            
        Returns:
            Compressor name
            
        Raises:
            FileFormatError: If flag is invalid
        """
        flag_map = {
            COMPRESSOR_ZSTD: "zstd",
            COMPRESSOR_DEFLATE: "deflate",
            COMPRESSOR_NONE: "none",
        }
        
        if flag not in flag_map:
            raise FileFormatError(f"Invalid compressor flag: {flag}")
        
        return flag_map[flag]
    
    @staticmethod
    def is_ecc_enabled(flags: int) -> bool:
        """Check if ECC is enabled.
        
        Args:
            flags: Header flags
            
        Returns:
            True if ECC is enabled
        """
        return bool(flags & FLAG_ECC_ENABLED)
    
    @staticmethod
    def is_little_endian(flags: int) -> bool:
        """Check if file is little-endian.
        
        Args:
            flags: Header flags
            
        Returns:
            True if little-endian
        """
        return bool(flags & FLAG_LITTLE_ENDIAN)
    
    @staticmethod
    def align_offset(offset: int) -> int:
        """Align offset to section boundary.
        
        Args:
            offset: Original offset
            
        Returns:
            Aligned offset
        """
        return ((offset + SECTION_ALIGNMENT - 1) // SECTION_ALIGNMENT) * SECTION_ALIGNMENT
    
    @staticmethod
    def get_algorithm_name(algorithm: int) -> str:
        """Get algorithm name from code.
        
        Args:
            algorithm: Algorithm code
            
        Returns:
            Algorithm name
            
        Raises:
            FileFormatError: If algorithm is unsupported
        """
        algorithm_map = {
            0: "hnsw",
            1: "ivfpq",
        }
        
        if algorithm not in algorithm_map:
            raise FileFormatError(f"Unsupported algorithm: {algorithm}")
        
        return algorithm_map[algorithm]
    
    @staticmethod
    def get_algorithm_code(algorithm: str) -> int:
        """Get algorithm code from name.
        
        Args:
            algorithm: Algorithm name
            
        Returns:
            Algorithm code
            
        Raises:
            FileFormatError: If algorithm is unsupported
        """
        algorithm_map = {
            "hnsw": 0,
            "ivfpq": 1,
        }
        
        if algorithm not in algorithm_map:
            raise FileFormatError(f"Unsupported algorithm: {algorithm}")
        
        return algorithm_map[algorithm]
