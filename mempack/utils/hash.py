"""Hash utilities for MemPack."""

from __future__ import annotations

import hashlib
import struct
from typing import Union

import xxhash

from ..errors import ValidationError


def compute_xxh3(data: Union[bytes, bytearray, memoryview]) -> int:
    """Compute XXH3 hash of data.
    
    Args:
        data: Data to hash
        
    Returns:
        64-bit hash value
    """
    return xxhash.xxh3_64(data).intdigest()


def compute_crc32(data: Union[bytes, bytearray, memoryview]) -> int:
    """Compute CRC32 hash of data.
    
    Args:
        data: Data to hash
        
    Returns:
        32-bit CRC32 value
    """
    return xxhash.xxh32(data).intdigest()


def compute_sha256(data: Union[bytes, bytearray, memoryview]) -> bytes:
    """Compute SHA256 hash of data.
    
    Args:
        data: Data to hash
        
    Returns:
        32-byte SHA256 hash
    """
    return hashlib.sha256(data).digest()


def verify_checksum(
    data: Union[bytes, bytearray, memoryview],
    expected: int,
    algorithm: str = "xxh3",
) -> bool:
    """Verify data against expected checksum.
    
    Args:
        data: Data to verify
        expected: Expected checksum value
        algorithm: Hash algorithm ('xxh3', 'crc32', 'sha256')
        
    Returns:
        True if checksum matches
        
    Raises:
        ValidationError: If algorithm is unsupported
    """
    if algorithm == "xxh3":
        actual = compute_xxh3(data)
    elif algorithm == "crc32":
        actual = compute_crc32(data)
    elif algorithm == "sha256":
        actual = int.from_bytes(compute_sha256(data)[:8], byteorder='big')
    else:
        raise ValidationError(f"Unsupported hash algorithm: {algorithm}", "algorithm")
    
    return actual == expected


def pack_checksum(checksum: int, width: int = 8) -> bytes:
    """Pack a checksum into bytes.
    
    Args:
        checksum: Checksum value
        width: Number of bytes (4 or 8)
        
    Returns:
        Packed checksum
        
    Raises:
        ValidationError: If width is invalid
    """
    if width == 4:
        return struct.pack('<I', checksum & 0xFFFFFFFF)
    elif width == 8:
        return struct.pack('<Q', checksum & 0xFFFFFFFFFFFFFFFF)
    else:
        raise ValidationError(f"Invalid checksum width: {width}", "width")


def unpack_checksum(data: bytes, width: int = 8) -> int:
    """Unpack a checksum from bytes.
    
    Args:
        data: Packed checksum data
        width: Number of bytes (4 or 8)
        
    Returns:
        Unpacked checksum
        
    Raises:
        ValidationError: If width is invalid or data is too short
    """
    if len(data) < width:
        raise ValidationError(f"Data too short for {width}-byte checksum", "data")
    
    if width == 4:
        return struct.unpack('<I', data[:4])[0]
    elif width == 8:
        return struct.unpack('<Q', data[:8])[0]
    else:
        raise ValidationError(f"Invalid checksum width: {width}", "width")


def compute_block_checksum(
    data: Union[bytes, bytearray, memoryview],
    algorithm: str = "xxh3",
) -> int:
    """Compute checksum for a data block.
    
    Args:
        data: Block data
        algorithm: Hash algorithm
        
    Returns:
        Checksum value
    """
    if algorithm == "xxh3":
        return compute_xxh3(data)
    elif algorithm == "crc32":
        return compute_crc32(data)
    elif algorithm == "sha256":
        return int.from_bytes(compute_sha256(data)[:8], byteorder='big')
    else:
        raise ValidationError(f"Unsupported hash algorithm: {algorithm}", "algorithm")


def verify_block_checksum(
    data: Union[bytes, bytearray, memoryview],
    expected: int,
    algorithm: str = "xxh3",
) -> bool:
    """Verify a block's checksum.
    
    Args:
        data: Block data
        expected: Expected checksum
        algorithm: Hash algorithm
        
    Returns:
        True if checksum matches
    """
    actual = compute_block_checksum(data, algorithm)
    return actual == expected


def compute_file_checksum(
    file_path: str,
    algorithm: str = "xxh3",
    chunk_size: int = 65536,
) -> int:
    """Compute checksum of a file.
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm
        chunk_size: Read chunk size
        
    Returns:
        File checksum
        
    Raises:
        IOError: If file cannot be read
    """
    if algorithm == "xxh3":
        hasher = xxhash.xxh3_64()
    elif algorithm == "crc32":
        hasher = xxhash.xxh32()
    elif algorithm == "sha256":
        hasher = hashlib.sha256()
    else:
        raise ValidationError(f"Unsupported hash algorithm: {algorithm}", "algorithm")
    
    try:
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                hasher.update(chunk)
        
        if algorithm == "sha256":
            return int.from_bytes(hasher.digest()[:8], byteorder='big')
        else:
            return hasher.intdigest()
            
    except Exception as e:
        from ..errors import IOError
        raise IOError(f"Failed to compute checksum of {file_path}: {e}", file_path)
