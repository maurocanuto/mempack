"""Error correction utilities for MemPack."""

from __future__ import annotations

from typing import Optional, Tuple

from ..errors import ECCError


def encode_reed_solomon(
    data: bytes,
    k: int = 10,
    m: int = 2,
) -> Tuple[bytes, Optional[bytes]]:
    """Encode data with Reed-Solomon error correction.
    
    Args:
        data: Data to encode
        k: Number of data blocks
        m: Number of parity blocks
        
    Returns:
        Tuple of (encoded_data, parity_data)
        
    Raises:
        ECCError: If encoding fails or reedsolo is not available
    """
    try:
        import reedsolo
    except ImportError:
        raise ECCError("reedsolo package not available for error correction")
    
    try:
        # Create Reed-Solomon codec
        rs = reedsolo.RSCodec(m)
        
        # Encode data
        encoded = rs.encode(data)
        
        # Split into data and parity
        data_part = encoded[:len(data)]
        parity_part = encoded[len(data):]
        
        return data_part, parity_part
        
    except Exception as e:
        raise ECCError(f"Reed-Solomon encoding failed: {e}")


def decode_reed_solomon(
    data: bytes,
    parity: bytes,
    k: int = 10,
    m: int = 2,
) -> bytes:
    """Decode data with Reed-Solomon error correction.
    
    Args:
        data: Data part
        parity: Parity part
        k: Number of data blocks
        m: Number of parity blocks
        
    Returns:
        Decoded data
        
    Raises:
        ECCError: If decoding fails or reedsolo is not available
    """
    try:
        import reedsolo
    except ImportError:
        raise ECCError("reedsolo package not available for error correction")
    
    try:
        # Create Reed-Solomon codec
        rs = reedsolo.RSCodec(m)
        
        # Combine data and parity
        encoded = data + parity
        
        # Decode
        decoded = rs.decode(encoded)
        
        return decoded
        
    except Exception as e:
        raise ECCError(f"Reed-Solomon decoding failed: {e}")


def is_ecc_available() -> bool:
    """Check if error correction is available.
    
    Returns:
        True if reedsolo is available
    """
    try:
        import reedsolo
        return True
    except ImportError:
        return False


def calculate_ecc_overhead(
    data_size: int,
    k: int = 10,
    m: int = 2,
) -> int:
    """Calculate ECC overhead in bytes.
    
    Args:
        data_size: Size of original data
        k: Number of data blocks
        m: Number of parity blocks
        
    Returns:
        ECC overhead in bytes
    """
    if not is_ecc_available():
        return 0
    
    # Calculate block size
    block_size = (data_size + k - 1) // k
    
    # Calculate total overhead
    total_blocks = k + m
    total_size = total_blocks * block_size
    
    return total_size - data_size


def encode_block_ecc(
    data: bytes,
    block_size: int = 4096,
    k: int = 10,
    m: int = 2,
) -> Tuple[bytes, Optional[bytes]]:
    """Encode a data block with ECC.
    
    Args:
        data: Data to encode
        block_size: ECC block size
        k: Number of data blocks
        m: Number of parity blocks
        
    Returns:
        Tuple of (encoded_data, parity_data)
        
    Raises:
        ECCError: If encoding fails
    """
    if not is_ecc_available():
        return data, None
    
    # Pad data to block size
    padded_data = data.ljust(block_size, b'\x00')
    
    # Encode with Reed-Solomon
    return encode_reed_solomon(padded_data, k, m)


def decode_block_ecc(
    data: bytes,
    parity: Optional[bytes],
    block_size: int = 4096,
    k: int = 10,
    m: int = 2,
) -> bytes:
    """Decode a data block with ECC.
    
    Args:
        data: Data part
        parity: Parity part (None if ECC not used)
        block_size: ECC block size
        k: Number of data blocks
        m: Number of parity blocks
        
    Returns:
        Decoded data
        
    Raises:
        ECCError: If decoding fails
    """
    if parity is None:
        return data
    
    if not is_ecc_available():
        raise ECCError("ECC data present but reedsolo not available")
    
    try:
        # Decode with Reed-Solomon
        decoded = decode_reed_solomon(data, parity, k, m)
        
        # Remove padding
        return decoded.rstrip(b'\x00')
        
    except Exception as e:
        raise ECCError(f"Block ECC decoding failed: {e}")


def verify_ecc_data(
    data: bytes,
    parity: Optional[bytes],
    k: int = 10,
    m: int = 2,
) -> bool:
    """Verify ECC data integrity.
    
    Args:
        data: Data part
        parity: Parity part
        k: Number of data blocks
        m: Number of parity blocks
        
    Returns:
        True if data is valid
    """
    if parity is None:
        return True
    
    if not is_ecc_available():
        return False
    
    try:
        # Try to decode
        decode_reed_solomon(data, parity, k, m)
        return True
    except ECCError:
        return False
