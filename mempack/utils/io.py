"""I/O utilities for MemPack."""

from __future__ import annotations

import mmap
import os
import tempfile
from pathlib import Path
from typing import BinaryIO, Optional, Union

from ..errors import IOError


def atomic_write(
    file_path: Union[str, Path],
    data: bytes,
    mode: str = "wb",
    temp_suffix: str = ".tmp",
) -> None:
    """Atomically write data to a file.
    
    Args:
        file_path: Path to write to
        data: Data to write
        mode: File mode
        temp_suffix: Suffix for temporary file
        
    Raises:
        IOError: If write operation fails
    """
    file_path = Path(file_path)
    temp_path = file_path.with_suffix(file_path.suffix + temp_suffix)
    
    try:
        # Write to temporary file
        with open(temp_path, mode) as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        
        # Atomic rename
        os.rename(temp_path, file_path)
        
    except Exception as e:
        # Clean up temporary file on error
        if temp_path.exists():
            temp_path.unlink()
        raise IOError(f"Failed to write {file_path}: {e}", str(file_path))
    
    finally:
        # Ensure temporary file is cleaned up
        if temp_path.exists():
            temp_path.unlink()


def pread(fd: int, size: int, offset: int) -> bytes:
    """Read data from a file descriptor at a specific offset.
    
    Args:
        fd: File descriptor
        size: Number of bytes to read
        offset: Offset to read from
        
    Returns:
        Read data
        
    Raises:
        IOError: If read operation fails
    """
    try:
        # Seek to offset
        os.lseek(fd, offset, os.SEEK_SET)
        
        # Read data
        data = b""
        remaining = size
        while remaining > 0:
            chunk = os.read(fd, remaining)
            if not chunk:
                break
            data += chunk
            remaining -= len(chunk)
        
        return data
        
    except Exception as e:
        raise IOError(f"Failed to read {size} bytes at offset {offset}: {e}")


def mmap_file(
    file_path: Union[str, Path],
    mode: str = "r",
    offset: int = 0,
    length: Optional[int] = None,
) -> mmap.mmap:
    """Memory map a file.
    
    Args:
        file_path: Path to file
        mode: Mapping mode ('r', 'w', 'c')
        offset: Offset to start mapping
        length: Length to map (None = entire file)
        
    Returns:
        Memory mapped file
        
    Raises:
        IOError: If mapping fails
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise IOError(f"File not found: {file_path}", str(file_path))
    
    try:
        with open(file_path, "rb" if mode == "r" else "r+b") as f:
            if length is None:
                length = file_path.stat().st_size - offset
            
            return mmap.mmap(f.fileno(), length, access=getattr(mmap, f"ACCESS_{mode.upper()}"), offset=offset)
            
    except Exception as e:
        raise IOError(f"Failed to memory map {file_path}: {e}", str(file_path))


def align_offset(offset: int, alignment: int = 4096) -> int:
    """Align an offset to the specified boundary.
    
    Args:
        offset: Original offset
        alignment: Alignment boundary
        
    Returns:
        Aligned offset
    """
    return ((offset + alignment - 1) // alignment) * alignment


def ensure_dir(file_path: Union[str, Path]) -> None:
    """Ensure the directory containing the file exists.
    
    Args:
        file_path: Path to file
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def get_temp_file(
    prefix: str = "mempack_",
    suffix: str = ".tmp",
    dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Get a temporary file path.
    
    Args:
        prefix: File prefix
        suffix: File suffix
        dir: Directory (None = system temp)
        
    Returns:
        Temporary file path
    """
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=dir)
    os.close(fd)  # Close the file descriptor, we just want the path
    return Path(path)


def safe_remove(file_path: Union[str, Path]) -> None:
    """Safely remove a file, ignoring errors.
    
    Args:
        file_path: Path to file
    """
    try:
        Path(file_path).unlink()
    except (OSError, FileNotFoundError):
        pass


def get_file_size(file_path: Union[str, Path]) -> int:
    """Get file size in bytes.
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in bytes
        
    Raises:
        IOError: If file doesn't exist or can't be accessed
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise IOError(f"File not found: {file_path}", str(file_path))
    
    try:
        return file_path.stat().st_size
    except Exception as e:
        raise IOError(f"Failed to get size of {file_path}: {e}", str(file_path))


def copy_file_range(
    src_fd: int,
    dst_fd: int,
    count: int,
    src_offset: int = 0,
    dst_offset: Optional[int] = None,
) -> int:
    """Copy data between file descriptors efficiently.
    
    Args:
        src_fd: Source file descriptor
        dst_fd: Destination file descriptor
        count: Number of bytes to copy
        src_offset: Source offset
        dst_offset: Destination offset (None = current position)
        
    Returns:
        Number of bytes copied
        
    Raises:
        IOError: If copy operation fails
    """
    try:
        if dst_offset is not None:
            os.lseek(dst_fd, dst_offset, os.SEEK_SET)
        
        return os.copy_file_range(src_fd, dst_fd, count, src_offset)
        
    except Exception as e:
        raise IOError(f"Failed to copy {count} bytes: {e}")
