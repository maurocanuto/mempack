"""ANN index file format and I/O."""

from __future__ import annotations

import mmap
import os
import struct
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ..errors import FileFormatError, IOError, IndexError
from ..logging import index_logger
from ..types import HNSWParams
from .hnsw import HNSWIndex
from ..pack.spec import ANNHeader, ANN_MAGIC, FORMAT_VERSION, ANN_HEADER_SIZE


class ANNFile:
    """ANN index file reader/writer."""
    
    def __init__(
        self,
        file_path: Union[str, Path],
        mmap_enabled: bool = True,
    ) -> None:
        """Initialize the ANN file.
        
        Args:
            file_path: Path to the .ann file
            mmap_enabled: Whether to use memory mapping
        """
        self.file_path = Path(file_path)
        self.mmap_enabled = mmap_enabled
        
        # File state
        self._file = None
        self._mmap = None
        self._header = None
        self._index = None
        self._id_mapping = None
    
    def write(
        self,
        index: HNSWIndex,
        algorithm: str = "hnsw",
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write an index to the ANN file.
        
        Args:
            index: HNSW index to write
            algorithm: Algorithm name
            params: Algorithm-specific parameters
            
        Raises:
            IOError: If writing fails
        """
        if not index._is_built:
            raise IndexError("Index must be built before writing", algorithm)
        
        try:
            # Ensure directory exists
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create header
            header = ANNHeader(
                magic=ANN_MAGIC,
                version=FORMAT_VERSION,
                algorithm=self._get_algorithm_code(algorithm),
                dimensions=index.dimensions,
                vector_count=len(index),
                id_width=4,  # 32-bit IDs
                params=self._pack_params(params or {}),
            )
            
            # Write header
            header_data = header.pack()
            
            # Save index to temporary file
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_path = temp_file.name
                index.save(temp_path)
                
                # Read index data
                with open(temp_path, 'rb') as f:
                    index_data = f.read()
                
                # Clean up temp file
                os.unlink(temp_path)
            
            # Write complete file
            with open(self.file_path, 'wb') as f:
                f.write(header_data)
                f.write(index_data)
            
            index_logger.info(f"ANN file written: {self.file_path}")
            
        except Exception as e:
            raise IOError(f"Failed to write ANN file: {e}", str(self.file_path))
    
    def read(self) -> HNSWIndex:
        """Read an index from the ANN file.
        
        Returns:
            HNSW index
            
        Raises:
            FileFormatError: If file format is invalid
            IOError: If reading fails
        """
        if not self.file_path.exists():
            raise IOError(f"ANN file not found: {self.file_path}", str(self.file_path))
        
        try:
            # Open file
            self._file = open(self.file_path, 'rb')
            
            if self.mmap_enabled:
                self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
            
            # Read header
            self._read_header()
            
            # Create index
            self._create_index()
            
            # Load index data
            self._load_index_data()
            
            index_logger.info(f"ANN file loaded: {self.file_path}")
            
            return self._index
            
        except Exception as e:
            self.close()
            raise IOError(f"Failed to read ANN file: {e}", str(self.file_path))
    
    def _read_header(self) -> None:
        """Read and validate file header.
        
        Raises:
            FileFormatError: If header is invalid
        """
        if self._mmap:
            header_data = self._mmap[:ANN_HEADER_SIZE]
        else:
            self._file.seek(0)
            header_data = self._file.read(ANN_HEADER_SIZE)
        
        self._header = ANNHeader.unpack(header_data)
        self._header.validate()
        
        index_logger.debug(f"ANN header: algorithm={self._header.algorithm}, dims={self._header.dimensions}")
    
    def _create_index(self) -> None:
        """Create HNSW index from header.
        
        Raises:
            IndexError: If algorithm is unsupported
        """
        algorithm = self._get_algorithm_name(self._header.algorithm)
        
        if algorithm != "hnsw":
            raise IndexError(f"Unsupported algorithm: {algorithm}", algorithm)
        
        # Extract HNSW parameters
        params = self._unpack_params(self._header.params)
        
        print(f"[DEBUG ANN] Creating HNSW index with params: {params}")
        print(f"[DEBUG ANN] Dimensions: {self._header.dimensions}, max_elements: {self._header.vector_count}")
        
        self._index = HNSWIndex(
            dimensions=self._header.dimensions,
            max_elements=self._header.vector_count,
            M=params.get("M", 32),
            ef_construction=params.get("ef_construction", 200),
            ef_search=params.get("ef_search", 64),
            allow_replace_deleted=params.get("allow_replace_deleted", True),
        )
    
    def _load_index_data(self) -> None:
        """Load index data from file.
        
        Raises:
            IOError: If loading fails
        """
        # Read index data (skip header)
        offset = ANN_HEADER_SIZE
        
        if self._mmap:
            index_data = self._mmap[offset:]
        else:
            self._file.seek(offset)
            index_data = self._file.read()
        
        # Save to temporary file and load
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(index_data)
            temp_file.flush()
            
            try:
                self._index.load(temp_path)
            finally:
                os.unlink(temp_path)
    
    def _get_algorithm_code(self, algorithm: str) -> int:
        """Get algorithm code from name.
        
        Args:
            algorithm: Algorithm name
            
        Returns:
            Algorithm code
        """
        algorithm_map = {
            "hnsw": 0,
            "ivfpq": 1,
        }
        
        if algorithm not in algorithm_map:
            raise IndexError(f"Unsupported algorithm: {algorithm}", algorithm)
        
        return algorithm_map[algorithm]
    
    def _get_algorithm_name(self, algorithm_code: int) -> str:
        """Get algorithm name from code.
        
        Args:
            algorithm_code: Algorithm code
            
        Returns:
            Algorithm name
        """
        algorithm_map = {
            0: "hnsw",
            1: "ivfpq",
        }
        
        if algorithm_code not in algorithm_map:
            raise IndexError(f"Unsupported algorithm code: {algorithm_code}", "unknown")
        
        return algorithm_map[algorithm_code]
    
    def _pack_params(self, params: Dict[str, Any]) -> bytes:
        """Pack parameters into bytes.
        
        Args:
            params: Parameters dictionary
            
        Returns:
            Packed parameters
        """
        # Pack HNSW parameters
        M = params.get("M", 32)
        ef_construction = params.get("ef_construction", 200)
        ef_search = params.get("ef_search", 64)
        allow_replace_deleted = params.get("allow_replace_deleted", True)
        
        return struct.pack('<IIII', M, ef_construction, ef_search, 1 if allow_replace_deleted else 0) + b'\x00' * 16
    
    def _unpack_params(self, params_data: bytes) -> Dict[str, Any]:
        """Unpack parameters from bytes.
        
        Args:
            params_data: Packed parameters
            
        Returns:
            Parameters dictionary
        """
        if len(params_data) < 16:
            return {}
        
        M, ef_construction, ef_search, allow_replace_deleted = struct.unpack('<IIII', params_data[:16])
        
        return {
            "M": M,
            "ef_construction": ef_construction,
            "ef_search": ef_search,
            "allow_replace_deleted": bool(allow_replace_deleted),
        }
    
    def get_header(self) -> ANNHeader:
        """Get file header.
        
        Returns:
            File header
        """
        return self._header
    
    def get_stats(self) -> Dict[str, Any]:
        """Get file statistics.
        
        Returns:
            Statistics dictionary
        """
        if self._index is None:
            return {
                "file_size": self.file_path.stat().st_size if self.file_path.exists() else 0,
                "algorithm": "unknown",
                "dimensions": 0,
                "vector_count": 0,
            }
        
        stats = self._index.get_stats()
        stats.update({
            "file_size": self.file_path.stat().st_size if self.file_path.exists() else 0,
            "algorithm": self._get_algorithm_name(self._header.algorithm),
        })
        
        return stats
    
    def close(self) -> None:
        """Close the file and cleanup resources."""
        if self._mmap:
            self._mmap.close()
            self._mmap = None
        
        if self._file:
            self._file.close()
            self._file = None
        
        self._index = None
        self._header = None
    
    def __enter__(self) -> ANNFile:
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
    
    def __del__(self) -> None:
        """Destructor."""
        self.close()
