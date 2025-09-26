"""Custom exceptions for MemPack."""

from __future__ import annotations


class MemPackError(Exception):
    """Base exception for all MemPack errors."""
    pass


class CorruptBlockError(MemPackError):
    """Raised when a block fails integrity checks."""
    
    def __init__(self, block_id: int, expected_checksum: int, actual_checksum: int) -> None:
        self.block_id = block_id
        self.expected_checksum = expected_checksum
        self.actual_checksum = actual_checksum
        super().__init__(
            f"Block {block_id} failed integrity check: "
            f"expected {expected_checksum:08x}, got {actual_checksum:08x}"
        )


class VerifyError(MemPackError):
    """Raised when file verification fails."""
    
    def __init__(self, message: str, details: str = "") -> None:
        self.details = details
        super().__init__(f"Verification failed: {message}" + (f" ({details})" if details else ""))


class FileFormatError(MemPackError):
    """Raised when file format is invalid or unsupported."""
    
    def __init__(self, message: str, file_path: str = "") -> None:
        self.file_path = file_path
        super().__init__(f"Invalid file format: {message}" + (f" in {file_path}" if file_path else ""))


class IndexError(MemPackError):
    """Raised when index operations fail."""
    
    def __init__(self, message: str, index_type: str = "") -> None:
        self.index_type = index_type
        super().__init__(f"Index error: {message}" + (f" ({index_type})" if index_type else ""))


class EmbeddingError(MemPackError):
    """Raised when embedding operations fail."""
    
    def __init__(self, message: str, model: str = "") -> None:
        self.model = model
        super().__init__(f"Embedding error: {message}" + (f" ({model})" if model else ""))


class CompressionError(MemPackError):
    """Raised when compression/decompression fails."""
    
    def __init__(self, message: str, algorithm: str = "") -> None:
        self.algorithm = algorithm
        super().__init__(f"Compression error: {message}" + (f" ({algorithm})" if algorithm else ""))


class ECCError(MemPackError):
    """Raised when error correction operations fail."""
    
    def __init__(self, message: str, block_id: int = -1) -> None:
        self.block_id = block_id
        super().__init__(f"ECC error: {message}" + (f" (block {block_id})" if block_id >= 0 else ""))


class ChunkingError(MemPackError):
    """Raised when text chunking fails."""
    
    def __init__(self, message: str, text_length: int = 0) -> None:
        self.text_length = text_length
        super().__init__(f"Chunking error: {message}" + (f" (text length: {text_length})" if text_length > 0 else ""))


class ConfigurationError(MemPackError):
    """Raised when configuration is invalid."""
    
    def __init__(self, message: str, parameter: str = "") -> None:
        self.parameter = parameter
        super().__init__(f"Configuration error: {message}" + (f" (parameter: {parameter})" if parameter else ""))


class IOError(MemPackError):
    """Raised when I/O operations fail."""
    
    def __init__(self, message: str, file_path: str = "") -> None:
        self.file_path = file_path
        super().__init__(f"I/O error: {message}" + (f" ({file_path})" if file_path else ""))


class NotFoundError(MemPackError):
    """Raised when a requested resource is not found."""
    
    def __init__(self, resource_type: str, resource_id: str) -> None:
        self.resource_type = resource_type
        self.resource_id = resource_id
        super().__init__(f"{resource_type} not found: {resource_id}")


class ValidationError(MemPackError):
    """Raised when data validation fails."""
    
    def __init__(self, message: str, field: str = "") -> None:
        self.field = field
        super().__init__(f"Validation error: {message}" + (f" (field: {field})" if field else ""))
