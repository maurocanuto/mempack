"""Logging configuration for MemPack."""

from __future__ import annotations

import logging
import sys
from typing import Optional


def setup_logger(
    name: str = "mempack",
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    stream: Optional[object] = None,
) -> logging.Logger:
    """Set up a logger for MemPack components.
    
    Args:
        name: Logger name
        level: Logging level
        format_string: Custom format string
        stream: Output stream (defaults to stderr)
        
    Returns:
        Configured logger
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    if stream is None:
        stream = sys.stderr
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    handler = logging.StreamHandler(stream)
    handler.setLevel(level)
    
    formatter = logging.Formatter(format_string)
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    logger.propagate = False
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific component.
    
    Args:
        name: Component name (will be prefixed with 'mempack.')
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f"mempack.{name}")


# Default loggers for common components
builder_logger = get_logger("builder")
retriever_logger = get_logger("retriever")
embedding_logger = get_logger("embedding")
index_logger = get_logger("index")
pack_logger = get_logger("pack")
cli_logger = get_logger("cli")
