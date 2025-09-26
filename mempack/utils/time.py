"""Time utilities for MemPack."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator, Optional


class Timer:
    """A simple timer for measuring elapsed time."""
    
    def __init__(self) -> None:
        self._start_time: Optional[float] = None
        self._elapsed: float = 0.0
        self._running: bool = False
    
    def start(self) -> None:
        """Start the timer."""
        if self._running:
            return
        self._start_time = time.perf_counter()
        self._running = True
    
    def stop(self) -> float:
        """Stop the timer and return elapsed time in seconds.
        
        Returns:
            Elapsed time in seconds
        """
        if not self._running or self._start_time is None:
            return self._elapsed
        
        self._elapsed += time.perf_counter() - self._start_time
        self._running = False
        return self._elapsed
    
    def reset(self) -> None:
        """Reset the timer."""
        self._start_time = None
        self._elapsed = 0.0
        self._running = False
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self._running and self._start_time is not None:
            return self._elapsed + (time.perf_counter() - self._start_time)
        return self._elapsed
    
    @property
    def running(self) -> bool:
        """Check if timer is running."""
        return self._running
    
    def __enter__(self) -> Timer:
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()


@contextmanager
def time_ms() -> Generator[Timer, None, None]:
    """Context manager that returns a timer measuring milliseconds.
    
    Yields:
        Timer instance
        
    Example:
        with time_ms() as timer:
            # do work
            pass
        print(f"Operation took {timer.elapsed * 1000:.2f}ms")
    """
    timer = Timer()
    timer.start()
    try:
        yield timer
    finally:
        timer.stop()


def time_ms_decorator(func):
    """Decorator to measure function execution time in milliseconds.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function that prints execution time
    """
    def wrapper(*args, **kwargs):
        with time_ms() as timer:
            result = func(*args, **kwargs)
        print(f"{func.__name__} took {timer.elapsed * 1000:.2f}ms")
        return result
    return wrapper


def get_timestamp() -> float:
    """Get current timestamp in seconds since epoch.
    
    Returns:
        Current timestamp
    """
    return time.time()


def get_timestamp_ms() -> int:
    """Get current timestamp in milliseconds since epoch.
    
    Returns:
        Current timestamp in milliseconds
    """
    return int(time.time() * 1000)


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_rate(items: int, seconds: float) -> str:
    """Format processing rate.
    
    Args:
        items: Number of items processed
        seconds: Time taken in seconds
        
    Returns:
        Formatted rate string
    """
    if seconds == 0:
        return "∞ items/s"
    
    rate = items / seconds
    
    if rate < 1:
        return f"{1/rate:.1f}s/item"
    elif rate < 1000:
        return f"{rate:.1f} items/s"
    elif rate < 1000000:
        return f"{rate/1000:.1f}k items/s"
    else:
        return f"{rate/1000000:.1f}M items/s"
