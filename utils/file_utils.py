"""File utility functions for temporary file management."""

import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Generator


@contextmanager
def temp_video_dir(prefix: str = "qwen3vl_video_") -> Generator[Path, None, None]:
    """Context manager for automatic cleanup of temporary video directories.
    
    Args:
        prefix: Prefix for temporary directory name
        
    Yields:
        Path to temporary directory
        
    Example:
        >>> with temp_video_dir() as tmp_dir:
        ...     output_path = tmp_dir / "converted.mp4"
        ...     # do something with output_path
        ... # tmp_dir is automatically cleaned up
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix=prefix))
    try:
        yield tmp_dir
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


def normalize_video_path(video) -> str | None:
    """Normalize video path from various input formats.
    
    Args:
        video: Video input (str, dict, or object with path/name attribute)
        
    Returns:
        Normalized video path or None if invalid
    """
    if not video:
        return None
    
    if isinstance(video, str):
        return video
    
    if isinstance(video, dict):
        return video.get("path") or video.get("name") or video.get("video")
    
    return getattr(video, "path", None) or getattr(video, "name", None)
