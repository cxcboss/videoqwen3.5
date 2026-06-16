"""Formatting utility functions for time and duration."""

from typing import Optional


def format_srt_time(seconds: float) -> str:
    """Convert seconds to SRT time format (HH:MM:SS,mmm).
    
    Args:
        seconds: Time in seconds
        
    Returns:
        SRT formatted time string
        
    Example:
        >>> format_srt_time(61.5)
        '00:01:01,500'
    """
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def format_duration(seconds: Optional[float]) -> str:
    """Format video duration for display.
    
    Args:
        seconds: Duration in seconds, or None if unknown
        
    Returns:
        Formatted duration string
        
    Example:
        >>> format_duration(65.5)
        '65.5 秒，字幕结尾时间不要超过 00:01:05,500'
    """
    if seconds is None or seconds <= 0:
        return "未知，请按开头 / 中段 / 结尾合理安排字幕时间"
    return f"{seconds:.1f} 秒，字幕结尾时间不要超过 {format_srt_time(seconds)}"


def format_elapsed(seconds: float) -> str:
    """Format elapsed time for display.
    
    Args:
        seconds: Elapsed time in seconds
        
    Returns:
        Formatted elapsed time string
        
    Example:
        >>> format_elapsed(65.5)
        '识别耗时：1 分 5.50 秒'
    """
    if seconds < 60:
        return f"识别耗时：{seconds:.2f} 秒"
    
    minutes = int(seconds // 60)
    remaining_seconds = seconds - minutes * 60
    return f"识别耗时：{minutes} 分 {remaining_seconds:.2f} 秒"


def srt_time_to_seconds(srt_time: str) -> float:
    """Convert SRT time format to seconds.
    
    Args:
        srt_time: SRT time string (HH:MM:SS,mmm)
        
    Returns:
        Time in seconds
        
    Example:
        >>> srt_time_to_seconds('00:01:01,500')
        61.5
    """
    hours, minutes, rest = srt_time.split(':')
    secs, millis = rest.split(',')
    return int(hours) * 3600 + int(minutes) * 60 + int(secs) + int(millis) / 1000
