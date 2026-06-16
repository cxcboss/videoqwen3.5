"""System monitoring utilities for CPU, memory, and disk usage."""

import os
import platform
from typing import Any

import psutil


def get_system_status() -> dict[str, Any]:
    """Get comprehensive system status.
    
    Returns:
        Dictionary with system status information
    """
    # CPU
    cpu_percent = psutil.cpu_percent(interval=0.5)
    cpu_count = psutil.cpu_count()
    cpu_freq = psutil.cpu_freq()
    
    # Memory
    memory = psutil.virtual_memory()
    
    # Disk
    disk = psutil.disk_usage("/")
    
    # Platform info
    system_info = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }
    
    return {
        "cpu_percent": cpu_percent,
        "cpu_count": cpu_count,
        "cpu_freq_current": cpu_freq.current if cpu_freq else 0,
        "cpu_freq_max": cpu_freq.max if cpu_freq else 0,
        "memory_total": memory.total,
        "memory_used": memory.used,
        "memory_available": memory.available,
        "memory_percent": memory.percent,
        "disk_total": disk.total,
        "disk_used": disk.used,
        "disk_free": disk.free,
        "disk_percent": disk.percent,
        "system_info": system_info,
    }


def format_bytes(size_bytes: int) -> str:
    """Format bytes to human readable string.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string (e.g., "16.0 GB")
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def get_system_status_display() -> str:
    """Get formatted system status for display.
    
    Returns:
        Formatted markdown string
    """
    status = get_system_status()
    
    # CPU
    cpu_bar = _make_bar(status["cpu_percent"])
    cpu_info = (
        f"**CPU**\n"
        f"使用率: {cpu_bar} {status['cpu_percent']:.1f}%\n"
        f"核心数: {status['cpu_count']} | "
        f"频率: {status['cpu_freq_current']:.0f} MHz"
    )
    
    # Memory
    mem = status
    mem_used = format_bytes(mem["memory_used"])
    mem_total = format_bytes(mem["memory_total"])
    mem_bar = _make_bar(mem["memory_percent"])
    mem_info = (
        f"**内存**\n"
        f"使用率: {mem_bar} {mem['memory_percent']:.1f}%\n"
        f"已用: {mem_used} / {mem_total}"
    )
    
    # Disk
    disk_used = format_bytes(mem["disk_used"])
    disk_total = format_bytes(mem["disk_total"])
    disk_bar = _make_bar(mem["disk_percent"])
    disk_info = (
        f"**磁盘**\n"
        f"使用率: {disk_bar} {mem['disk_percent']:.1f}%\n"
        f"已用: {disk_used} / {disk_total}"
    )
    
    # System
    sys_info = status["system_info"]
    system_text = (
        f"**系统信息**\n"
        f"{sys_info['system']} {sys_info['release']}\n"
        f"架构: {sys_info['machine']} | Python: {sys_info['python']}"
    )
    
    return f"{cpu_info}\n\n{mem_info}\n\n{disk_info}\n\n{system_text}"


def _make_bar(percent: float, width: int = 20) -> str:
    """Make a progress bar.
    
    Args:
        percent: Percentage (0-100)
        width: Bar width in characters
        
    Returns:
        Progress bar string
    """
    filled = int(width * percent / 100)
    empty = width - filled
    return f"`{'█' * filled}{'░' * empty}`"
