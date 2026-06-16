"""Video splitting module for handling long videos by segmenting them."""

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

from utils.file_utils import temp_video_dir
from utils.formatters import format_srt_time, srt_time_to_seconds


# Default segment duration in seconds
SEGMENT_DURATION = 60


def get_video_duration(video_path: str) -> Optional[float]:
    """Get video duration using ffprobe.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Duration in seconds, or None if unavailable
    """
    ffprobe_path = shutil.which("ffprobe")
    if not ffprobe_path:
        return None
    
    try:
        result = subprocess.run(
            [
                ffprobe_path,
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except (subprocess.TimeoutExpired, ValueError):
        pass
    
    return None


def split_video(
    video_path: str,
    segment_duration: float = SEGMENT_DURATION,
) -> Tuple[List[str], float]:
    """Split video into segments of specified duration.
    
    Args:
        video_path: Path to input video
        segment_duration: Duration of each segment in seconds
        
    Returns:
        Tuple of (list of segment paths, total duration)
        
    Raises:
        RuntimeError: If ffmpeg not found or splitting fails
    """
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise RuntimeError("系统未安装 ffmpeg，无法分割视频")
    
    total_duration = get_video_duration(video_path)
    if total_duration is None:
        raise RuntimeError("无法获取视频时长")
    
    if total_duration <= segment_duration:
        return [video_path], total_duration
    
    tmp_context = temp_video_dir(prefix="qwen3vl_split_")
    tmp_dir = tmp_context.__enter__()
    
    segment_paths = []
    start = 0.0
    segment_index = 0
    
    try:
        while start < total_duration:
            end = min(start + segment_duration, total_duration)
            output_path = tmp_dir / f"segment_{segment_index:03d}.mp4"
            
            command = [
                ffmpeg_path,
                "-y",
                "-ss", str(start),
                "-i", video_path,
                "-t", str(end - start),
                "-map", "0:v:0",
                "-map", "0:a?",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "ultrafast",
                "-crf", "28",
                "-an",
                "-movflags", "+faststart",
                str(output_path),
            ]
            
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=120,
            )
            
            if result.returncode != 0:
                error_text = result.stderr.strip() or result.stdout.strip()
                raise RuntimeError(f"视频分割失败：{error_text[-500:]}")
            
            segment_paths.append(str(output_path))
            segment_index += 1
            start = end
        
        # Keep track of tmp_dir so we can clean up later
        # Store the context manager reference for cleanup
        split_video._active_tmp_context = tmp_context
        split_video._active_tmp_dir = tmp_dir
        
        return segment_paths, total_duration
        
    except Exception:
        tmp_context.__exit__(None, None, None)
        raise


def cleanup_split_files() -> None:
    """Clean up temporary split video files."""
    tmp_context = getattr(split_video, "_active_tmp_context", None)
    if tmp_context is not None:
        try:
            tmp_context.__exit__(None, None, None)
        except Exception:
            pass
        split_video._active_tmp_context = None
        split_video._active_tmp_dir = None


def merge_srt_segments(
    srt_segments: List[str],
    segment_durations: List[float],
) -> str:
    """Merge multiple SRT segments into a single unified SRT.
    
    Adjusts timestamps so each segment continues from where the previous one ended.
    Ensures subtitle numbering is sequential.
    
    Args:
        srt_segments: List of SRT text for each segment
        segment_durations: Duration of each segment in seconds
        
    Returns:
        Merged SRT text
    """
    merged_lines = []
    subtitle_index = 1
    time_offset = 0.0
    
    for i, (srt_text, seg_duration) in enumerate(zip(srt_segments, segment_durations)):
        if not srt_text or not srt_text.strip():
            time_offset += seg_duration
            continue
        
        lines = srt_text.strip().split('\n')
        j = 0
        
        while j < len(lines):
            line = lines[j].strip()
            
            # Skip empty lines
            if not line:
                j += 1
                continue
            
            # Check if this is a subtitle number
            if re.match(r'^\d+$', line):
                # Write new sequential number
                merged_lines.append(str(subtitle_index))
                subtitle_index += 1
                j += 1
                
                # Next line should be timestamp
                if j < len(lines):
                    timestamp_line = lines[j].strip()
                    if '-->' in timestamp_line:
                        adjusted_timestamp = _adjust_timestamp(timestamp_line, time_offset)
                        merged_lines.append(adjusted_timestamp)
                        j += 1
                        
                        # Collect subtitle text lines
                        while j < len(lines):
                            text_line = lines[j].strip()
                            if not text_line or re.match(r'^\d+$', text_line):
                                break
                            merged_lines.append(text_line)
                            j += 1
                        
                        merged_lines.append("")  # Blank line separator
                        continue
            
            j += 1
        
        time_offset += seg_duration
    
    return "\n".join(merged_lines).strip()


def _adjust_timestamp(timestamp_line: str, offset_seconds: float) -> str:
    """Adjust SRT timestamp by adding offset.
    
    Args:
        timestamp_line: SRT timestamp line (e.g., "00:00:01,000 --> 00:00:03,000")
        offset_seconds: Time offset to add
        
    Returns:
        Adjusted timestamp line
    """
    parts = timestamp_line.split(' --> ')
    if len(parts) != 2:
        return timestamp_line
    
    start_str, end_str = parts
    
    try:
        start_sec = srt_time_to_seconds(start_str.strip()) + offset_seconds
        end_sec = srt_time_to_seconds(end_str.strip()) + offset_seconds
        
        return f"{format_srt_time(start_sec)} --> {format_srt_time(end_sec)}"
    except (ValueError, IndexError):
        return timestamp_line
