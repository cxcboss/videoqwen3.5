"""Video processing module for handling video input and format conversion."""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional, Tuple

from qwen_vl_utils import vision_process

from config import VIDEO_CONFIG, VideoConfig
from utils.file_utils import temp_video_dir


class VideoProcessor:
    """Handles video processing, format conversion, and message building."""
    
    def __init__(self, config: Optional[VideoConfig] = None):
        """Initialize video processor.
        
        Args:
            config: Video configuration (uses default if None)
        """
        self.config = config or VIDEO_CONFIG
        self._torchcodec_initialized = False
    
    def ensure_torchcodec_backend(self) -> None:
        """Ensure torchcodec backend is initialized (once)."""
        if self._torchcodec_initialized:
            return
        
        os.environ["FORCE_QWENVL_VIDEO_READER"] = "torchcodec"
        vision_process.FORCE_QWENVL_VIDEO_READER = "torchcodec"
        
        if "torchcodec" in vision_process.VIDEO_READER_BACKENDS:
            vision_process.VIDEO_READER_BACKENDS["torchvision"] = (
                vision_process.VIDEO_READER_BACKENDS["torchcodec"]
            )
        
        vision_process.get_video_reader_backend.cache_clear()
        self._torchcodec_initialized = True
    
    def build_video_messages(
        self,
        video_path: str,
        model_size: str = "2B",
        video_duration: Optional[float] = None
    ) -> list[dict[str, Any]]:
        """Build video messages for Qwen3-VL.
        
        Args:
            video_path: Path to video file
            model_size: Model size (2B, 4B, 8B)
            video_duration: Video duration in seconds (for dynamic fps)
            
        Returns:
            List of message dictionaries
        """
        # Calculate pixels based on model size
        size_multiplier = {"2B": 1.0, "4B": 1.5, "8B": 2.0}.get(model_size, 1.0)
        max_pixels = int(self.config.base_pixels * size_multiplier)
        
        # Calculate fps based on video duration
        if video_duration and video_duration < self.config.short_video_threshold:
            fps = self.config.short_video_fps
        elif video_duration and video_duration > self.config.long_video_threshold:
            fps = self.config.long_video_fps
        else:
            fps = self.config.default_fps
        
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": str(Path(video_path).resolve()),
                        "max_pixels": max_pixels,
                        "fps": fps,
                    },
                    {"type": "text", "text": "请分析这个视频"},
                ],
            }
        ]
    
    def process_vision_info(
        self,
        messages: list[dict[str, Any]]
    ) -> Tuple[list, list, dict, Optional[list]]:
        """Process vision information from messages.
        
        Args:
            messages: Video messages
            
        Returns:
            Tuple of (images, videos, video_kwargs, video_metadatas)
        """
        try:
            images, videos, video_kwargs = vision_process.process_vision_info(
                messages,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
        except Exception:
            # Fallback: try with converted video
            raise
        
        # Process videos and metadata
        video_metadatas = None
        if videos is not None:
            videos, video_metadatas = zip(*videos)
            videos = list(videos)
            video_metadatas = list(video_metadatas)
        
        return images, videos, video_kwargs, video_metadatas
    
    def process_video_messages(
        self,
        video_path: str,
        model_size: str = "2B"
    ) -> Tuple[list, list, list, dict, Optional[list], Optional[float]]:
        """Process video messages with automatic format conversion.
        
        Args:
            video_path: Path to video file
            model_size: Model size (2B, 4B, 8B)
            
        Returns:
            Tuple of (messages, images, videos, video_kwargs, video_metadatas, duration)
            
        Raises:
            RuntimeError: If video processing fails
        """
        try:
            messages = self.build_video_messages(video_path, model_size)
            images, videos, video_kwargs, video_metadatas = self.process_vision_info(messages)
            
            # Estimate duration
            duration = self._estimate_video_duration(video_metadatas)
            
            return messages, images, videos, video_kwargs, video_metadatas, duration
            
        except Exception:
            # Try converting video format
            with temp_video_dir() as tmp_dir:
                converted_video_path = self._convert_video_to_compatible_mp4(video_path, tmp_dir)
                messages = self.build_video_messages(converted_video_path, model_size)
                images, videos, video_kwargs, video_metadatas = self.process_vision_info(messages)
                
                duration = self._estimate_video_duration(video_metadatas)
                
                return messages, images, videos, video_kwargs, video_metadatas, duration
    
    def _estimate_video_duration(
        self,
        video_metadatas: Optional[list[dict[str, Any]]]
    ) -> Optional[float]:
        """Estimate video duration from metadata.
        
        Args:
            video_metadatas: Video metadata list
            
        Returns:
            Duration in seconds, or None if unavailable
        """
        if not video_metadatas:
            return None
        
        metadata = video_metadatas[0]
        fps = metadata.get("fps")
        total_frames = metadata.get("total_num_frames")
        
        if not fps or not total_frames:
            return None
        
        return float(total_frames) / float(fps)
    
    def _convert_video_to_compatible_mp4(
        self,
        video_path: str,
        tmp_dir: Path,
        timeout: int = 120
    ) -> str:
        """Convert video to compatible MP4 format using ffmpeg.
        
        Args:
            video_path: Path to input video
            tmp_dir: Temporary directory for output
            timeout: Conversion timeout in seconds
            
        Returns:
            Path to converted video
            
        Raises:
            RuntimeError: If conversion fails or ffmpeg not found
        """
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            raise RuntimeError("系统未安装 ffmpeg，无法转换视频格式")
        
        output_path = tmp_dir / "compatible.mp4"
        
        command = [
            ffmpeg_path,
            "-y",
            "-i", video_path,
            "-map", "0:v:0",
            "-map", "0:a?",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
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
            timeout=timeout,
        )
        
        if result.returncode != 0:
            error_text = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"视频转换失败：{error_text[-500:]}")
        
        return str(output_path)


# Global video processor instance
video_processor = VideoProcessor()
