"""Configuration management for video understanding project."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str
    model_id: str
    max_pixels_multiplier: float = 1.0


@dataclass
class VideoConfig:
    """Video processing configuration."""
    default_fps: float = 1.0
    short_video_fps: float = 2.0
    long_video_fps: float = 0.5
    short_video_threshold: float = 10.0
    long_video_threshold: float = 60.0
    base_pixels: int = 360 * 420


@dataclass
class GenerationConfig:
    """Text generation configuration."""
    max_analysis_tokens: int = 2048
    max_subtitle_tokens: int = 4096
    do_sample: bool = False


@dataclass
class ServerConfig:
    """Server configuration."""
    host: str = "127.0.0.1"
    port: int = 7861
    max_concurrent: int = 1


@dataclass
class AppConfig:
    """Main application configuration."""
    server: ServerConfig = field(default_factory=ServerConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    default_model: str = "2B"


# Model mappings
MODELS = {
    "2B": ModelConfig("2B", "Qwen/Qwen3-VL-2B-Instruct", 1.0),
    "4B": ModelConfig("4B", "Qwen/Qwen3-VL-4B-Instruct", 1.5),
    "8B": ModelConfig("8B", "Qwen/Qwen3-VL-8B-Instruct", 2.0),
}

# Global configuration instances
VIDEO_CONFIG = VideoConfig()
GENERATION_CONFIG = GenerationConfig()
SERVER_CONFIG = ServerConfig()

# Prompt templates
ANALYSIS_PROMPT = """请理解这个视频，并严格按下面结构输出中文结果：

## 视频内容总结
用一段话总结视频讲了什么。

## 视频主要场景
用列表列出视频里出现的主要场景、人物、物体或动作。

## 视频时间线概述
按时间顺序概述视频里的关键变化；如果无法精确判断时间点，可以用"开头 / 中段 / 结尾"。

## AI生成的讲解文案
生成一段适合配音讲解的中文文案。

要求：只根据视频可见内容回答，不要编造无法观察到的信息。
"""

SUBTITLE_PROMPT_TEMPLATE = """你是一个专业短视频旁白字幕编剧。

下面是 Qwen3-VL 对视频的理解结果：

{analysis_result}

视频时长约为：{duration_text}

{style_instruction}

请根据上面的理解结果，生成一份中文旁白字幕。

要求：
1. 只输出标准 SRT 字幕内容，不要输出 Markdown，不要输出解释。
2. 字幕序号从 1 开始递增。
3. 时间格式必须是 SRT 标准格式：HH:MM:SS,mmm --> HH:MM:SS,mmm。
4. 字幕时间必须从 00:00:00,000 开始，并覆盖到视频结尾附近。
5. 每条字幕建议 1 到 2 行中文旁白，语言自然，适合配音。
6. 不要编造视频中看不到的信息。
"""


def get_config() -> AppConfig:
    """Get application configuration."""
    return AppConfig()
