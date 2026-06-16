"""Subtitle generation module for creating SRT subtitles."""

import re
from typing import Any, Optional

import torch

from config import SUBTITLE_PROMPT_TEMPLATE, GenerationConfig
from core.model_manager import model_manager
from utils.formatters import format_duration, srt_time_to_seconds


class SubtitleGenerator:
    """Generates SRT subtitles from video analysis results."""
    
    def __init__(self, config: Optional[GenerationConfig] = None):
        """Initialize subtitle generator.
        
        Args:
            config: Generation configuration (uses default if None)
        """
        self.config = config or GenerationConfig()
    
    def normalize_srt(self, subtitle_text: str) -> str:
        """Normalize SRT subtitle text.
        
        Args:
            subtitle_text: Raw SRT text
            
        Returns:
            Normalized SRT text
        """
        text = subtitle_text.strip()
        
        # Remove markdown code blocks
        text = re.sub(r"^```(?:srt)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text).strip()
        
        normalized_lines: list[str] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            
            if not line:
                if normalized_lines and normalized_lines[-1] != "":
                    normalized_lines.append("")
                continue
            
            # Handle combined number and timestamp
            combined_match = re.match(
                r"^(\d+)[\.\)]?\s+(\d{2}:\d{2}:\d{2},\d{3}\s+-->\s+\d{2}:\d{2}:\d{2},\d{3})$",
                line,
            )
            if combined_match:
                if normalized_lines and normalized_lines[-1] != "":
                    normalized_lines.append("")
                normalized_lines.append(combined_match.group(1))
                normalized_lines.append(combined_match.group(2))
                continue
            
            # Handle number only
            number_match = re.match(r"^(\d+)[\.\)]$", line)
            if number_match:
                if normalized_lines and normalized_lines[-1] != "":
                    normalized_lines.append("")
                normalized_lines.append(number_match.group(1))
                continue
            
            normalized_lines.append(line)
        
        return "\n".join(normalized_lines).strip()
    
    def build_style_instruction(self, style_text: Optional[str]) -> str:
        """Build style instruction for subtitle generation.
        
        Args:
            style_text: Reference style text
            
        Returns:
            Style instruction string
        """
        style_text = (style_text or "").strip()
        
        if not style_text:
            return "未提供风格样本文案，请使用自然、清晰、适合短视频旁白的表达。"
        
        return f"""下面是用户提供的一篇或多篇参考文案，请学习其表达风格、句式节奏、用词倾向和情绪语气，但不要复制原文内容：

{style_text}"""
    
    def build_subtitle_prompt(
        self,
        analysis_result: str,
        duration_text: str,
        style_instruction: str
    ) -> str:
        """Build prompt for subtitle generation.
        
        Args:
            analysis_result: Video analysis result
            duration_text: Formatted duration text
            style_instruction: Style instruction
            
        Returns:
            Complete subtitle prompt
        """
        return SUBTITLE_PROMPT_TEMPLATE.format(
            analysis_result=analysis_result,
            duration_text=duration_text,
            style_instruction=style_instruction,
        )
    
    def calculate_max_tokens(self, duration_seconds: Optional[float]) -> int:
        """Calculate max tokens based on video duration.
        
        Args:
            duration_seconds: Video duration in seconds
            
        Returns:
            Maximum tokens for generation
        """
        duration = duration_seconds or 30.0
        
        # Analysis tokens: 512 base + 64 per 10 seconds, max 2048
        analysis_tokens = min(2048, 512 + int(duration / 10) * 64)
        
        # Subtitle tokens: 400 base + 8 per second, max 4096
        subtitle_tokens = min(4096, 400 + int(duration) * 8)
        
        return analysis_tokens + subtitle_tokens
    
    def generate_subtitle(
        self,
        model: Any,
        processor: Any,
        analysis_result: str,
        duration_seconds: Optional[float],
        style_text: Optional[str]
    ) -> str:
        """Generate SRT subtitle from analysis result.
        
        Args:
            model: Qwen3-VL model
            processor: Model processor
            analysis_result: Video analysis result
            duration_seconds: Video duration in seconds
            style_text: Optional style reference text
            
        Returns:
            Normalized SRT subtitle text
        """
        duration_text = format_duration(duration_seconds)
        style_instruction = self.build_style_instruction(style_text)
        
        subtitle_prompt = self.build_subtitle_prompt(
            analysis_result,
            duration_text,
            style_instruction,
        )
        
        # Generate subtitle
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": subtitle_prompt}],
            }
        ]
        
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        inputs = processor(text=text, return_tensors="pt")
        inputs = inputs.to(model.device)
        
        max_tokens = self.calculate_max_tokens(duration_seconds)
        
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=self.config.do_sample,
            )
        
        subtitle_result = model_manager.decode_generation(processor, inputs, generated_ids)
        return self.normalize_srt(subtitle_result)
    
    def adjust_srt_timing(self, srt_text: str, offset_seconds: float) -> str:
        """Adjust SRT timing by offset.
        
        Args:
            srt_text: SRT text
            offset_seconds: Time offset in seconds
            
        Returns:
            Adjusted SRT text
        """
        if offset_seconds == 0:
            return srt_text
        
        adjusted_lines = []
        for line in srt_text.split('\n'):
            if '-->' in line:
                start, end = line.split(' --> ')
                start_sec = srt_time_to_seconds(start) + offset_seconds
                end_sec = srt_time_to_seconds(end) + offset_seconds
                
                from utils.formatters import format_srt_time
                adjusted_lines.append(
                    f"{format_srt_time(start_sec)} --> {format_srt_time(end_sec)}"
                )
            else:
                adjusted_lines.append(line)
        
        return '\n'.join(adjusted_lines)


# Global subtitle generator instance
subtitle_generator = SubtitleGenerator()
