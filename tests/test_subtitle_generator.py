"""Tests for subtitle generator."""

import pytest
from core.subtitle_generator import SubtitleGenerator


class TestNormalizeSrt:
    """Tests for normalize_srt method."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = SubtitleGenerator()
    
    def test_basic_srt(self):
        """Test basic SRT text normalization."""
        input_text = """1
00:00:01,000 --> 00:00:03,000
测试字幕"""
        result = self.generator.normalize_srt(input_text)
        assert "1" in result
        assert "00:00:01,000 --> 00:00:03,000" in result
        assert "测试字幕" in result
    
    def test_markdown_code_block(self):
        """Test SRT text wrapped in markdown code block."""
        input_text = "```srt\n1\n00:00:01,000 --> 00:00:03,000\n测试字幕\n```"
        result = self.generator.normalize_srt(input_text)
        assert "```" not in result
        assert "1" in result
    
    def test_combined_number_and_timestamp(self):
        """Test combined number and timestamp on same line."""
        input_text = "1 00:00:01,000 --> 00:00:03,000"
        result = self.generator.normalize_srt(input_text)
        assert "1" in result
        assert "00:00:01,000 --> 00:00:03,000" in result
    
    def test_number_with_dot(self):
        """Test number with dot."""
        input_text = "1.\n00:00:01,000 --> 00:00:03,000\n测试字幕"
        result = self.generator.normalize_srt(input_text)
        assert "1" in result
    
    def test_number_with_parenthesis(self):
        """Test number with parenthesis."""
        input_text = "1)\n00:00:01,000 --> 00:00:03,000\n测试字幕"
        result = self.generator.normalize_srt(input_text)
        assert "1" in result
    
    def test_empty_input(self):
        """Test empty input."""
        assert self.generator.normalize_srt("") == ""
        assert self.generator.normalize_srt("   ") == ""
    
    def test_multiple_subtitles(self):
        """Test multiple subtitles."""
        input_text = """1
00:00:01,000 --> 00:00:03,000
第一行字幕

2
00:00:03,000 --> 00:00:06,000
第二行字幕"""
        result = self.generator.normalize_srt(input_text)
        assert "1" in result
        assert "2" in result
        assert "第一行字幕" in result
        assert "第二行字幕" in result


class TestBuildStyleInstruction:
    """Tests for build_style_instruction method."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = SubtitleGenerator()
    
    def test_none_style(self):
        """Test None style text."""
        result = self.generator.build_style_instruction(None)
        assert "未提供风格样本文案" in result
    
    def test_empty_style(self):
        """Test empty style text."""
        result = self.generator.build_style_instruction("")
        assert "未提供风格样本文案" in result
    
    def test_whitespace_style(self):
        """Test whitespace-only style text."""
        result = self.generator.build_style_instruction("   ")
        assert "未提供风格样本文案" in result
    
    def test_valid_style(self):
        """Test valid style text."""
        style = "这是一段风格参考文案"
        result = self.generator.build_style_instruction(style)
        assert style in result
        assert "参考文案" in result


class TestCalculateMaxTokens:
    """Tests for calculate_max_tokens method."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = SubtitleGenerator()
    
    def test_none_duration(self):
        """Test with None duration."""
        result = self.generator.calculate_max_tokens(None)
        assert result > 0
    
    def test_short_video(self):
        """Test with short video duration."""
        result = self.generator.calculate_max_tokens(5.0)
        assert result > 0
    
    def test_long_video(self):
        """Test with long video duration."""
        result = self.generator.calculate_max_tokens(120.0)
        assert result > 0
    
    def test_increasing_duration(self):
        """Test that max tokens increase with duration."""
        result_short = self.generator.calculate_max_tokens(10.0)
        result_long = self.generator.calculate_max_tokens(60.0)
        assert result_long >= result_short


class TestAdjustSrtTiming:
    """Tests for adjust_srt_timing method."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = SubtitleGenerator()
    
    def test_zero_offset(self):
        """Test with zero offset."""
        srt_text = "1\n00:00:01,000 --> 00:00:03,000\n测试字幕"
        result = self.generator.adjust_srt_timing(srt_text, 0)
        assert result == srt_text
    
    def test_positive_offset(self):
        """Test with positive offset."""
        srt_text = "1\n00:00:01,000 --> 00:00:03,000\n测试字幕"
        result = self.generator.adjust_srt_timing(srt_text, 1.0)
        assert "00:00:02,000" in result
        assert "00:00:04,000" in result
    
    def test_negative_offset(self):
        """Test with negative offset."""
        srt_text = "1\n00:00:02,000 --> 00:00:04,000\n测试字幕"
        result = self.generator.adjust_srt_timing(srt_text, -1.0)
        assert "00:00:01,000" in result
        assert "00:00:03,000" in result
