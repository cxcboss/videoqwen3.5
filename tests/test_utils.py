"""Tests for utility functions."""

import pytest
from utils.file_utils import temp_video_dir, normalize_video_path
from utils.formatters import format_srt_time, format_duration, format_elapsed, srt_time_to_seconds


class TestTempVideoDir:
    """Tests for temp_video_dir context manager."""
    
    def test_creates_temp_directory(self):
        """Test that context manager creates temporary directory."""
        with temp_video_dir() as tmp_dir:
            assert tmp_dir.exists()
            assert tmp_dir.is_dir()
    
    def test_cleans_up_after_use(self):
        """Test that temporary directory is cleaned up."""
        tmp_path = None
        with temp_video_dir() as tmp_dir:
            tmp_path = tmp_dir
        
        # Directory should be cleaned up after context exits
        assert not tmp_path.exists()
    
    def test_custom_prefix(self):
        """Test custom prefix for temporary directory."""
        with temp_video_dir(prefix="custom_prefix_") as tmp_dir:
            assert "custom_prefix_" in str(tmp_dir)


class TestNormalizeVideoPath:
    """Tests for normalize_video_path function."""
    
    def test_string_path(self):
        """Test string path input."""
        assert normalize_video_path("/path/to/video.mp4") == "/path/to/video.mp4"
    
    def test_dict_with_path(self):
        """Test dict input with path key."""
        video = {"path": "/path/to/video.mp4"}
        assert normalize_video_path(video) == "/path/to/video.mp4"
    
    def test_dict_with_name(self):
        """Test dict input with name key."""
        video = {"name": "/path/to/video.mp4"}
        assert normalize_video_path(video) == "/path/to/video.mp4"
    
    def test_object_with_path(self):
        """Test object with path attribute."""
        class Video:
            path = "/path/to/video.mp4"
        assert normalize_video_path(Video()) == "/path/to/video.mp4"
    
    def test_none_input(self):
        """Test None input."""
        assert normalize_video_path(None) is None
    
    def test_empty_string(self):
        """Test empty string input."""
        assert normalize_video_path("") is None
    
    def test_empty_dict(self):
        """Test empty dict input."""
        assert normalize_video_path({}) is None


class TestFormatSrtTime:
    """Tests for format_srt_time function."""
    
    def test_zero_seconds(self):
        """Test zero seconds."""
        assert format_srt_time(0) == "00:00:00,000"
    
    def test_one_second(self):
        """Test one second."""
        assert format_srt_time(1) == "00:00:01,000"
    
    def test_minutes_and_seconds(self):
        """Test minutes and seconds."""
        assert format_srt_time(61.5) == "00:01:01,500"
    
    def test_hours_minutes_seconds(self):
        """Test hours, minutes, and seconds."""
        assert format_srt_time(3661.123) == "01:01:01,123"
    
    def test_milliseconds_rounding(self):
        """Test milliseconds rounding."""
        assert format_srt_time(1.555) == "00:00:01,555"
        assert format_srt_time(1.556) == "00:00:01,556"


class TestFormatDuration:
    """Tests for format_duration function."""
    
    def test_none_duration(self):
        """Test None duration."""
        result = format_duration(None)
        assert "未知" in result
    
    def test_zero_duration(self):
        """Test zero duration."""
        result = format_duration(0)
        assert "未知" in result
    
    def test_positive_duration(self):
        """Test positive duration."""
        result = format_duration(65.5)
        assert "65.5 秒" in result
        assert "00:01:05,500" in result


class TestFormatElapsed:
    """Tests for format_elapsed function."""
    
    def test_seconds_only(self):
        """Test elapsed time in seconds."""
        result = format_elapsed(30.5)
        assert "识别耗时：30.50 秒" in result
    
    def test_minutes_and_seconds(self):
        """Test elapsed time in minutes and seconds."""
        result = format_elapsed(65.5)
        assert "识别耗时：1 分 5.50 秒" in result
    
    def test_exact_minute(self):
        """Test exact minute."""
        result = format_elapsed(60)
        assert "识别耗时：1 分 0.00 秒" in result


class TestSrtTimeToSeconds:
    """Tests for srt_time_to_seconds function."""
    
    def test_zero_time(self):
        """Test zero time."""
        assert srt_time_to_seconds("00:00:00,000") == 0
    
    def test_one_second(self):
        """Test one second."""
        assert srt_time_to_seconds("00:00:01,000") == 1
    
    def test_minutes_and_seconds(self):
        """Test minutes and seconds."""
        assert srt_time_to_seconds("00:01:01,500") == 61.5
    
    def test_hours_minutes_seconds(self):
        """Test hours, minutes, and seconds."""
        assert srt_time_to_seconds("01:01:01,123") == 3661.123
