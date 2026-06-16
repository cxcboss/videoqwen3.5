"""Tests for video splitter module."""

import pytest
from core.video_splitter import merge_srt_segments, _adjust_timestamp


class TestMergeSrtSegments:
    """Tests for merge_srt_segments function."""
    
    def test_single_segment(self):
        """Test merging single segment returns unchanged."""
        srt = "1\n00:00:01,000 --> 00:00:03,000\n测试字幕"
        result = merge_srt_segments([srt], [3.0])
        assert "1" in result
        assert "00:00:01,000 --> 00:00:03,000" in result
    
    def test_two_segments(self):
        """Test merging two segments adjusts timestamps."""
        srt1 = "1\n00:00:01,000 --> 00:00:03,000\n第一段"
        srt2 = "1\n00:00:01,000 --> 00:00:03,000\n第二段"
        result = merge_srt_segments([srt1, srt2], [5.0, 5.0])
        
        # First segment unchanged
        assert "00:00:01,000 --> 00:00:03,000" in result
        # Second segment offset by 5 seconds
        assert "00:00:06,000 --> 00:00:08,000" in result
        # Sequential numbering
        assert "1" in result
        assert "2" in result
        # Both texts present
        assert "第一段" in result
        assert "第二段" in result
    
    def test_empty_segment(self):
        """Test merging with empty segment."""
        srt1 = "1\n00:00:01,000 --> 00:00:03,000\n第一段"
        result = merge_srt_segments([srt1, ""], [5.0, 5.0])
        assert "第一段" in result
    
    def test_multiple_subtitles_per_segment(self):
        """Test merging segments with multiple subtitles."""
        srt1 = "1\n00:00:01,000 --> 00:00:02,000\n字幕A\n\n2\n00:00:03,000 --> 00:00:04,000\n字幕B"
        srt2 = "1\n00:00:01,000 --> 00:00:02,000\n字幕C"
        result = merge_srt_segments([srt1, srt2], [5.0, 5.0])
        
        # Sequential numbering: 1, 2, 3
        lines = result.split('\n')
        numbers = [l.strip() for l in lines if l.strip().isdigit()]
        assert numbers == ["1", "2", "3"]
        
        # All texts present
        assert "字幕A" in result
        assert "字幕B" in result
        assert "字幕C" in result


class TestAdjustTimestamp:
    """Tests for _adjust_timestamp function."""
    
    def test_no_offset(self):
        """Test timestamp with zero offset."""
        ts = "00:00:01,000 --> 00:00:03,000"
        result = _adjust_timestamp(ts, 0)
        assert result == ts
    
    def test_positive_offset(self):
        """Test timestamp with positive offset."""
        ts = "00:00:01,000 --> 00:00:03,000"
        result = _adjust_timestamp(ts, 5.0)
        assert "00:00:06,000" in result
        assert "00:00:08,000" in result
    
    def test_negative_offset(self):
        """Test timestamp with negative offset."""
        ts = "00:00:06,000 --> 00:00:08,000"
        result = _adjust_timestamp(ts, -5.0)
        assert "00:00:01,000" in result
        assert "00:00:03,000" in result
    
    def test_invalid_format(self):
        """Test invalid timestamp returns original."""
        ts = "invalid timestamp"
        result = _adjust_timestamp(ts, 5.0)
        assert result == ts
