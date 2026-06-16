"""Tests for history manager."""

import json
import tempfile
import os
import pytest
from core.history_manager import HistoryManager


class TestHistoryManager:
    """Tests for HistoryManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Use temporary file for testing
        self.tmp_file = tempfile.NamedTemporaryFile(
            suffix='.json',
            delete=False
        )
        self.tmp_file.close()
        self.manager = HistoryManager(self.tmp_file.name)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.tmp_file.name):
            os.remove(self.tmp_file.name)
    
    def test_initial_empty_history(self):
        """Test that history is empty initially."""
        history = self.manager.get_history()
        assert history == []
    
    def test_add_record(self):
        """Test adding a record."""
        self.manager.add_record(
            video_name="test.mp4",
            model_size="2B",
            srt_content="1\n00:00:01,000 --> 00:00:03,000\n测试字幕",
            elapsed_time=10.5,
            duration_seconds=30.0,
        )
        
        history = self.manager.get_history()
        assert len(history) == 1
        assert history[0]["video_name"] == "test.mp4"
        assert history[0]["model_size"] == "2B"
        assert history[0]["elapsed_time"] == 10.5
    
    def test_add_multiple_records(self):
        """Test adding multiple records."""
        for i in range(3):
            self.manager.add_record(
                video_name=f"test_{i}.mp4",
                model_size="2B",
                srt_content=f"字幕 {i}",
                elapsed_time=float(i),
            )
        
        history = self.manager.get_history()
        assert len(history) == 3
    
    def test_get_history_with_limit(self):
        """Test getting history with limit."""
        for i in range(5):
            self.manager.add_record(
                video_name=f"test_{i}.mp4",
                model_size="2B",
                srt_content=f"字幕 {i}",
                elapsed_time=float(i),
            )
        
        history = self.manager.get_history(limit=3)
        assert len(history) == 3
        # Should get the last 3 records
        assert history[0]["video_name"] == "test_2.mp4"
        assert history[2]["video_name"] == "test_4.mp4"
    
    def test_get_history_json(self):
        """Test getting history as JSON."""
        self.manager.add_record(
            video_name="test.mp4",
            model_size="2B",
            srt_content="测试字幕",
            elapsed_time=10.0,
        )
        
        json_str = self.manager.get_history_json()
        data = json.loads(json_str)
        assert len(data) == 1
        assert data[0]["video_name"] == "test.mp4"
    
    def test_clear_history(self):
        """Test clearing history."""
        self.manager.add_record(
            video_name="test.mp4",
            model_size="2B",
            srt_content="测试字幕",
            elapsed_time=10.0,
        )
        
        self.manager.clear_history()
        history = self.manager.get_history()
        assert history == []
    
    def test_get_statistics(self):
        """Test getting statistics."""
        # Empty history
        stats = self.manager.get_statistics()
        assert stats["total_analyses"] == 0
        
        # Add some records
        self.manager.add_record(
            video_name="test1.mp4",
            model_size="2B",
            srt_content="字幕1",
            elapsed_time=10.0,
        )
        self.manager.add_record(
            video_name="test2.mp4",
            model_size="4B",
            srt_content="字幕2",
            elapsed_time=20.0,
        )
        
        stats = self.manager.get_statistics()
        assert stats["total_analyses"] == 2
        assert stats["total_elapsed_time"] == 30.0
        assert stats["average_elapsed_time"] == 15.0
        assert stats["models_used"]["2B"] == 1
        assert stats["models_used"]["4B"] == 1
    
    def test_max_records_limit(self):
        """Test that max records limit is enforced."""
        # Set a small limit for testing
        self.manager._max_records = 5
        
        # Add more records than the limit
        for i in range(10):
            self.manager.add_record(
                video_name=f"test_{i}.mp4",
                model_size="2B",
                srt_content=f"字幕 {i}",
                elapsed_time=float(i),
            )
        
        history = self.manager.get_history()
        assert len(history) == 5
        # Should keep the last 5 records
        assert history[0]["video_name"] == "test_5.mp4"
