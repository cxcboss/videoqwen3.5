"""History management module for storing analysis records."""

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from config import get_config


class HistoryManager:
    """Manages analysis history records."""
    
    def __init__(self, history_file: Optional[str] = None):
        """Initialize history manager.
        
        Args:
            history_file: Path to history JSON file
        """
        if history_file is None:
            config = get_config()
            history_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..",
                "history.json"
            )
        
        self._history_file = Path(history_file)
        self._lock = threading.Lock()
        self._max_records = 100  # Keep last 100 records
        
        # Ensure directory exists
        self._history_file.parent.mkdir(parents=True, exist_ok=True)
    
    def _load_history(self) -> list[dict[str, Any]]:
        """Load history from file.
        
        Returns:
            List of history records
        """
        if not self._history_file.exists():
            return []
        
        try:
            with open(self._history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    
    def _save_history(self, history: list[dict[str, Any]]) -> None:
        """Save history to file.
        
        Args:
            history: List of history records
        """
        # Keep only the last N records
        if len(history) > self._max_records:
            history = history[-self._max_records:]
        
        with open(self._history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    
    def add_record(
        self,
        video_name: str,
        model_size: str,
        srt_content: str,
        elapsed_time: float,
        duration_seconds: Optional[float] = None,
    ) -> None:
        """Add a new analysis record.
        
        Args:
            video_name: Name of the video file
            model_size: Model size used (2B, 4B, 8B)
            srt_content: Generated SRT content
            elapsed_time: Processing time in seconds
            duration_seconds: Video duration in seconds
        """
        record = {
            "id": datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
            "timestamp": datetime.now().isoformat(),
            "video_name": video_name,
            "model_size": model_size,
            "elapsed_time": round(elapsed_time, 2),
            "duration_seconds": round(duration_seconds, 2) if duration_seconds else None,
            "srt_preview": srt_content[:200] + "..." if len(srt_content) > 200 else srt_content,
            "srt_length": len(srt_content),
        }
        
        with self._lock:
            history = self._load_history()
            history.append(record)
            self._save_history(history)
    
    def get_history(self, limit: Optional[int] = None) -> list[dict[str, Any]]:
        """Get analysis history.
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List of history records
        """
        with self._lock:
            history = self._load_history()
            
            if limit:
                history = history[-limit:]
            
            return history
    
    def get_history_json(self) -> str:
        """Get history as JSON string.
        
        Returns:
            JSON string of history
        """
        history = self.get_history(limit=20)
        return json.dumps(history, ensure_ascii=False, indent=2)
    
    def get_history_display(self) -> str:
        """Get history as formatted display text.
        
        Returns:
            Formatted history string
        """
        history = self.get_history(limit=20)
        
        if not history:
            return "暂无分析记录。"
        
        lines = []
        for i, record in enumerate(reversed(history), 1):
            video = record.get("video_name", "未知")
            model = record.get("model_size", "?")
            elapsed = record.get("elapsed_time", 0)
            duration = record.get("duration_seconds")
            ts = record.get("timestamp", "")
            preview = record.get("srt_preview", "")
            
            duration_str = f"{duration:.1f}s" if duration else "未知"
            ts_short = ts[:19].replace("T", " ") if ts else ""
            
            lines.append(
                f"**{i}. {video}**\n"
                f"模型: {model} | 视频时长: {duration_str} | 耗时: {elapsed:.1f}s | {ts_short}\n"
                f"预览: `{preview[:80]}...`\n"
            )
        
        return "\n---\n".join(lines)
    
    def clear_history(self) -> None:
        """Clear all history records."""
        with self._lock:
            self._save_history([])
    
    def get_statistics(self) -> dict[str, Any]:
        """Get history statistics.
        
        Returns:
            Statistics dictionary
        """
        with self._lock:
            history = self._load_history()
            
            if not history:
                return {
                    "total_analyses": 0,
                    "total_elapsed_time": 0,
                    "average_elapsed_time": 0,
                    "models_used": {},
                }
            
            total_elapsed = sum(r.get("elapsed_time", 0) for r in history)
            models_used = {}
            for r in history:
                model = r.get("model_size", "unknown")
                models_used[model] = models_used.get(model, 0) + 1
            
            return {
                "total_analyses": len(history),
                "total_elapsed_time": round(total_elapsed, 2),
                "average_elapsed_time": round(total_elapsed / len(history), 2),
                "models_used": models_used,
            }


# Global history manager instance
history_manager = HistoryManager()
