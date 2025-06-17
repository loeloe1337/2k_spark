"""
Refresh status service for tracking and reporting refresh progress.
"""

import json
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

class RefreshStatusService:
    """
    Service for tracking and reporting refresh progress in real-time.
    """
    
    def __init__(self):
        """Initialize the refresh status service."""
        self._status_lock = threading.Lock()
        self._current_status = {
            "status": "idle",  # idle, running, completed, failed
            "stage": "",
            "progress": 0,  # 0-100
            "message": "",
            "start_time": None,
            "end_time": None,
            "error": None,
            "stages_completed": [],
            "current_stage_start": None
        }
        
        # Define the stages of the refresh process
        self._stages = [
            {"id": "auth", "name": "Authenticating", "weight": 5},
            {"id": "player_stats", "name": "Loading player statistics", "weight": 15},
            {"id": "upcoming_matches", "name": "Loading upcoming matches", "weight": 15},
            {"id": "winner_models", "name": "Loading winner prediction models", "weight": 10},
            {"id": "score_models", "name": "Loading score prediction models", "weight": 10},
            {"id": "generating_predictions", "name": "Generating match predictions", "weight": 30},
            {"id": "saving_results", "name": "Saving prediction results", "weight": 10},
            {"id": "validation", "name": "Validating predictions", "weight": 5}
        ]
        
        # Calculate cumulative progress weights
        total_weight = sum(stage["weight"] for stage in self._stages)
        cumulative = 0
        for stage in self._stages:
            stage["start_progress"] = cumulative
            cumulative += stage["weight"]
            stage["end_progress"] = cumulative
            stage["progress_range"] = stage["weight"]
        
        # Normalize to 0-100
        for stage in self._stages:
            stage["start_progress"] = (stage["start_progress"] / total_weight) * 100
            stage["end_progress"] = (stage["end_progress"] / total_weight) * 100
    
    def start_refresh(self) -> None:
        """Start a new refresh process."""
        with self._status_lock:
            self._current_status = {
                "status": "running",
                "stage": "",
                "progress": 0,
                "message": "Starting refresh process...",
                "start_time": datetime.now(timezone.utc).isoformat(),
                "end_time": None,
                "error": None,
                "stages_completed": [],
                "current_stage_start": datetime.now(timezone.utc).isoformat()
            }
    
    def update_stage(self, stage_id: str, message: str = None, sub_progress: float = 0) -> None:
        """
        Update the current stage and progress.
        
        Args:
            stage_id: ID of the current stage
            message: Optional custom message
            sub_progress: Progress within the current stage (0-100)
        """
        with self._status_lock:
            # Find the stage
            stage_info = next((s for s in self._stages if s["id"] == stage_id), None)
            if not stage_info:
                return
            
            # Calculate overall progress
            stage_progress = min(max(sub_progress, 0), 100) / 100  # Normalize to 0-1
            overall_progress = stage_info["start_progress"] + (stage_info["progress_range"] * stage_progress)
            
            # Update status
            self._current_status.update({
                "stage": stage_info["name"],
                "progress": round(overall_progress, 1),
                "message": message or stage_info["name"],
                "current_stage_start": datetime.now(timezone.utc).isoformat()
            })
    
    def complete_stage(self, stage_id: str) -> None:
        """Mark a stage as completed."""
        with self._status_lock:
            if stage_id not in self._current_status["stages_completed"]:
                self._current_status["stages_completed"].append(stage_id)
            
            # Find the stage and set progress to its end
            stage_info = next((s for s in self._stages if s["id"] == stage_id), None)
            if stage_info:
                self._current_status["progress"] = stage_info["end_progress"]
    
    def complete_refresh(self, success: bool = True, error: str = None) -> None:
        """Complete the refresh process."""
        with self._status_lock:
            self._current_status.update({
                "status": "completed" if success else "failed",
                "progress": 100 if success else self._current_status["progress"],
                "message": "Refresh completed successfully!" if success else f"Refresh failed: {error}",
                "end_time": datetime.now(timezone.utc).isoformat(),
                "error": error if not success else None
            })
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current refresh status."""
        with self._status_lock:
            # Calculate duration if refresh is running or completed
            status = self._current_status.copy()
            
            if status["start_time"]:
                start_time = datetime.fromisoformat(status["start_time"])
                end_time = datetime.fromisoformat(status["end_time"]) if status["end_time"] else datetime.now(timezone.utc)
                duration = (end_time - start_time).total_seconds()
                status["duration_seconds"] = round(duration, 1)
            
            return status
    
    def reset(self) -> None:
        """Reset the status to idle."""
        with self._status_lock:
            self._current_status = {
                "status": "idle",
                "stage": "",
                "progress": 0,
                "message": "",
                "start_time": None,
                "end_time": None,
                "error": None,
                "stages_completed": [],
                "current_stage_start": None
            }


# Global instance
_refresh_status_service = RefreshStatusService()

def get_refresh_status_service() -> RefreshStatusService:
    """Get the global refresh status service instance."""
    return _refresh_status_service
