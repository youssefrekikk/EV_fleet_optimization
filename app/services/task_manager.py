from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import threading
import queue

class TaskManager:
    """Manages long-running tasks with persistent status tracking."""
    
    def __init__(self, status_file: str = "data/analysis_results/task_status.json"):
        self.status_file = Path(status_file)
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        self.tasks = {}
        self.task_queue = queue.Queue()
        self.worker_thread = None
        self._load_status()
    
    def _load_status(self):
        """Load existing task status from file."""
        if self.status_file.exists():
            try:
                with open(self.status_file, 'r') as f:
                    self.tasks = json.load(f)
            except Exception:
                self.tasks = {}
        else:
            self.tasks = {}
    
    def _save_status(self):
        """Save current task status to file."""
        try:
            with open(self.status_file, 'w') as f:
                json.dump(self.tasks, f, indent=2, default=str)
        except Exception as e:
            print(f"Failed to save task status: {e}")
    
    def start_task(self, task_id: str, task_type: str, params: Dict[str, Any] = None) -> str:
        """Start a new task and return its ID."""
        task_info = {
            "id": task_id,
            "type": task_type,
            "params": params or {},
            "status": "running",
            "started_at": datetime.now().isoformat(),
            "progress": 0,
            "message": "Task started",
            "result": None,
            "error": None
        }
        
        self.tasks[task_id] = task_info
        self._save_status()
        
        # Add to processing queue
        self.task_queue.put((task_id, task_info))
        
        # Start worker thread if not running
        if not self.worker_thread or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(target=self._process_tasks, daemon=True)
            self.worker_thread.start()
        
        return task_id
    
    def update_task(self, task_id: str, **updates):
        """Update task status and progress."""
        if task_id in self.tasks:
            self.tasks[task_id].update(updates)
            self._save_status()
    
    def complete_task(self, task_id: str, result: Any = None, error: str = None):
        """Mark task as completed with result or error."""
        if task_id in self.tasks:
            self.tasks[task_id].update({
                "status": "completed" if error is None else "failed",
                "finished_at": datetime.now().isoformat(),
                "progress": 100,
                "result": result,
                "error": error
            })
            self._save_status()
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a task."""
        return self.tasks.get(task_id)
    
    def get_all_tasks(self) -> Dict[str, Dict[str, Any]]:
        """Get all tasks."""
        return self.tasks.copy()
    
    def get_running_tasks(self) -> Dict[str, Dict[str, Any]]:
        """Get only running tasks."""
        return {k: v for k, v in self.tasks.items() if v.get("status") == "running"}
    
    def clear_completed_tasks(self, max_age_hours: int = 24):
        """Clear old completed tasks to keep status file manageable."""
        cutoff = datetime.now().timestamp() - (max_age_hours * 3600)
        
        to_remove = []
        for task_id, task_info in self.tasks.items():
            if task_info.get("status") in ["completed", "failed"]:
                finished_at = task_info.get("finished_at")
                if finished_at:
                    try:
                        task_time = datetime.fromisoformat(finished_at).timestamp()
                        if task_time < cutoff:
                            to_remove.append(task_id)
                    except:
                        pass
        
        for task_id in to_remove:
            del self.tasks[task_id]
        
        if to_remove:
            self._save_status()
    
    def _process_tasks(self):
        """Background worker to process tasks."""
        while True:
            try:
                task_id, task_info = self.task_queue.get(timeout=1)
                # Don't mark as completed here - let the actual ML service do it
                # Just update status to show it's being processed
                self.update_task(task_id, status="processing", message="Task queued for processing")
                
            except queue.Empty:
                continue
            except Exception as e:
                if 'task_id' in locals():
                    self.complete_task(task_id, error=str(e))

# Global task manager instance
task_manager = TaskManager()

def start_model_training(data_dir: str, model_choice: str = "random_forest") -> str:
    """Start a model training task."""
    task_id = f"model_training_{int(time.time())}"
    task_manager.start_task(
        task_id=task_id,
        task_type="model_training",
        params={
            "data_dir": data_dir,
            "model_choice": model_choice
        }
    )
    return task_id

def get_training_status(task_id: str) -> Optional[Dict[str, Any]]:
    """Get status of a model training task."""
    return task_manager.get_task_status(task_id)

def get_all_training_tasks() -> Dict[str, Dict[str, Any]]:
    """Get all model training tasks."""
    all_tasks = task_manager.get_all_tasks()
    return {k: v for k, v in all_tasks.items() if v.get("type") == "model_training"}

