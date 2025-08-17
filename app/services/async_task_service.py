import asyncio
import threading
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Coroutine
import queue
import concurrent.futures
from dataclasses import dataclass, asdict
from enum import Enum

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class TaskProgress:
    current: int
    total: int
    percentage: float
    message: str
    eta_seconds: Optional[float] = None

@dataclass
class TaskResult:
    task_id: str
    status: TaskStatus
    progress: TaskProgress
    started_at: datetime
    result: Optional[Any] = None
    error: Optional[str] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None

class AsyncTaskService:
    """High-performance async task service with progress tracking."""
    
    def __init__(self, max_workers: int = 4, status_file: str = "data/analysis_results/async_tasks.json"):
        self.max_workers = max_workers
        self.status_file = Path(status_file)
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.tasks: Dict[str, TaskResult] = {}
        self.task_queue = asyncio.Queue()
        self.running_tasks = set()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        
        self._load_status()
        self._start_worker()
    
    def _load_status(self):
        """Load task status from disk."""
        if self.status_file.exists():
            try:
                with open(self.status_file, 'r') as f:
                    data = json.load(f)
                    for task_id, task_data in data.items():
                        if task_data.get('status') == TaskStatus.RUNNING.value:
                            # Reset running tasks to pending
                            task_data['status'] = TaskStatus.PENDING.value
                        
                        # Convert string status back to enum
                        if 'status' in task_data:
                            task_data['status'] = TaskStatus(task_data['status'])
                        
                        # Convert string datetime back to datetime object
                        if 'started_at' in task_data:
                            task_data['started_at'] = datetime.fromisoformat(task_data['started_at'])
                        
                        if 'completed_at' in task_data and task_data['completed_at']:
                            task_data['completed_at'] = datetime.fromisoformat(task_data['completed_at'])
                        
                        # Convert progress dict back to TaskProgress
                        if 'progress' in task_data:
                            progress_data = task_data['progress']
                            task_data['progress'] = TaskProgress(**progress_data)
                        
                        self.tasks[task_id] = TaskResult(**task_data)
            except Exception as e:
                print(f"Failed to load task status: {e}")
    
    def _save_status(self):
        """Save task status to disk."""
        try:
            with open(self.status_file, 'w') as f:
                json.dump({k: asdict(v) for k, v in self.tasks.items()}, f, indent=2, default=str)
        except Exception as e:
            print(f"Failed to save task status: {e}")
    
    def _start_worker(self):
        """Start the background task worker."""
        def worker():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._process_tasks())
        
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
    
    async def _process_tasks(self):
        """Process tasks from the queue."""
        while True:
            try:
                task_id, task_func, task_args, task_kwargs = await self.task_queue.get()
                
                if task_id in self.running_tasks:
                    continue
                
                self.running_tasks.add(task_id)
                
                # Update task status
                self.tasks[task_id].status = TaskStatus.RUNNING
                self.tasks[task_id].started_at = datetime.now()
                self._save_status()
                
                try:
                    # Run task in thread pool
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        self.executor, 
                        self._run_task_with_progress, 
                        task_id, task_func, task_args, task_kwargs
                    )
                    
                    # Mark as completed
                    self.tasks[task_id].status = TaskStatus.COMPLETED
                    self.tasks[task_id].result = result
                    self.tasks[task_id].completed_at = datetime.now()
                    self.tasks[task_id].progress = TaskProgress(100, 100, 100.0, "Completed")
                    
                except Exception as e:
                    # Mark as failed
                    self.tasks[task_id].status = TaskStatus.FAILED
                    self.tasks[task_id].error = str(e)
                    self.tasks[task_id].completed_at = datetime.now()
                    self.tasks[task_id].progress = TaskProgress(0, 100, 0.0, f"Failed: {str(e)}")
                
                finally:
                    self.running_tasks.discard(task_id)
                    self._save_status()
                
            except Exception as e:
                print(f"Error in task processor: {e}")
                await asyncio.sleep(1)
    
    def _run_task_with_progress(self, task_id: str, task_func: Callable, args: tuple, kwargs: dict) -> Any:
        """Run a task with progress tracking."""
        try:
            # Create progress callback
            def progress_callback(current: int, total: int, message: str = ""):
                percentage = (current / total * 100) if total > 0 else 0
                self.tasks[task_id].progress = TaskProgress(current, total, percentage, message)
                self._save_status()
            
            # Add progress callback to kwargs
            kwargs['progress_callback'] = progress_callback
            
            # Run the task
            return task_func(*args, **kwargs)
            
        except Exception as e:
            raise e
    
    async def submit_task(self, task_id: str, task_func: Callable, *args, **kwargs) -> str:
        """Submit a task for execution."""
        # Create task result
        if task_id in self.tasks:
            print(f"Task with ID {task_id} already exists. Not submitting a new task.")
            return task_id
        self.tasks[task_id] = TaskResult(
            task_id=task_id,
            status=TaskStatus.PENDING,
            progress=TaskProgress(0, 100, 0.0, "Queued"),
            started_at=datetime.now(),
            metadata=kwargs.get('metadata', {})
        )
        
        # Add to queue
        await self.task_queue.put((task_id, task_func, args, kwargs))
        self._save_status()
        
        return task_id
    
    def get_task_status(self, task_id: str) -> Optional[TaskResult]:
        """Get task status."""
        return self.tasks.get(task_id)
    
    def get_all_tasks(self) -> Dict[str, TaskResult]:
        """Get all tasks."""
        return self.tasks.copy()
    
    def get_running_tasks(self) -> Dict[str, TaskResult]:
        """Get running tasks."""
        return {k: v for k, v in self.tasks.items() if v.status == TaskStatus.RUNNING}
    
    def get_pending_tasks(self) -> Dict[str, TaskResult]:
        """Get pending tasks."""
        return {k: v for k, v in self.tasks.items() if v.status == TaskStatus.PENDING}
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task."""
        if task_id in self.tasks:
            if self.tasks[task_id].status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
                self.tasks[task_id].status = TaskStatus.CANCELLED
                self.tasks[task_id].completed_at = datetime.now()
                self.running_tasks.discard(task_id)
                self._save_status()
                return True
        return False
    
    def clear_completed_tasks(self, max_age_hours: int = 24):
        """Clear old completed tasks."""
        cutoff = datetime.now().timestamp() - (max_age_hours * 3600)
        
        to_remove = []
        for task_id, task_result in self.tasks.items():
            if task_result.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                if task_result.status == TaskStatus.CANCELLED:
                    to_remove.append(task_id)
                elif task_result.completed_at:
                    if task_result.completed_at.timestamp() < cutoff:
                        to_remove.append(task_id)
        
        for task_id in to_remove:
            del self.tasks[task_id]
        
        if to_remove:
            self._save_status()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get task service statistics."""
        status_counts = {}
        for status in TaskStatus:
            status_counts[status.value] = sum(1 for t in self.tasks.values() if t.status == status)
        
        return {
            "total_tasks": len(self.tasks),
            "running_tasks": len(self.running_tasks),
            "status_counts": status_counts,
            "max_workers": self.max_workers,
            "queue_size": self.task_queue.qsize() if hasattr(self.task_queue, 'qsize') else 0
        }

# Global async task service instance
async_task_service = AsyncTaskService()

# Convenience functions for common task types
async def submit_data_generation_task(fleet_size: int, simulation_days: int, **kwargs) -> str:
    """Submit a data generation task."""
    from .generation_service import generate_synthetic_data
    
    task_id = f"data_generation_{int(time.time())}"
    
    def generation_worker(fleet_size, simulation_days, progress_callback=None, **kwargs):
        # Simulate progress updates
        if progress_callback:
            progress_callback(0, 100, "Initializing generator...")
        
        # This would be the actual generation logic
        # For now, just simulate
        for i in range(10):
            if progress_callback:
                progress_callback(i * 10, 100, f"Generating data step {i+1}/10")
            time.sleep(0.5)  # Simulate work
        
        return {"fleet_size": fleet_size, "simulation_days": simulation_days}
    
    await async_task_service.submit_task(
        task_id, 
        generation_worker, 
        fleet_size, 
        simulation_days, 
        metadata={"type": "data_generation", **kwargs}
    )
    
    return task_id

async def submit_model_training_task(data_dir: str, model_choice: str = "random_forest", **kwargs) -> str:
    """Submit a model training task."""
    from .ml_service import train_model
    
    task_id = f"model_training_{int(time.time())}"
    
    def training_worker(data_dir, model_choice, progress_callback=None, **kwargs):
        # Simulate progress updates
        if progress_callback:
            progress_callback(0, 100, "Loading data...")
        
        # This would be the actual training logic
        # For now, just simulate
        for i in range(10):
            if progress_callback:
                progress_callback(i * 10, 100, f"Training step {i+1}/10")
            time.sleep(1)  # Simulate work
        
        return {"data_dir": data_dir, "model_choice": model_choice}
    
    await async_task_service.submit_task(
        task_id, 
        training_worker, 
        data_dir, 
        model_choice, 
        metadata={"type": "model_training", **kwargs}
    )
    
    return task_id
