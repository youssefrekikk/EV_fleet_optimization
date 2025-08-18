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
        self.fix_stuck_tasks()
    
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
            if task_info.get("status") in ["completed", "failed","stopped"]:
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
    
    def fix_stuck_tasks(self):
        """Mark previously running tasks as stopped if app restarted."""
        stuck_states = ["running", "processing"]
        changed = False
        for task_id, task_info in self.tasks.items():
            if task_info.get("status") in stuck_states:
                task_info["status"] = "stopped"
                task_info["finished_at"] = datetime.now().isoformat()
                task_info["message"] = "Task stopped unexpectedly (app closed)"
                changed = True
        if changed:
            self._save_status()
    
    def _process_tasks(self):
        """Background worker to process tasks."""
        while True:
            try:
                task_id, task_info = self.task_queue.get(timeout=1)
                
                # Update status to show it's being processed
                self.update_task(task_id, status="processing", message="Task queued for processing")
                
                # Execute the actual task based on type
                task_type = task_info.get("type")
                
                if task_type == "data_generation":
                    self._execute_data_generation_task(task_id, task_info)
                elif task_type == "network_build":
                    self._execute_network_build_task(task_id, task_info)
                elif task_type == "optimization":
                    self._execute_optimization_task(task_id, task_info)
                else:
                    # Don't mark as completed here - let the actual ML service do it
                    self.update_task(task_id, status="processing", message="Task queued for processing")
                
            except queue.Empty:
                continue
            except Exception as e:
                if 'task_id' in locals():
                    self.complete_task(task_id, error=str(e))
    
    def _execute_data_generation_task(self, task_id: str, task_info: Dict[str, Any]):
        """Execute data generation task."""
        try:
            from .generation_service import generate_synthetic_data
            
            params = task_info.get("params", {})
            self.update_task(task_id, progress=10, message="Starting data generation...")
            
            # Extract parameters
            fleet_size = params.get("fleet_size", 50)
            simulation_days = params.get("simulation_days", 7)
            start_date = params.get("start_date", "2024-01-01")
            ev_models_market_share = params.get("ev_models_market_share")
            driver_profiles_proportion = params.get("driver_profiles_proportion")
            
            self.update_task(task_id, progress=20, message="Initializing generator...")
            
            # Call the actual data generation function
            result = generate_synthetic_data(
                fleet_size=fleet_size,
                simulation_days=simulation_days,
                start_date=start_date,
                ev_models_market_share=ev_models_market_share,
                driver_profiles_proportion=driver_profiles_proportion
            )
            
            self.update_task(task_id, progress=100, message="Data generation completed!")
            self.complete_task(task_id, result=result)
            
        except Exception as e:
            self.complete_task(task_id, error=str(e))
    
    def _execute_network_build_task(self, task_id: str, task_info: Dict[str, Any]):
        """Execute network build task."""
        try:
            from ..services.network_service import build_or_load_network
            
            self.update_task(task_id, progress=10, message="Starting network build...")
            
            # Call the network build function
            result = build_or_load_network()
            
            self.update_task(task_id, progress=100, message="Network build completed!")
            self.complete_task(task_id, result=result)
            
        except Exception as e:
            self.complete_task(task_id, error=str(e))
    
    def _execute_optimization_task(self, task_id: str, task_info: Dict[str, Any]):
        """Execute optimization task."""
        try:
            from .opt_service import run_optimization
            
            params = task_info.get("params", {})
            self.update_task(task_id, progress=10, message="Starting optimization...")
            
            # Extract optimization parameters
            routes_csv = params.get("routes_csv")
            fleet_csv = params.get("fleet_csv") 
            weather_csv = params.get("weather_csv")
            
            # Build optimization config from individual parameters
            optimization_config = {
                "algorithm": params.get("algorithm", "dijkstra"),
                "gamma_time_weight": params.get("gamma_time_weight", 0.5),
                "price_weight": params.get("price_weight", 1.0),
                "battery_buffer": params.get("battery_buffer", 0.15),
                "max_detour": params.get("max_detour", 5.0),
                "planning_mode": params.get("planning_mode", "single_trip"),
                "soc_objective": params.get("soc_objective", "balanced"),
                "reserve_soc": params.get("reserve_soc", 0.2),
                "horizon_trips": params.get("horizon_trips", 1),
                "eval_max_days": params.get("eval_max_days", 7),
                "trip_sample_frac": params.get("trip_sample_frac", 1.0)
            }
            
            self.update_task(task_id, progress=20, message="Loading data and running optimization...")
            
            # Call the actual optimization function with correct parameters
            result = run_optimization(
                routes_csv=routes_csv,
                fleet_csv=fleet_csv,
                weather_csv=weather_csv,
                date="2024-01-01",  # Default date, could be made configurable
                algorithm=optimization_config.get("algorithm", "dijkstra"),
                soc_planning=True  # Enable SOC planning
            )
            
            self.update_task(task_id, progress=100, message="Optimization completed!")
            self.complete_task(task_id, result=result)
            
        except Exception as e:
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

