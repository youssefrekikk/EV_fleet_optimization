from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional
import threading
import time

import pandas as pd

from src.models.consumption_prediction.consumption_model_v2 import SegmentEnergyPredictor
from .task_manager import task_manager


def train_model(data_dir: str, model_choice: str = "random_forest", task_id: str = None) -> Dict[str, Any]:
    """Train model with progress tracking via task manager."""
    
    def update_progress(progress: int, message: str):
        """Update task progress."""
        if task_id:
            task_manager.update_task(task_id, progress=progress, message=message)
    
    try:
        update_progress(5, "Verifying data directory...")
        
        # Verify data directory exists
        data_path = Path(data_dir)
        if not data_path.exists():
            error_msg = f"Data directory '{data_dir}' does not exist"
            if task_id:
                task_manager.complete_task(task_id, error=error_msg)
            return {
                "error": error_msg,
                "error_type": "FileNotFoundError",
                "data_shape": None,
                "columns": None,
            }
        
        update_progress(10, "Checking required files...")
        
        # Check if required files exist
        required_files = ["segments.csv", "fleet_info.csv", "weather.csv"]
        missing_files = [f for f in required_files if not (data_path / f).exists()]
        if missing_files:
            error_msg = f"Missing required files: {missing_files}"
            if task_id:
                task_manager.complete_task(task_id, error=error_msg)
            return {
                "error": error_msg,
                "data_shape": None,
                "columns": None,
            }
        
        update_progress(15, "Initializing predictor...")
        
        # Initialize predictor (class now handles Path conversion)
        pred = SegmentEnergyPredictor(data_dir=str(data_path))
        
        update_progress(20, "Loading data...")
        
        # Load data with better error handling
        try:
            data = pred.load_data()
        except Exception as load_error:
            error_msg = f"Failed to load data: {str(load_error)}"
            if task_id:
                task_manager.complete_task(task_id, error=error_msg)
            return {
                "error": error_msg,
                "error_type": type(load_error).__name__,
                "data_shape": None,
                "columns": None,
                "load_error_details": str(load_error)
            }
        
        update_progress(30, "Preparing numeric columns...")
        
        # Ensure numeric columns are properly typed
        numeric_cols = ['distance_m', 'energy_kwh', 'start_elevation_m', 'end_elevation_m']
        for col in numeric_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Check for any remaining string columns that should be numeric
        string_numeric_cols = []
        for col in data.columns:
            if data[col].dtype == 'object':
                try:
                    pd.to_numeric(data[col], errors='raise')
                    string_numeric_cols.append(col)
                except:
                    pass
        
        if string_numeric_cols:
            for col in string_numeric_cols:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        update_progress(40, "Engineering features...")
        
        # Engineer features
        try:
            feats = pred.engineer_features(data)
        except Exception as feat_error:
            error_msg = f"Failed to engineer features: {str(feat_error)}"
            if task_id:
                task_manager.complete_task(task_id, error=error_msg)
            return {
                "error": error_msg,
                "error_type": type(feat_error).__name__,
                "data_shape": data.shape if 'data' in locals() else None,
                "columns": list(data.columns) if 'data' in locals() else None,
                "feature_error_details": str(feat_error)
            }
        
        update_progress(50, "Preparing features...")
        
        # Prepare features
        try:
            X, y = pred.prepare_features(feats)
        except Exception as prep_error:
            error_msg = f"Failed to prepare features: {str(prep_error)}"
            if task_id:
                task_manager.complete_task(task_id, error=error_msg)
            return {
                "error": error_msg,
                "error_type": type(prep_error).__name__,
                "data_shape": feats.shape if 'feats' in locals() else None,
                "columns": list(feats.columns) if 'feats' in locals() else None,
                "prepare_error_details": str(prep_error)
            }

        update_progress(60, "Training models...")
        
        # Train models
        try:
            trained_models = {}

            if model_choice == "random_forest":
                model = pred.tune_random_forest(X, y)
                pred.print_model_metrics(model, X, y, "Random Forest")
                trained_models["Random Forest"] = model

            elif model_choice == "xgboost":
                model = pred.tune_xgboost(X, y)
                pred.print_model_metrics(model, X, y, "XGBoost (RandomSearch)")
                trained_models["XGBoost"] = model

            elif model_choice == "xgboost_optuna":
                model = pred.tune_xgboost_optuna(X, y, n_trials=100, timeout=600)
                pred.print_model_metrics(model, X, y, "XGBoost Optuna")
                trained_models["XGBoost Optuna"] = model

            elif model_choice == "gradient_boosting":
                model = pred.tune_gradient_boosting(X, y)
                pred.print_model_metrics(model, X, y, "Gradient Boosting")
                trained_models["Gradient Boosting"] = model

            elif model_choice == "lightgbm":
                model = pred.tune_lightgbm(X, y)
                pred.print_model_metrics(model, X, y, "LightGBM")
                trained_models["LightGBM"] = model

            elif model_choice == "catboost":
                model = pred.tune_catboost(X, y)
                pred.print_model_metrics(model, X, y, "CatBoost")
                trained_models["CatBoost"] = model

            elif model_choice == "all":
                # Run all tuned models
                trained_models["Random Forest"] = pred.tune_random_forest(X, y)
                pred.print_model_metrics(trained_models["Random Forest"], X, y, "Random Forest")

                trained_models["Gradient Boosting"] = pred.tune_gradient_boosting(X, y)
                pred.print_model_metrics(trained_models["Gradient Boosting"], X, y, "Gradient Boosting")

                trained_models["XGBoost (RandomSearch)"] = pred.tune_xgboost(X, y)
                pred.print_model_metrics(trained_models["XGBoost (RandomSearch)"], X, y, "XGBoost (RandomSearch)")

                trained_models["XGBoost Optuna"] = pred.tune_xgboost_optuna(X, y, n_trials=200, timeout=600)
                pred.print_model_metrics(trained_models["XGBoost Optuna"], X, y, "XGBoost Optuna")

                trained_models["LightGBM"] = pred.tune_lightgbm(X, y)
                pred.print_model_metrics(trained_models["LightGBM"], X, y, "LightGBM")

                trained_models["CatBoost"] = pred.tune_catboost(X, y)
                pred.print_model_metrics(trained_models["CatBoost"], X, y, "CatBoost")

            # Pick the best model
            best = pred.get_best_model()
            print(f"Best model: {best}")

        except Exception as train_error:
            error_msg = f"Failed to train models: {str(train_error)}"
            if task_id:
                task_manager.complete_task(task_id, error=error_msg)
            return {
                "error": error_msg,
                "error_type": type(train_error).__name__,
                "data_shape": X.shape if 'X' in locals() else None,
                "columns": list(X.columns) if 'X' in locals() else None,
                "train_error_details": str(train_error)
            }
        update_progress(80, "Saving model...")
        
        # Save model
        try:
            model_dir = Path("data/models")
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / f"segment_energy_{best}.joblib"
            pred.save_model(str(model_path))
        except Exception as save_error:
            error_msg = f"Failed to save model: {str(save_error)}"
            if task_id:
                task_manager.complete_task(task_id, error=error_msg)
            return {
                "error": error_msg,
                "error_type": type(save_error).__name__,
                "data_shape": X.shape if 'X' in locals() else None,
                "columns": list(X.columns) if 'X' in locals() else None,
                "save_error_details": str(save_error)
            }

        update_progress(100, "Training completed successfully!")
        
        result = {
            "model": best,
            "model_path": str(model_path),
            "metrics": pred.model_performance.get(best, {}),
            "feature_count": len(X.columns),
            "sample_count": len(X),
            "all_models_performance": pred.model_performance,
            "feature_importance": pred.feature_importance.get(best, {}),
            "training_summary": {
                "total_models_trained": len(pred.models),
                "best_model": best,
                "best_model_metrics": pred.model_performance.get(best, {}),
                "model_comparison": {name: metrics for name, metrics in pred.model_performance.items()}
            }
        }
        
        # Complete task with actual results
        if task_id:
            task_manager.complete_task(task_id, result=result)
        
        return result
        
    except Exception as e:
        # Return detailed error info for debugging
        error_msg = str(e)
        if task_id:
            task_manager.complete_task(task_id, error=error_msg)
        
        return {
            "error": error_msg,
            "error_type": type(e).__name__,
            "data_shape": None,
            "columns": None,
            "general_error_details": str(e)
        }


def start_training_background(data_dir: str, model_choice: str = "random_forest", task_id: str = None) -> None:
    """Start training in a background thread."""
    def training_worker():
        try:
            train_model(data_dir, model_choice, task_id)
        except Exception as e:
            if task_id:
                task_manager.complete_task(task_id, error=str(e))
    
    # Start training in background thread
    thread = threading.Thread(target=training_worker, daemon=True)
    thread.start()
    
    return thread

