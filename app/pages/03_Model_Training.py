import os, sys
from pathlib import Path

import streamlit as st
import pandas as pd

# Add project root to path for imports
CURRENT_DIR = os.path.dirname(__file__)
APP_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, '..'))
for p in [APP_DIR, PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from services.config_service import merged_runtime_config
from services.ml_service import train_model
from services.task_manager import start_model_training, get_training_status, get_all_training_tasks

st.set_page_config(page_title="Model Training", page_icon="🤖", layout="wide")

st.title("🤖 ML Model Training Studio")
st.caption("Train energy consumption prediction models for EV fleet optimization")

# Check for required data
data_dir = Path("data/synthetic")
required_files = ["segments.csv", "fleet_info.csv", "weather.csv"]

missing_files = [f for f in required_files if not (data_dir / f).exists()]

if missing_files:
    st.error(f"""
    ❌ **Missing required data files:**
    
    {', '.join(missing_files)}
    
    Please generate synthetic data first in the **Data Generation** page.
    """)
    st.stop()

# Data overview
st.subheader("📊 Data Overview")
col1, col2, col3, col4 = st.columns(4)

with col1:
    segments_count = len(pd.read_csv(data_dir / "segments.csv"))
    st.metric("Total Segments", f"{segments_count:,}")
    
with col2:
    fleet_count = len(pd.read_csv(data_dir / "fleet_info.csv"))
    st.metric("Total Vehicles", f"{fleet_count:,}")
    
with col3:
    weather_count = len(pd.read_csv(data_dir / "weather.csv"))
    st.metric("Weather Days", f"{weather_count:,}")
    
with col4:
    models_dir = Path("data/models")
    trained_models = len(list(models_dir.glob("*.joblib"))) if models_dir.exists() else 0
    st.metric("Trained Models", f"{trained_models:,}")

# Check for existing trained models
st.subheader("🔍 Existing Models")
models_dir = Path("data/models")
if models_dir.exists():
    model_files = list(models_dir.glob("*.joblib"))
    if model_files:
        for model_file in model_files:
            st.success(f"✅ {model_file.name}")
    else:
        st.info("No trained models found yet.")
else:
    st.info("Models directory doesn't exist yet.")

# Training form
st.subheader("🚀 Start Training")
with st.form("training_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        model_choice = st.selectbox(
            "Model Type",
            ["random_forest", "gradient_boosting", "xgboost","xgboost_optuna", "lightgbm", "catboost","all"],
            help="Choose the ML algorithm to train"
        )
    
    with col2:
        data_directory = st.text_input(
            "Data Directory",
            value="data/synthetic",
            help="Path to synthetic data directory"
        )
    
    submitted = st.form_submit_button("🚀 Start Training", type="primary")
    
    if submitted:
        # Start training task
        task_id = start_model_training(data_directory, model_choice)
        st.session_state['current_training_task'] = task_id
        
        # Start actual training in background
        from services.ml_service import start_training_background
        start_training_background(data_directory, model_choice, task_id)
        
        st.success(f"🎯 Training started! Task ID: {task_id}")
        st.rerun()

# Show current training status
if 'current_training_task' in st.session_state:
    task_id = st.session_state['current_training_task']
    status = get_training_status(task_id)
    
    if status:
        st.subheader("📈 Training Progress")
        
        # Progress bar
        progress = status.get('progress', 0)
        st.progress(progress / 100)
        
        # Status info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Status", status.get('status', 'Unknown'))
        with col2:
            st.metric("Progress", f"{progress}%")
        with col3:
            started_at = status.get('started_at', 'Unknown')
            if started_at != 'Unknown':
                st.metric("Started", started_at[:19])
        
        # Show current message
        if status.get('message'):
            st.info(f"💬 {status['message']}")
        
        # Check if this is a real training task or just a placeholder
        if status.get('status') == 'completed' and status.get('result', {}).get('message') == 'Task processed':
            st.warning("⚠️ **Training Status Issue Detected**")
            st.error("""
            The task manager marked this task as completed before the actual training finished.
            This usually happens when there's a mismatch between the task manager and the ML service.
            
            **What to do:**
            1. Check if there are any Python processes running in the background
            2. Look for training logs in the terminal/console
            3. Try refreshing the page or starting a new training session
            """)
            
            # Show system status
            st.subheader("🔍 System Status Check")
            import psutil
            try:
                python_processes = [p for p in psutil.process_iter(['pid', 'name', 'cmdline']) 
                                  if 'python' in p.info['name'].lower()]
                if python_processes:
                    st.info(f"Found {len(python_processes)} Python processes running")
                    for proc in python_processes[:5]:  # Show first 5
                        st.write(f"- PID {proc.info['pid']}: {proc.info['name']}")
                else:
                    st.info("No Python processes found running")
            except:
                st.info("Could not check system processes")
            
            # Clear the problematic task
            if st.button("🗑️ Clear This Task", help="Remove the problematic task and start fresh"):
                del st.session_state['current_training_task']
                st.rerun()
            
            # Don't proceed with the rest of the logic
            st.stop()
        
        # Handle different statuses
        if status.get('status') == 'running' or status.get('status') == 'processing':
            st.info("🔄 Training in progress... This may take several minutes.")
            
            # Manual refresh button
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Refresh Status", help="Check for latest updates"):
                    st.rerun()
            
            with col2:
                if st.button("📊 Check Training Logs", help="View detailed training progress"):
                    st.subheader("📋 Training Logs")
                    st.info("Training logs will appear here as the model trains...")
                    st.info("💡 Tip: The training process includes:")
                    st.write("1. Data loading and validation")
                    st.write("2. Feature engineering")
                    st.write("3. Model training (Random Forest, XGBoost, LightGBM, etc.)")
                    st.write("4. Model evaluation and comparison")
                    st.write("5. Saving the best model")
            
            # Auto-refresh every 10 seconds
            st.markdown("""
            <script>
            setTimeout(function(){
                window.location.reload();
            }, 10000);
            </script>
            """, unsafe_allow_html=True)
            
        elif status.get('status') == 'completed':
            # Check if this is a premature completion (just "Task processed" message)
            result = status.get('result', {})
            if result and isinstance(result, dict) and result.get('message') == 'Task processed':
                st.warning("⚠️ Task marked as completed but training may still be in progress...")
                st.info("🔄 This usually means the task manager completed the task before training finished. Check the training logs below.")
                
                # Don't clear the task yet - let user see the actual training status
                st.subheader("📋 Training History")
                all_tasks = get_all_training_tasks()
                if all_tasks:
                    for task_id, task_info in all_tasks.items():
                        if task_info.get('type') == 'model_training':
                            with st.expander(f"Task {task_id} - {task_info.get('type', 'Unknown')}", expanded=True):
                                st.write(f"**Status:** {task_info.get('status', 'Unknown')}")
                                st.write(f"**Progress:** {task_info.get('progress', 0)}%")
                                if task_info.get('result'):
                                    st.write("**Result:**")
                                    st.json(task_info.get('result', {}))
                                if task_info.get('error'):
                                    st.error(f"**Error:** {task_info.get('error')}")
                
                st.stop()  # Don't proceed with the success display
            
            # Real completion with actual results
            st.success("✅ Training completed successfully!")
            result = status.get('result', {})
            
            if result and isinstance(result, dict):
                # Display training summary
                st.subheader("🎯 Training Results")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Best Model", result.get('model', 'Unknown'))
                    st.metric("Features", result.get('feature_count', 'Unknown'))
                
                with col2:
                    st.metric("Samples", result.get('sample_count', 'Unknown'))
                    st.metric("Models Trained", result.get('training_summary', {}).get('total_models_trained', 'Unknown'))
                
                with col3:
                    if 'model_path' in result:
                        st.metric("Model Saved", "✅", help=f"Path: {result['model_path']}")
                
                with col4:
                    if 'metrics' in result and result['metrics']:
                        best_metrics = result['metrics']
                        st.metric("Best R²", f"{best_metrics.get('R2', 'N/A'):.4f}")
                
                # Show detailed metrics for all models
                if 'training_summary' in result and 'model_comparison' in result['training_summary']:
                    st.subheader("📊 Model Performance Comparison")
                    
                    model_comparison = result['training_summary']['model_comparison']
                    if model_comparison:
                        # Create a comparison table
                        comparison_data = []
                        for model_name, metrics in model_comparison.items():
                            comparison_data.append({
                                "Model": model_name.replace('_', ' ').title(),
                                "MAE": f"{metrics.get('MAE', 'N/A'):.4f}",
                                "RMSE": f"{metrics.get('RMSE', 'N/A'):.4f}",
                                "R²": f"{metrics.get('R2', 'N/A'):.4f}",
                                "MAPE": f"{metrics.get('MAPE', 'N/A'):.2f}%"
                            })
                        
                        if comparison_data:
                            st.dataframe(comparison_data, use_container_width=True)
                
                # Show feature importance if available
                if 'feature_importance' in result and result['feature_importance']:
                    st.subheader("🔍 Feature Importance")
                    
                    # Sort features by importance
                    feature_importance = result['feature_importance']
                    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                    
                    # Show top 10 features
                    top_features = sorted_features[:10]
                    feature_names = [f[0] for f in top_features]
                    importance_values = [f[1] for f in top_features]
                    
                    # Create a bar chart
                    import plotly.express as px
                    fig = px.bar(
                        x=importance_values,
                        y=feature_names,
                        orientation='h',
                        title="Top 10 Most Important Features",
                        labels={'x': 'Importance', 'y': 'Feature'}
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Show model file info
                if 'model_path' in result:
                    st.subheader("💾 Model Files")
                    st.success(f"**Model saved to:** `{result['model_path']}`")
                    
                    # Check if model file exists
                    model_path = Path(result['model_path'])
                    if model_path.exists():
                        file_size = model_path.stat().st_size / (1024 * 1024)  # MB
                        st.info(f"**File size:** {file_size:.2f} MB")
                        st.info(f"**Ready for:** Optimization & Inference")
                    else:
                        st.warning("⚠️ Model file not found at specified path")
                
                # Next steps
                st.subheader("🚀 Next Steps")
                st.info("""
                🎯 **Your model is ready! Here's what you can do next:**
                
                1. **Route Optimization**: Use this trained model for energy-aware routing
                2. **Model Analysis**: Compare different models' performance
                3. **Feature Analysis**: Understand which factors most affect energy consumption
                4. **Production Use**: Deploy the model for real-time predictions
                """)
                
                # Success celebration
                st.balloons()
            else:
                st.warning("⚠️ Training completed but no detailed results available")
                if result:
                    st.json(result)
            
            # Clear the task from session
            del st.session_state['current_training_task']
            
        elif status.get('status') == 'failed':
            st.error("❌ Training failed!")
            error = status.get('error', 'Unknown error')
            st.error(f"Error: {error}")
            
            # Clear the task from session
            del st.session_state['current_training_task']

# Show all training tasks
st.subheader("📋 Training History")
all_tasks = get_all_training_tasks()

if all_tasks:
    for task_id, task_info in all_tasks.items():
        with st.expander(f"Task {task_id} - {task_info.get('type', 'Unknown')}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**Status:** {task_info.get('status', 'Unknown')}")
                st.write(f"**Started:** {task_info.get('started_at', 'Unknown')[:19]}")
            
            with col2:
                if task_info.get('finished_at'):
                    st.write(f"**Finished:** {task_info.get('finished_at', 'Unknown')[:19]}")
                st.write(f"**Progress:** {task_info.get('progress', 0)}%")
            
            with col3:
                if task_info.get('params'):
                    st.write("**Parameters:**")
                    st.json(task_info.get('params', {}))
            
            if task_info.get('result'):
                st.write("**Result:**")
                st.json(task_info.get('result', {}))
            
            if task_info.get('error'):
                st.error(f"**Error:** {task_info.get('error')}")
else:
    st.info("No training tasks found yet.")

# Model comparison and tips
st.subheader("📚 Model Comparison & Tips")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **🔍 Model Types:**
    
    - **Random Forest**: Good balance of speed and accuracy
    - **Gradient Boosting**: Often best performance, slower training
    - **XGBoost**: Excellent performance, good for large datasets
    - **LightGBM**: Fast training, good performance
    - **CatBoost**: Handles categorical features well
    
    **📊 Performance Metrics:**
    - **MAE**: Mean Absolute Error (lower is better)
    - **RMSE**: Root Mean Square Error (lower is better)
    - **R²**: Coefficient of determination (closer to 1 is better)
    - **MAPE**: Mean Absolute Percentage Error (lower is better)
    """)

with col2:
    st.markdown("""
    **💡 Training Tips:**
    
    - **Quick Test**: Use Random Forest for fast iteration
    - **Production**: Use Gradient Boosting or XGBoost
    - **Large Datasets**: Consider LightGBM for speed
    - **Feature Engineering**: The model automatically handles this
    - **Validation**: Uses holdout validation for unbiased estimates
    
    **⚡ Performance:**
    - Small fleet (10-50): 1-3 minutes
    - Medium fleet (100-200): 3-8 minutes  
    - Large fleet (500+): 8-15 minutes
    """)

# Quick actions
st.subheader("🚀 Quick Actions")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔄 Refresh All", help="Refresh training status and data"):
        st.rerun()

with col2:
    if st.button("📊 View Data Sample", help="Show sample of training data"):
        try:
            sample_data = pd.read_csv(data_dir / "segments.csv", nrows=5)
            st.dataframe(sample_data)
        except Exception as e:
            st.error(f"Failed to load sample data: {e}")

with col3:
    if st.button("🧹 Clear Completed Tasks", help="Remove old completed tasks"):
        from services.task_manager import task_manager
        task_manager.clear_completed_tasks()
        st.success("Cleared completed tasks!")
        st.rerun()

