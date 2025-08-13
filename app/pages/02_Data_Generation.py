import os, sys
import streamlit as st
import time

# Ensure imports resolve when Streamlit runs pages directly
CURRENT_DIR = os.path.dirname(__file__)
APP_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
for p in [APP_DIR, PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from services.config_service import merged_runtime_config
from services.network_service import build_or_load_network
from services.generation_service import generate_synthetic_data
from services.task_manager import task_manager


st.set_page_config(page_title="Data Generation", page_icon="🧪", layout="wide")

st.title("🧪 Data Generation Studio")
st.caption("Build the road network and generate synthetic datasets with progress")

cfg = merged_runtime_config()

col1, col2 = st.columns(2)
with col1:
    st.subheader("🛣️ Road Network")
    if st.button("Build / Load Bay Area Network", use_container_width=True, type="primary"):
        # Start network build task
        task_id = f"network_build_{int(time.time())}"
        task_manager.start_task(
            task_id=task_id,
            task_type="network_build",
            params={"description": "Building Bay Area road network"}
        )
        st.success(f"Network build started! Task ID: {task_id}")
        st.rerun()

with col2:
    st.subheader("📊 Synthetic Data")
    fleet_size = cfg["fleet"]["fleet_size"]
    sim_days = cfg["fleet"]["simulation_days"]
    st.write("Using market shares and driver proportions from UI overrides if set.")
    if st.button(f"Generate data (fleet={fleet_size}, days={sim_days})", use_container_width=True, type="primary"):
        # Start data generation task
        task_id = f"data_generation_{int(time.time())}"
        task_manager.start_task(
            task_id=task_id,
            task_type="data_generation",
            params={
                "fleet_size": fleet_size,
                "simulation_days": sim_days,
                "start_date": cfg["fleet"].get("start_date", "2024-01-01"),
                "ev_models_market_share": cfg.get("ev_models_market_share"),
                "driver_profiles_proportion": cfg.get("driver_profiles_proportion"),
            }
        )
        st.success(f"Data generation started! Task ID: {task_id}")
        st.rerun()

# Show running tasks
st.subheader("📈 Task Status")
running_tasks = task_manager.get_running_tasks()
if running_tasks:
    for task_id, task_info in running_tasks.items():
        with st.expander(f"Task {task_id} - {task_info.get('type', 'Unknown')}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**Status:** {task_info.get('status', 'Unknown')}")
                st.write(f"**Started:** {task_info.get('started_at', 'Unknown')[:19]}")
            
            with col2:
                st.write(f"**Progress:** {task_info.get('progress', 0)}%")
                if task_info.get('message'):
                    st.write(f"**Message:** {task_info.get('message')}")
            
            with col3:
                if task_info.get('params'):
                    st.write("**Parameters:**")
                    st.json(task_info.get('params', {}))
else:
    st.info("No tasks currently running.")

st.info("💡 **Tip:** Tasks run in the background. You can navigate to other pages while they're running and check the Home page for global status updates.")

