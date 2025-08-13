import os, sys
import streamlit as st
import time
from pathlib import Path

CURRENT_DIR = os.path.dirname(__file__)
APP_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
for p in [APP_DIR, PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from services.opt_service import run_optimization
from services.task_manager import task_manager


st.set_page_config(page_title="Optimization", page_icon="🛰️", layout="wide")

st.title("🛰️ Route Optimization Studio")
st.caption("Run energy-aware routing and SOC planning, then compare KPIs")

# Check for required data
data_dir = Path("data/synthetic")
required_files = ["routes.csv", "fleet_info.csv", "weather.csv"]
missing_files = [f for f in required_files if not (data_dir / f).exists()]

if missing_files:
    st.error(f"""
    ❌ **Missing required data files:**
    
    {', '.join(missing_files)}
    
    Please generate synthetic data first in the **Data Generation** page.
    """)
    st.stop()

routes_csv = "data/synthetic/routes.csv"
fleet_csv = "data/synthetic/fleet_info.csv"
weather_csv = "data/synthetic/weather.csv"

st.subheader("⚙️ Optimization Parameters")
col1, col2 = st.columns(2)

with col1:
    date = st.text_input("Date (YYYY-MM-DD)", value="2024-01-01", help="Date to optimize routes for")
    algorithm = st.selectbox("Routing algorithm", ["dijkstra", "astar"], index=1, help="Pathfinding algorithm to use")

with col2:
    soc_planning = st.checkbox("SOC-aware planning", value=False, help="Enable battery state-of-charge planning")
    if soc_planning:
        reserve_soc = st.slider("Reserve SOC (%)", 10, 30, 20, help="Minimum battery level to maintain")

st.subheader("🚀 Run Optimization")
if st.button("Run optimization", use_container_width=True, type="primary"):
    # Start optimization task
    task_id = f"optimization_{int(time.time())}"
    task_manager.start_task(
        task_id=task_id,
        task_type="optimization",
        params={
            "routes_csv": routes_csv,
            "fleet_csv": fleet_csv,
            "weather_csv": weather_csv,
            "date": date,
            "algorithm": algorithm,
            "soc_planning": soc_planning,
            "reserve_soc": reserve_soc if soc_planning else None
        }
    )
    st.success(f"Optimization started! Task ID: {task_id}")
    st.rerun()

# Show running optimization tasks
st.subheader("📈 Optimization Status")
running_tasks = task_manager.get_running_tasks()
optimization_tasks = {k: v for k, v in running_tasks.items() if v.get('type') == 'optimization'}

if optimization_tasks:
    for task_id, task_info in optimization_tasks.items():
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
    st.info("No optimization tasks currently running.")

st.info("💡 **Tip:** Optimization runs in the background. You can navigate to other pages while it's running and check the Home page for global status updates.")

