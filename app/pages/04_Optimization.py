import os, sys
import streamlit as st
import time
from pathlib import Path

CURRENT_DIR = os.path.dirname(__file__)
APP_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, '..'))
for p in [APP_DIR, PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from services.opt_service import run_optimization
from services.task_manager import task_manager
from services.optimization_config_service import optimization_config_service


st.set_page_config(page_title="Optimization", page_icon="🛰️", layout="wide")

st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, rgba(10,18,31,0.9), rgba(25,36,56,0.9)); border-radius: 20px; border: 2px solid rgba(0, 255, 166, 0.3); margin-bottom: 2rem;">
    <h1>🛰️ Route Optimization Studio</h1>
    <p style="color: #a0aec0; font-size: 1.1rem;">Configure and run energy-aware routing and SOC planning with comprehensive parameter control</p>
</div>
""", unsafe_allow_html=True)

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

# Load current optimization configuration
current_config = optimization_config_service.get_merged_config()



# Optimization Configuration Form
st.markdown("""
<div style="background: rgba(0, 255, 166, 0.1); padding: 1rem; border-radius: 12px; border-left: 4px solid #00ffa6; margin: 1rem 0;">
    <h3>⚙️ Optimization Configuration</h3>
    <p>Fine-tune routing algorithms, charging strategies, and planning parameters for optimal fleet performance.</p>
</div>
""", unsafe_allow_html=True)

with st.form("optimization_config_form"):
    # Routing Configuration
    st.subheader("🛣️ Routing Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        routing_algo = st.selectbox(
            "Routing Algorithm",
            options=current_config["routing"]["algorithm"]["options"],
            index=current_config["routing"]["algorithm"]["options"].index(current_config["routing"]["algorithm"]["value"]),
            help=current_config["routing"]["algorithm"]["help"]
        )
        
        gamma_time = st.number_input(
            "Gamma Time Weight (kWh/hour)",
            min_value=current_config["routing"]["gamma_time_weight"]["min"],
            max_value=current_config["routing"]["gamma_time_weight"]["max"],
            value=current_config["routing"]["gamma_time_weight"]["value"],
            step=current_config["routing"]["gamma_time_weight"]["step"],
            help=current_config["routing"]["gamma_time_weight"]["help"]
        )
    
    with col2:
        st.info(f"""
        **Current Algorithm:** {routing_algo.upper()}
        
        **Time Weight:** {gamma_time} kWh/hour
        
        **Explanation:** {current_config["routing"]["algorithm"]["description"]}
        """)
    
    # Charging Configuration
    st.subheader("🔌 Charging & Cost Optimization")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        price_weight = st.number_input(
            "Price Sensitivity (kWh/USD)",
            min_value=current_config["charging"]["price_weight_kwh_per_usd"]["min"],
            max_value=current_config["charging"]["price_weight_kwh_per_usd"]["max"],
            value=current_config["charging"]["price_weight_kwh_per_usd"]["value"],
            step=current_config["charging"]["price_weight_kwh_per_usd"]["step"],
            help=current_config["charging"]["price_weight_kwh_per_usd"]["help"]
        )
    
    with col2:
        battery_buffer = st.slider(
            "Battery Safety Buffer (%)",
            min_value=int(current_config["charging"]["battery_buffer_percentage"]["min"] * 100),
            max_value=int(current_config["charging"]["battery_buffer_percentage"]["max"] * 100),
            value=int(current_config["charging"]["battery_buffer_percentage"]["value"] * 100),
            step=int(current_config["charging"]["battery_buffer_percentage"]["step"] * 100),
            help=current_config["charging"]["battery_buffer_percentage"]["help"]
        )
    
    with col3:
        max_detour = st.number_input(
            "Max Charging Detour (km)",
            min_value=current_config["charging"]["max_detour_for_charging_km"]["min"],
            max_value=current_config["charging"]["max_detour_for_charging_km"]["max"],
            value=current_config["charging"]["max_detour_for_charging_km"]["value"],
            step=current_config["charging"]["max_detour_for_charging_km"]["step"],
            help=current_config["charging"]["max_detour_for_charging_km"]["help"]
        )
    
    # Planning Configuration
    st.subheader("📅 Planning & Horizon Configuration")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        planning_mode = st.selectbox(
            "Planning Strategy",
            options=current_config["planning"]["planning_mode"]["options"],
            index=current_config["planning"]["planning_mode"]["options"].index(current_config["planning"]["planning_mode"]["value"]),
            help=current_config["planning"]["planning_mode"]["help"]
        )
        
        horizon_trips = st.number_input(
            "Look-ahead Trips",
            min_value=current_config["planning"]["horizon_trips"]["min"],
            max_value=current_config["planning"]["horizon_trips"]["max"],
            value=current_config["planning"]["horizon_trips"]["value"],
            step=current_config["planning"]["horizon_trips"]["step"],
            help=current_config["planning"]["horizon_trips"]["help"]
        )
    
    with col2:
        soc_objective = st.selectbox(
            "SOC Routing Objective",
            options=current_config["soc_routing"]["soc_objective"]["options"],
            index=current_config["soc_routing"]["soc_objective"]["options"].index(current_config["soc_routing"]["soc_objective"]["value"]),
            help=current_config["soc_routing"]["soc_objective"]["help"]
        )
        
        reserve_soc = st.slider(
            "SOC Reserve Fraction",
            min_value=current_config["planning"]["reserve_soc"]["min"],
            max_value=current_config["planning"]["reserve_soc"]["max"],
            value=current_config["planning"]["reserve_soc"]["value"],
            step=current_config["planning"]["reserve_soc"]["step"],
            help=current_config["planning"]["reserve_soc"]["help"]
        )
    
    with col3:
        alpha_time = st.number_input(
            "Value of Time (USD/hour)",
            min_value=current_config["soc_routing"]["alpha_usd_per_hour"]["min"],
            max_value=current_config["soc_routing"]["alpha_usd_per_hour"]["max"],
            value=current_config["soc_routing"]["alpha_usd_per_hour"]["value"],
            step=current_config["soc_routing"]["alpha_usd_per_hour"]["step"],
            help=current_config["soc_routing"]["alpha_usd_per_hour"]["help"]
        )
        
        beta_energy = st.number_input(
            "Energy to Cost Conversion",
            min_value=current_config["soc_routing"]["beta_kwh_to_usd"]["min"],
            max_value=current_config["soc_routing"]["beta_kwh_to_usd"]["max"],
            value=current_config["soc_routing"]["beta_kwh_to_usd"]["value"],
            step=current_config["soc_routing"]["beta_kwh_to_usd"]["step"],
            help=current_config["soc_routing"]["beta_kwh_to_usd"]["help"]
        )
    
    # Fleet Evaluation Settings
    st.subheader("📊 Fleet Evaluation Settings")
    col1, col2 = st.columns(2)
    
    with col1:
        eval_max_days = st.number_input(
            "Max Days to Evaluate",
            min_value=current_config["evaluation"]["fleet_eval_max_days"]["min"],
            max_value=current_config["evaluation"]["fleet_eval_max_days"]["max"],
            value=current_config["evaluation"]["fleet_eval_max_days"]["value"],
            step=current_config["evaluation"]["fleet_eval_max_days"]["step"],
            help=current_config["evaluation"]["fleet_eval_max_days"]["help"]
        )
    
    with col2:
        trip_sample_frac = st.slider(
            "Trip Sampling Fraction",
            min_value=current_config["evaluation"]["fleet_eval_trip_sample_frac"]["min"],
            max_value=current_config["evaluation"]["fleet_eval_trip_sample_frac"]["max"],
            value=current_config["evaluation"]["fleet_eval_trip_sample_frac"]["value"],
            step=current_config["evaluation"]["fleet_eval_trip_sample_frac"]["step"],
            help=current_config["evaluation"]["fleet_eval_trip_sample_frac"]["help"]
        )
    
    # Submit Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        submitted = st.form_submit_button("💾 Save Configuration & Run Optimization", use_container_width=True, type="primary")

if submitted:
    # Save the configuration
    new_config = {
        "routing": {
            "algorithm": {"value": routing_algo},
            "gamma_time_weight": {"value": gamma_time}
        },
        "charging": {
            "price_weight_kwh_per_usd": {"value": price_weight},
            "battery_buffer_percentage": {"value": battery_buffer / 100.0},
            "max_detour_for_charging_km": {"value": max_detour}
        },
        "planning": {
            "planning_mode": {"value": planning_mode},
            "horizon_trips": {"value": horizon_trips},
            "reserve_soc": {"value": reserve_soc}
        },
        "soc_routing": {
            "soc_objective": {"value": soc_objective},
            "alpha_usd_per_hour": {"value": alpha_time},
            "beta_kwh_to_usd": {"value": beta_energy}
        },
        "evaluation": {
            "fleet_eval_max_days": {"value": eval_max_days},
            "fleet_eval_trip_sample_frac": {"value": trip_sample_frac}
        }
    }
    
    if optimization_config_service.save_config(new_config):
        st.success("✅ Configuration saved successfully!")
        
        # Start optimization task
        task_id = f"optimization_{int(time.time())}"
        task_manager.start_task(
            task_id=task_id,
            task_type="optimization",
            params={
                "routes_csv": routes_csv,
                "fleet_csv": fleet_csv,
                "weather_csv": weather_csv,
                "algorithm": routing_algo,
                "gamma_time_weight": gamma_time,
                "price_weight": price_weight,
                "battery_buffer": battery_buffer / 100.0,
                "max_detour": max_detour,
                "planning_mode": planning_mode,
                "soc_objective": soc_objective,
                "reserve_soc": reserve_soc,
                "horizon_trips": horizon_trips,
                "eval_max_days": eval_max_days,
                "trip_sample_frac": trip_sample_frac
            }
        )
        st.success(f"🚀 Optimization started! Task ID: {task_id}")
        st.rerun()
    else:
        st.error("❌ Failed to save configuration. Please try again.")

st.markdown("---")

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

st.markdown("---")

# Current Configuration Display
st.subheader("🔧 Current Configuration")
st.markdown("""
<div style="background: rgba(76, 175, 80, 0.1); padding: 1rem; border-radius: 12px; border-left: 4px solid #4caf50; margin: 1rem 0;">
    <p>Active optimization parameters (saved configuration + defaults):</p>
</div>
""", unsafe_allow_html=True)

st.json(optimization_config_service.get_merged_config())

st.info("💡 **Tip:** Optimization runs in the background. You can navigate to other pages while it's running and check the Home page for global status updates.")

