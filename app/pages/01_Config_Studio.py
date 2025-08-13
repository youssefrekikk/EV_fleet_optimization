import os, sys
import streamlit as st

# Ensure 'app' directory is on sys.path so 'services' imports resolve
CURRENT_DIR = os.path.dirname(__file__)
APP_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, '..'))
for p in [APP_DIR, PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from services.config_service import (
    load_overrides,
    save_overrides,
    merged_runtime_config,
    UIOverrides,
)
from config.ev_models import EV_MODELS
from config.driver_profiles import DRIVER_PROFILES


st.set_page_config(page_title="Config Studio", page_icon="🛠️", layout="wide")

st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, rgba(10,18,31,0.9), rgba(25,36,56,0.9)); border-radius: 20px; border: 2px solid rgba(0, 255, 166, 0.3); margin-bottom: 2rem;">
    <h1>⚙️ Config Studio — Bay Area EV Fleet</h1>
    <p style="color: #a0aec0; font-size: 1.1rem;">Safely configure fleet, charging, and optimization settings with intelligent constraints</p>
</div>
""", unsafe_allow_html=True)

ui = load_overrides()

# Fleet Configuration Section
st.markdown("""
<div style="background: rgba(0, 255, 166, 0.1); padding: 1rem; border-radius: 12px; border-left: 4px solid #00ffa6; margin-bottom: 1rem;">
    <h3>🚗 Fleet Configuration</h3>
    <p>Define the size and scope of your EV fleet simulation in the SF Bay Area.</p>
</div>
""", unsafe_allow_html=True)

with st.form("config_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        fleet_size = st.number_input(
            "Fleet Size", 
            min_value=1, 
            max_value=500, 
            value=ui.fleet.fleet_size,
            help="Number of EVs to simulate. Start small (10-50) for testing, scale up for production analysis."
        )
    with col2:
        sim_days = st.number_input(
            "Simulation Days", 
            min_value=1, 
            max_value=60, 
            value=ui.fleet.simulation_days,
            help="How many days of operations to generate. 7-14 days gives good patterns, 30+ for seasonal analysis."
        )
    with col3:
        region = st.selectbox(
            "Region", 
            ["bay_area"], 
            index=0,
            help="Geography — currently SF Bay Area (San Francisco, Oakland, San Jose, Peninsula, North Bay)."
        )

    # Charging Configuration Section
    st.markdown("""
    <div style="background: rgba(0, 212, 255, 0.1); padding: 1rem; border-radius: 12px; border-left: 4px solid #00d4ff; margin: 1rem 0;">
        <h3>🔌 Charging Infrastructure</h3>
        <p>Configure home and public charging availability, power levels, and efficiency.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        enable_home = st.checkbox(
            "Enable Home Charging", 
            value=ui.charging.enable_home_charging,
            help="Allow drivers with home access to charge overnight. Most Bay Area residents have Level 2 (7.4 kW) home charging."
        )
    with col2:
        home_avail = st.slider(
            "Home Charging Availability", 
            0.0, 1.0, 
            float(ui.charging.home_charging_availability), 
            0.05,
            help="Share of drivers with home charging. 0.8 = 80% have home access, typical for Bay Area suburbs."
        )
    with col3:
        home_power = st.number_input(
            "Home Charging Power (kW)", 
            min_value=1.0, 
            max_value=350.0, 
            value=float(ui.charging.home_charging_power), 
            step=0.5,
            help="Typical Level 2 is 7.4 kW. Level 1 is 1.4 kW. High-end homes may have 11-22 kW."
        )
    with col4:
        dc_power = st.number_input(
            "Public Fast Charging Power (kW)", 
            min_value=20.0, 
            max_value=350.0, 
            value=float(ui.charging.public_fast_charging_power), 
            step=5.0,
            help="Fast DC chargers in kW. 50-150 kW common, 350 kW for ultra-fast charging stations."
        )

    # Optimization Configuration Section
    st.markdown("""
    <div style="background: rgba(255, 107, 107, 0.1); padding: 1rem; border-radius: 12px; border-left: 4px solid #ff6b6b; margin: 1rem 0;">
        <h3>🛰️ Route Optimization</h3>
        <p>Configure energy-aware routing algorithms and SOC planning parameters.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        algo = st.selectbox(
            "Routing Algorithm", 
            ["dijkstra", "astar"], 
            index=["dijkstra", "astar"].index(ui.optimization.route_optimization_algorithm),
            help="Dijkstra (robust, guaranteed optimal) vs A* (faster with good heuristic, may be suboptimal)."
        )
    with col2:
        gamma = st.number_input(
            "Gamma Time Weight (kWh/hour)", 
            min_value=0.0, 
            max_value=1.0, 
            value=float(ui.optimization.gamma_time_weight), 
            step=0.01,
            help="Trade-off time vs energy in routing cost. 0.02 = 1 hour of time = 0.02 kWh penalty. Higher values prioritize speed over efficiency."
        )
    with col3:
        price_w = st.number_input(
            "Price Weight (kWh/USD)", 
            min_value=0.0, 
            max_value=10.0, 
            value=float(ui.optimization.price_weight_kwh_per_usd), 
            step=0.1,
            help="Cost sensitivity for SOC planning objective. 2.0 = $1 of charging cost = 2 kWh penalty. Higher values make cost more important than energy."
        )
    with col4:
        buffer_pct = st.slider(
            "Battery Buffer (%)", 
            5, 50, 
            int(ui.optimization.battery_buffer_percentage * 100), 
            1,
            help="Safety margin above minimum SOC in planning. 15% = keep 15% buffer, prevents running too low on battery."
        )
    with col5:
        detour_km = st.number_input(
            "Max Detour for Charging (km)", 
            min_value=0.0, 
            max_value=50.0, 
            value=float(ui.optimization.max_detour_for_charging_km), 
            step=0.5,
            help="Limit for charging station detours. 5 km = don't go more than 5 km out of way to charge."
        )

    # Advanced Optimization Section
    st.markdown("""
    <div style="background: rgba(255, 193, 7, 0.1); padding: 1rem; border-radius: 12px; border-left: 4px solid #ffc107; margin: 1rem 0;">
        <h3>⚡ Advanced Optimization</h3>
        <p>Fine-tune fleet evaluation, SOC planning, and look-ahead parameters.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        fleet_max_days = st.number_input(
            "Fleet Eval Max Days", 
            min_value=0, 
            max_value=365, 
            value=int(ui.optimization.fleet_eval_max_days or 0), 
            help="0 means all days. Limit to speed up large fleet evaluations. 1-7 days for quick testing."
        )
    with col2:
        fleet_sample_frac = st.slider(
            "Trip Sample Fraction/Day", 
            0.0, 1.0, 
            float(ui.optimization.fleet_eval_trip_sample_frac or 0.7), 
            0.05,
            help="Speed up by sampling subset of trips. 0.7 = process 70% of trips per day. 1.0 = all trips (slower but complete)."
        )
    with col3:
        soc_objective = st.selectbox(
            "SOC Objective", 
            ["energy", "cost", "time", "weighted"], 
            index=["energy","cost","time","weighted"].index(ui.optimization.soc_objective),
            help="Optimize charging for: energy (minimize kWh), cost (minimize $), time (minimize hours), or weighted blend."
        )
    with col4:
        planning_mode = st.selectbox(
            "Planning Mode", 
            ["myopic", "next_trip", "rolling_horizon"], 
            index=["myopic","next_trip","rolling_horizon"].index(ui.optimization.planning_mode),
            help="Look-ahead mode: myopic (no future), next_trip (reserve for next), rolling_horizon (reserve for K trips)."
        )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        alpha_usd_per_hour = st.number_input(
            "Alpha (USD/hour)", 
            min_value=0.0, 
            max_value=200.0, 
            value=float(ui.optimization.alpha_usd_per_hour),
            help="Value of time for cost-based objective. $50/hour = 1 hour of time = $50 cost penalty. Typical: $20-100/hour."
        )
    with col2:
        beta_kwh_to_usd = st.number_input(
            "Beta (USD/kWh)", 
            min_value=0.0, 
            max_value=5.0, 
            value=float(ui.optimization.beta_kwh_to_usd),
            help="USD per kWh to blend with time. $0.15/kWh = typical Bay Area electricity rate. Higher values make energy more expensive."
        )
    with col3:
        reserve_soc = st.slider(
            "Reserve SOC Floor", 
            0.0, 0.95, 
            float(ui.optimization.reserve_soc), 
            0.01,
            help="Terminal SOC floor when no future energy known. 0.15 = keep 15% minimum. Higher values = more conservative planning."
        )
    with col4:
        reserve_kwh = st.number_input(
            "Reserve kWh (overrides SOC)", 
            min_value=0.0, 
            max_value=200.0, 
            value=float(ui.optimization.reserve_kwh),
            help="If > 0, overrides SOC floor. 10 kWh = keep 10 kWh minimum regardless of battery capacity."
        )

    col1, col2 = st.columns(2)
    with col1:
        horizon_trips = st.number_input(
            "Horizon Trips", 
            min_value=0, 
            max_value=10, 
            value=int(ui.optimization.horizon_trips),
            help="Look-ahead trips in rolling horizon. 3 = plan charging considering next 3 trips. Higher values = better planning but slower."
        )
    with col2:
        horizon_hours = st.number_input(
            "Horizon Hours", 
            min_value=0.0, 
            max_value=72.0, 
            value=float(ui.optimization.horizon_hours),
            help="Time horizon (hours) alternative to trips. 24 = plan for next 24 hours. Useful when trip count is unknown."
        )

    # EV Models and Driver Profiles Section
    st.markdown("""
    <div style="background: rgba(156, 39, 176, 0.1); padding: 1rem; border-radius: 12px; border-left: 4px solid #9c27b0; margin: 1rem 0;">
        <h3>🚗 Fleet Composition & Driver Behavior</h3>
        <p>Adjust EV model market shares and driver profile proportions for realistic Bay Area fleet simulation.</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("📊 EV Models Market Share (must sum to 1.0)", expanded=False):
        model_keys = list(EV_MODELS.keys())
        current_shares = ui.model_dump().get('ev_models_market_share') or {}
        sliders_models = {}
        
        # Group models by category for better organization
        tesla_models = [k for k in model_keys if 'tesla' in k]
        other_models = [k for k in model_keys if 'tesla' not in k]
        
        st.write("**Tesla Models (Bay Area dominant):**")
        for mk in tesla_models:
            default = float(current_shares.get(mk, EV_MODELS[mk].get('market_share', 0.0)))
            sliders_models[mk] = st.slider(f"{mk.replace('_', ' ').title()}", 0.0, 1.0, default, 0.01)
        
        st.write("**Other EV Models:**")
        for mk in other_models:
            default = float(current_shares.get(mk, EV_MODELS[mk].get('market_share', 0.0)))
            sliders_models[mk] = st.slider(f"{mk.replace('_', ' ').title()}", 0.0, 1.0, default, 0.01)
        
        total_models = sum(sliders_models.values())
        st.metric("Total Market Share", f"{total_models:.2f}")
        if abs(total_models - 1.0) > 1e-6:
            st.warning("⚠️ Total market share should be 1.0 — values will be normalized on save.")

    with st.expander("👥 Driver Profiles Proportion (must sum to 1.0)", expanded=False):
        prof_keys = list(DRIVER_PROFILES.keys())
        current_props = ui.model_dump().get('driver_profiles_proportion') or {}
        sliders_profiles = {}
        
        for pk in prof_keys:
            default = float(current_props.get(pk, DRIVER_PROFILES[pk].get('proportion', 0.0)))
            sliders_profiles[pk] = st.slider(f"{pk.replace('_', ' ').title()}", 0.0, 1.0, default, 0.01)
        
        total_prof = sum(sliders_profiles.values())
        st.metric("Total Proportion", f"{total_prof:.2f}")
        if abs(total_prof - 1.0) > 1e-6:
            st.warning("⚠️ Total proportion should be 1.0 — values will be normalized on save.")

    submitted = st.form_submit_button("💾 Save Configuration", use_container_width=True)

if submitted:
    new_ui = UIOverrides(
        fleet={
            "fleet_size": int(fleet_size),
            "simulation_days": int(sim_days),
            "region": region,
        },
        charging={
            "enable_home_charging": bool(enable_home),
            "home_charging_availability": float(home_avail),
            "home_charging_power": float(home_power),
            "public_fast_charging_power": float(dc_power),
        },
        optimization={
            "route_optimization_algorithm": algo,
            "gamma_time_weight": float(gamma),
            "price_weight_kwh_per_usd": float(price_w),
            "battery_buffer_percentage": float(buffer_pct) / 100.0,
            "max_detour_for_charging_km": float(detour_km),
            "fleet_eval_max_days": int(fleet_max_days) if fleet_max_days > 0 else None,
            "fleet_eval_trip_sample_frac": float(fleet_sample_frac),
            "soc_objective": soc_objective,
            "alpha_usd_per_hour": float(alpha_usd_per_hour),
            "beta_kwh_to_usd": float(beta_kwh_to_usd),
            "planning_mode": planning_mode,
            "reserve_soc": float(reserve_soc),
            "reserve_kwh": float(reserve_kwh),
            "horizon_trips": int(horizon_trips),
            "horizon_hours": float(horizon_hours),
        },
        ev_models_market_share=sliders_models,
        driver_profiles_proportion=sliders_profiles,
    )
    save_overrides(new_ui)
    st.success("✅ Configuration saved successfully! Changes will apply to next data generation and optimization runs.")

st.divider()

# Runtime Configuration Display
st.markdown("""
<div style="background: rgba(76, 175, 80, 0.1); padding: 1rem; border-radius: 12px; border-left: 4px solid #4caf50; margin: 1rem 0;">
    <h3>🔧 Effective Runtime Configuration</h3>
    <p>Current merged configuration (UI overrides + Python defaults):</p>
</div>
""", unsafe_allow_html=True)

st.json(merged_runtime_config())

