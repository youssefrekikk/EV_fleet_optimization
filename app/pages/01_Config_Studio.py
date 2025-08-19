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
    
    col5, col6 = st.columns(2)
    with col5:
        charging_efficiency = st.slider(
            "Charging Efficiency", 
            min_value=0.6, 
            max_value=1.0, 
            value=float(ui.charging.charging_efficiency), 
            step=0.05,
            help="Overall charging efficiency including losses. 0.9 = 90% efficiency typical."
        )

    # Route Optimization Section - Moved to Optimization Page
    st.markdown("""
    <div style="background: rgba(255, 107, 107, 0.1); padding: 1rem; border-radius: 12px; border-left: 4px solid #ff6b6b; margin: 1rem 0;">
        <h3>🛰️ Route Optimization</h3>
        <p>⚠️ <strong>Optimization configuration has been moved to the Optimization page for better organization.</strong></p>
        <p>Navigate to the Optimization page to configure routing algorithms, SOC planning, and advanced parameters.</p>
    </div>
    """, unsafe_allow_html=True)
    




    # Driver Profiles Configuration Section
    st.markdown("""
    <div style="background: rgba(255, 193, 7, 0.1); padding: 1rem; border-radius: 12px; border-left: 4px solid #ffc107; margin: 1rem 0;">
        <h3>👥 Driver Profiles Configuration</h3>
        <p>Configure driver profile proportions and detailed parameters for each profile type.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Profile Proportions
    st.subheader("📊 Profile Proportions")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        commuter_proportion = st.slider(
            "Commuter Proportion", 
            min_value=0.0, 
            max_value=1.0, 
            value=float(ui.driver_profiles.commuter_proportion), 
            step=0.01,
            help="Daily commuters (65-130 km/day, 2-4 trips, normal driving style)"
        )
    with col2:
        rideshare_proportion = st.slider(
            "Rideshare Proportion", 
            min_value=0.0, 
            max_value=1.0, 
            value=float(ui.driver_profiles.rideshare_proportion), 
            step=0.01,
            help="Rideshare drivers (160-320 km/day, 15-25 trips, efficient driving)"
        )
    with col3:
        delivery_proportion = st.slider(
            "Delivery Proportion", 
            min_value=0.0, 
            max_value=1.0, 
            value=float(ui.driver_profiles.delivery_proportion), 
            step=0.01,
            help="Delivery drivers (240-480 km/day, 20-40 trips, aggressive driving)"
        )
    with col4:
        casual_proportion = st.slider(
            "Casual Proportion", 
            min_value=0.0, 
            max_value=1.0, 
            value=float(ui.driver_profiles.casual_proportion), 
            step=0.01,
            help="Casual drivers (30-100 km/day, 1-3 trips, eco-friendly driving)"
        )
    
    total_proportion = commuter_proportion + rideshare_proportion + delivery_proportion + casual_proportion
    st.metric("Total Proportion", f"{total_proportion:.2f}")
    if abs(total_proportion - 1.0) > 0.01:
        st.warning("⚠️ Total proportion should be 1.0 (100%) — values will be normalized on save.")
    
    # Detailed Profile Parameters
    st.subheader("🔧 Detailed Profile Parameters")
    
    with st.expander("🚗 Commuter Profile Details", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            commuter_daily_km_min = st.number_input(
                "Daily KM Min", 
                min_value=30, 
                max_value=200, 
                value=ui.driver_profiles.commuter_daily_km_min,
                help="Minimum daily kilometers for commuters"
            )
            commuter_daily_km_max = st.number_input(
                "Daily KM Max", 
                min_value=50, 
                max_value=300, 
                value=ui.driver_profiles.commuter_daily_km_max,
                help="Maximum daily kilometers for commuters"
            )
            commuter_trips_per_day_min = st.number_input(
                "Trips/Day Min", 
                min_value=1, 
                max_value=10, 
                value=ui.driver_profiles.commuter_trips_per_day_min,
                help="Minimum trips per day for commuters"
            )
            commuter_trips_per_day_max = st.number_input(
                "Trips/Day Max", 
                min_value=2, 
                max_value=15, 
                value=ui.driver_profiles.commuter_trips_per_day_max,
                help="Maximum trips per day for commuters"
            )
        with col2:
            commuter_avg_speed_city = st.number_input(
                "City Speed (km/h)", 
                min_value=15, 
                max_value=50, 
                value=ui.driver_profiles.commuter_avg_speed_city,
                help="Average city speed for commuters"
            )
            commuter_avg_speed_highway = st.number_input(
                "Highway Speed (km/h)", 
                min_value=60, 
                max_value=120, 
                value=ui.driver_profiles.commuter_avg_speed_highway,
                help="Average highway speed for commuters"
            )
            commuter_home_charging_prob = st.slider(
                "Home Charging Probability", 
                min_value=0.0, 
                max_value=1.0, 
                value=float(ui.driver_profiles.commuter_home_charging_prob), 
                step=0.05,
                help="Probability of having home charging for commuters"
            )
        with col3:
            commuter_charging_threshold = st.slider(
                "Charging Threshold", 
                min_value=0.1, 
                max_value=0.8, 
                value=float(ui.driver_profiles.commuter_charging_threshold), 
                step=0.05,
                help="Battery level threshold for charging (commuters)"
            )
            commuter_weekend_factor = st.slider(
                "Weekend Factor", 
                min_value=0.1, 
                max_value=2.0, 
                value=float(ui.driver_profiles.commuter_weekend_factor), 
                step=0.1,
                help="Weekend driving factor for commuters"
            )
    
    with st.expander("🚕 Rideshare Profile Details", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            rideshare_daily_km_min = st.number_input(
                "Daily KM Min", 
                min_value=100, 
                max_value=500, 
                value=ui.driver_profiles.rideshare_daily_km_min,
                help="Minimum daily kilometers for rideshare drivers"
            )
            rideshare_daily_km_max = st.number_input(
                "Daily KM Max", 
                min_value=200, 
                max_value=800, 
                value=ui.driver_profiles.rideshare_daily_km_max,
                help="Maximum daily kilometers for rideshare drivers"
            )
            rideshare_trips_per_day_min = st.number_input(
                "Trips/Day Min", 
                min_value=10, 
                max_value=50, 
                value=ui.driver_profiles.rideshare_trips_per_day_min,
                help="Minimum trips per day for rideshare drivers"
            )
            rideshare_trips_per_day_max = st.number_input(
                "Trips/Day Max", 
                min_value=20, 
                max_value=80, 
                value=ui.driver_profiles.rideshare_trips_per_day_max,
                help="Maximum trips per day for rideshare drivers"
            )
        with col2:
            rideshare_avg_speed_city = st.number_input(
                "City Speed (km/h)", 
                min_value=15, 
                max_value=50, 
                value=ui.driver_profiles.rideshare_avg_speed_city,
                help="Average city speed for rideshare drivers"
            )
            rideshare_avg_speed_highway = st.number_input(
                "Highway Speed (km/h)", 
                min_value=60, 
                max_value=120, 
                value=ui.driver_profiles.rideshare_avg_speed_highway,
                help="Average highway speed for rideshare drivers"
            )
            rideshare_home_charging_prob = st.slider(
                "Home Charging Probability", 
                min_value=0.0, 
                max_value=1.0, 
                value=float(ui.driver_profiles.rideshare_home_charging_prob), 
                step=0.05,
                help="Probability of having home charging for rideshare drivers"
            )
        with col3:
            rideshare_charging_threshold = st.slider(
                "Charging Threshold", 
                min_value=0.1, 
                max_value=0.8, 
                value=float(ui.driver_profiles.rideshare_charging_threshold), 
                step=0.05,
                help="Battery level threshold for charging (rideshare)"
            )
            rideshare_weekend_factor = st.slider(
                "Weekend Factor", 
                min_value=0.1, 
                max_value=2.0, 
                value=float(ui.driver_profiles.rideshare_weekend_factor), 
                step=0.1,
                help="Weekend driving factor for rideshare drivers"
            )
    
    with st.expander("📦 Delivery Profile Details", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            delivery_daily_km_min = st.number_input(
                "Daily KM Min", 
                min_value=150, 
                max_value=800, 
                value=ui.driver_profiles.delivery_daily_km_min,
                help="Minimum daily kilometers for delivery drivers"
            )
            delivery_daily_km_max = st.number_input(
                "Daily KM Max", 
                min_value=300, 
                max_value=1000, 
                value=ui.driver_profiles.delivery_daily_km_max,
                help="Maximum daily kilometers for delivery drivers"
            )
            delivery_trips_per_day_min = st.number_input(
                "Trips/Day Min", 
                min_value=15, 
                max_value=80, 
                value=ui.driver_profiles.delivery_trips_per_day_min,
                help="Minimum trips per day for delivery drivers"
            )
            delivery_trips_per_day_max = st.number_input(
                "Trips/Day Max", 
                min_value=25, 
                max_value=100, 
                value=ui.driver_profiles.delivery_trips_per_day_max,
                help="Maximum trips per day for delivery drivers"
            )
        with col2:
            delivery_avg_speed_city = st.number_input(
                "City Speed (km/h)", 
                min_value=15, 
                max_value=50, 
                value=ui.driver_profiles.delivery_avg_speed_city,
                help="Average city speed for delivery drivers"
            )
            delivery_avg_speed_highway = st.number_input(
                "Highway Speed (km/h)", 
                min_value=60, 
                max_value=120, 
                value=ui.driver_profiles.delivery_avg_speed_highway,
                help="Average highway speed for delivery drivers"
            )
            delivery_home_charging_prob = st.slider(
                "Home Charging Probability", 
                min_value=0.0, 
                max_value=1.0, 
                value=float(ui.driver_profiles.delivery_home_charging_prob), 
                step=0.05,
                help="Probability of having home charging for delivery drivers"
            )
        with col3:
            delivery_charging_threshold = st.slider(
                "Charging Threshold", 
                min_value=0.1, 
                max_value=0.8, 
                value=float(ui.driver_profiles.delivery_charging_threshold), 
                step=0.05,
                help="Battery level threshold for charging (delivery)"
            )
            delivery_weekend_factor = st.slider(
                "Weekend Factor", 
                min_value=0.1, 
                max_value=2.0, 
                value=float(ui.driver_profiles.delivery_weekend_factor), 
                step=0.1,
                help="Weekend driving factor for delivery drivers"
            )
    
    with st.expander("🏠 Casual Profile Details", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            casual_daily_km_min = st.number_input(
                "Daily KM Min", 
                min_value=10, 
                max_value=150, 
                value=ui.driver_profiles.casual_daily_km_min,
                help="Minimum daily kilometers for casual drivers"
            )
            casual_daily_km_max = st.number_input(
                "Daily KM Max", 
                min_value=20, 
                max_value=300, 
                value=ui.driver_profiles.casual_daily_km_max,
                help="Maximum daily kilometers for casual drivers"
            )
            casual_trips_per_day_min = st.number_input(
                "Trips/Day Min", 
                min_value=1, 
                max_value=8, 
                value=ui.driver_profiles.casual_trips_per_day_min,
                help="Minimum trips per day for casual drivers"
            )
            casual_trips_per_day_max = st.number_input(
                "Trips/Day Max", 
                min_value=2, 
                max_value=12, 
                value=ui.driver_profiles.casual_trips_per_day_max,
                help="Maximum trips per day for casual drivers"
            )
        with col2:
            casual_avg_speed_city = st.number_input(
                "City Speed (km/h)", 
                min_value=15, 
                max_value=50, 
                value=ui.driver_profiles.casual_avg_speed_city,
                help="Average city speed for casual drivers"
            )
            casual_avg_speed_highway = st.number_input(
                "Highway Speed (km/h)", 
                min_value=60, 
                max_value=120, 
                value=ui.driver_profiles.casual_avg_speed_highway,
                help="Average highway speed for casual drivers"
            )
            casual_home_charging_prob = st.slider(
                "Home Charging Probability", 
                min_value=0.0, 
                max_value=1.0, 
                value=float(ui.driver_profiles.casual_home_charging_prob), 
                step=0.05,
                help="Probability of having home charging for casual drivers"
            )
        with col3:
            casual_charging_threshold = st.slider(
                "Charging Threshold", 
                min_value=0.1, 
                max_value=0.8, 
                value=float(ui.driver_profiles.casual_charging_threshold), 
                step=0.05,
                help="Battery level threshold for charging (casual)"
            )
            casual_weekend_factor = st.slider(
                "Weekend Factor", 
                min_value=0.1, 
                max_value=2.0, 
                value=float(ui.driver_profiles.casual_weekend_factor), 
                step=0.1,
                help="Weekend driving factor for casual drivers"
            )

    # Physics Engine Configuration Section
    st.markdown("""
    <div style="background: rgba(33, 150, 243, 0.1); padding: 1rem; border-radius: 12px; border-left: 4px solid #2196f3; margin: 1rem 0;">
        <h3>⚡ Physics Engine Configuration</h3>
        <p>Fine-tune the physics constants used in energy consumption calculations. Advanced settings for energy model accuracy.</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("🔧 Core Physics Constants", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            air_density = st.number_input(
                "Air Density (kg/m³)", 
                min_value=0.5, 
                max_value=2.0, 
                value=float(ui.physics.air_density), 
                step=0.01,
                help="Air density affects aerodynamic drag. 1.225 kg/m³ at sea level."
            )
            rolling_resistance = st.number_input(
                "Rolling Resistance", 
                min_value=0.005, 
                max_value=0.02, 
                value=float(ui.physics.rolling_resistance), 
                step=0.001,
                help="Tire rolling resistance coefficient. 0.008 typical for modern EV tires."
            )
            gravity = st.number_input(
                "Gravity (m/s²)", 
                min_value=9.0, 
                max_value=10.0, 
                value=float(ui.physics.gravity), 
                step=0.01,
                help="Gravitational acceleration. 9.81 m/s² standard."
            )
        with col2:
            regen_efficiency = st.number_input(
                "Regenerative Braking Efficiency", 
                min_value=0.5, 
                max_value=0.95, 
                value=float(ui.physics.regen_efficiency), 
                step=0.01,
                help="Energy recovery efficiency during braking. 0.8 = 80% recovery."
            )
            motor_efficiency = st.number_input(
                "Motor Efficiency", 
                min_value=0.7, 
                max_value=0.98, 
                value=float(ui.physics.motor_efficiency), 
                step=0.01,
                help="Electric motor efficiency. 0.9 = 90% efficiency."
            )
            battery_efficiency = st.number_input(
                "Battery Efficiency", 
                min_value=0.8, 
                max_value=0.99, 
                value=float(ui.physics.battery_efficiency), 
                step=0.01,
                help="Battery charge/discharge efficiency. 0.95 = 95% efficiency."
            )
        with col3:
            hvac_base_power = st.number_input(
                "HVAC Base Power (kW)", 
                min_value=0.5, 
                max_value=5.0, 
                value=float(ui.physics.hvac_base_power), 
                step=0.1,
                help="Base HVAC system power consumption. 2.0 kW typical."
            )
            auxiliary_power = st.number_input(
                "Auxiliary Power (kW)", 
                min_value=0.1, 
                max_value=1.0, 
                value=float(ui.physics.auxiliary_power), 
                step=0.05,
                help="Power for lights, electronics, etc. 0.25 kW typical."
            )
            auxiliary_usage_factor = st.number_input(
                "Auxiliary Usage Factor", 
                min_value=0.1, 
                max_value=1.0, 
                value=float(ui.physics.auxiliary_usage_factor), 
                step=0.05,
                help="Usage factor for auxiliary systems. 0.5 = 50% usage."
            )
    
    with st.expander("🌡️ Temperature & Regenerative Braking", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            reference_temp_k = st.number_input(
                "Reference Temperature (K)", 
                min_value=270.0, 
                max_value=310.0, 
                value=float(ui.physics.reference_temp_k), 
                step=0.1,
                help="Reference temperature for battery calculations. 298.15 K = 25°C."
            )
            temp_capacity_alpha = st.number_input(
                "Temperature Capacity Coefficient", 
                min_value=0.1, 
                max_value=1.0, 
                value=float(ui.physics.temp_capacity_alpha), 
                step=0.01,
                help="Temperature effect on battery capacity. 0.5 typical."
            )
        with col2:
            regen_speed_threshold = st.number_input(
                "Regen Speed Threshold (m/s)", 
                min_value=5.0, 
                max_value=25.0, 
                value=float(ui.physics.regen_speed_threshold), 
                step=0.5,
                help="Speed above which regenerative braking is fully efficient."
            )
            regen_hard_braking_threshold = st.number_input(
                "Hard Braking Threshold (m/s²)", 
                min_value=1.0, 
                max_value=5.0, 
                value=float(ui.physics.regen_hard_braking_threshold), 
                step=0.1,
                help="Deceleration threshold for hard braking detection."
            )
            regen_moderate_braking_threshold = st.number_input(
                "Moderate Braking Threshold (m/s²)", 
                min_value=0.5, 
                max_value=3.0, 
                value=float(ui.physics.regen_moderate_braking_threshold), 
                step=0.1,
                help="Deceleration threshold for moderate braking detection."
            )
            regen_hard_braking_efficiency = st.number_input(
                "Hard Braking Efficiency", 
                min_value=0.3, 
                max_value=0.8, 
                value=float(ui.physics.regen_hard_braking_efficiency), 
                step=0.01,
                help="Regenerative efficiency during hard braking."
            )
            regen_moderate_braking_efficiency = st.number_input(
                "Moderate Braking Efficiency", 
                min_value=0.7, 
                max_value=0.95, 
                value=float(ui.physics.regen_moderate_braking_efficiency), 
                step=0.01,
                help="Regenerative efficiency during moderate braking."
            )
    
    with st.expander("🔋 Battery & Efficiency Parameters", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            gas_constant = st.number_input(
                "Gas Constant (J/(mol·K))", 
                min_value=8.0, 
                max_value=9.0, 
                value=float(ui.physics.gas_constant), 
                step=0.001,
                help="Universal gas constant for battery calculations."
            )
            activation_energy = st.number_input(
                "Activation Energy (J/mol)", 
                min_value=15000.0, 
                max_value=25000.0, 
                value=float(ui.physics.activation_energy), 
                step=100.0,
                help="Typical activation energy for Li-ion batteries."
            )
            inverter_efficiency = st.number_input(
                "Inverter Efficiency", 
                min_value=0.9, 
                max_value=0.99, 
                value=float(ui.physics.inverter_efficiency), 
                step=0.01,
                help="Power inverter efficiency."
            )
            transmission_efficiency = st.number_input(
                "Transmission Efficiency", 
                min_value=0.95, 
                max_value=0.99, 
                value=float(ui.physics.transmission_efficiency), 
                step=0.01,
                help="Transmission efficiency."
            )
        with col2:
            hvac_cop_heat_pump = st.number_input(
                "HVAC Heat Pump COP", 
                min_value=2.0, 
                max_value=4.0, 
                value=float(ui.physics.hvac_cop_heat_pump), 
                step=0.1,
                help="Coefficient of performance for heat pump."
            )
            hvac_cop_resistive = st.number_input(
                "HVAC Resistive COP", 
                min_value=0.8, 
                max_value=1.2, 
                value=float(ui.physics.hvac_cop_resistive), 
                step=0.01,
                help="Coefficient of performance for resistive heating."
            )
            cabin_thermal_mass = st.number_input(
                "Cabin Thermal Mass (J/K)", 
                min_value=30000.0, 
                max_value=70000.0, 
                value=float(ui.physics.cabin_thermal_mass), 
                step=1000.0,
                help="Approximate cabin thermal mass."
            )
            target_cabin_temp = st.number_input(
                "Target Cabin Temperature (°C)", 
                min_value=18.0, 
                max_value=25.0, 
                value=float(ui.physics.target_cabin_temp), 
                step=0.5,
                help="Target cabin temperature for HVAC calculations."
            )
    
    with st.expander("🌦️ Environmental Factors", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            rolling_resistance_speed_factor = st.number_input(
                "Rolling Resistance Speed Factor", 
                min_value=0.1, 
                max_value=0.2, 
                value=float(ui.physics.rolling_resistance_speed_factor), 
                step=0.01,
                help="Speed-dependent rolling resistance factor."
            )
            rolling_resistance_cold_factor = st.number_input(
                "Cold Weather Factor", 
                min_value=1.1, 
                max_value=1.2, 
                value=float(ui.physics.rolling_resistance_cold_factor), 
                step=0.01,
                help="Rolling resistance increase in cold weather."
            )
            rolling_resistance_hot_factor = st.number_input(
                "Hot Weather Factor", 
                min_value=1.0, 
                max_value=1.1, 
                value=float(ui.physics.rolling_resistance_hot_factor), 
                step=0.01,
                help="Rolling resistance increase in hot weather."
            )
            rolling_resistance_rain_factor = st.number_input(
                "Rain Factor", 
                min_value=1.15, 
                max_value=1.25, 
                value=float(ui.physics.rolling_resistance_rain_factor), 
                step=0.01,
                help="Rolling resistance increase in rain."
            )
        with col2:
            humidity_density_factor = st.number_input(
                "Humidity Density Factor", 
                min_value=0.3, 
                max_value=0.5, 
                value=float(ui.physics.humidity_density_factor), 
                step=0.01,
                help="Humidity correction factor for air density."
            )
            cabin_heat_loss_coeff = st.number_input(
                "Cabin Heat Loss Coefficient (W/K)", 
                min_value=50.0, 
                max_value=150.0, 
                value=float(ui.physics.cabin_heat_loss_coeff), 
                step=5.0,
                help="Cabin heat loss coefficient."
            )
            min_consumption_per_km = st.number_input(
                "Minimum Consumption (kWh/km)", 
                min_value=0.01, 
                max_value=0.05, 
                value=float(ui.physics.min_consumption_per_km), 
                step=0.001,
                help="Minimum energy consumption per kilometer."
            )
            min_driving_time_hours = st.number_input(
                "Minimum Driving Time (hours)", 
                min_value=0.1, 
                max_value=1.0, 
                value=float(ui.physics.min_driving_time_hours), 
                step=0.1,
                help="Minimum assumed driving time."
            )

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
            "charging_efficiency": float(charging_efficiency),
        },
        physics={
            "air_density": float(air_density),
            "rolling_resistance": float(rolling_resistance),
            "gravity": float(gravity),
            "regen_efficiency": float(regen_efficiency),
            "motor_efficiency": float(motor_efficiency),
            "battery_efficiency": float(battery_efficiency),
            "hvac_base_power": float(hvac_base_power),
            "auxiliary_power": float(auxiliary_power),
            "auxiliary_usage_factor": float(auxiliary_usage_factor),
            "reference_temp_k": float(reference_temp_k),
            "temp_capacity_alpha": float(temp_capacity_alpha),
            "regen_speed_threshold": float(regen_speed_threshold),
            "regen_hard_braking_threshold": float(regen_hard_braking_threshold),
            "regen_moderate_braking_threshold": float(regen_moderate_braking_threshold),
            "regen_hard_braking_efficiency": float(regen_hard_braking_efficiency),
            "regen_moderate_braking_efficiency": float(regen_moderate_braking_efficiency),
            "gas_constant": float(gas_constant),
            "activation_energy": float(activation_energy),
            "inverter_efficiency": float(inverter_efficiency),
            "transmission_efficiency": float(transmission_efficiency),
            "hvac_cop_heat_pump": float(hvac_cop_heat_pump),
            "hvac_cop_resistive": float(hvac_cop_resistive),
            "cabin_thermal_mass": float(cabin_thermal_mass),
            "cabin_heat_loss_coeff": float(cabin_heat_loss_coeff),
            "target_cabin_temp": float(target_cabin_temp),
            "rolling_resistance_speed_factor": float(rolling_resistance_speed_factor),
            "rolling_resistance_cold_factor": float(rolling_resistance_cold_factor),
            "rolling_resistance_hot_factor": float(rolling_resistance_hot_factor),
            "rolling_resistance_rain_factor": float(rolling_resistance_rain_factor),
            "humidity_density_factor": float(humidity_density_factor),
            "min_consumption_per_km": float(min_consumption_per_km),
            "min_driving_time_hours": float(min_driving_time_hours),
        },
        driver_profiles={
            "commuter_proportion": float(commuter_proportion),
            "rideshare_proportion": float(rideshare_proportion),
            "delivery_proportion": float(delivery_proportion),
            "casual_proportion": float(casual_proportion),
            # Commuter profile parameters
            "commuter_daily_km_min": int(commuter_daily_km_min),
            "commuter_daily_km_max": int(commuter_daily_km_max),
            "commuter_trips_per_day_min": int(commuter_trips_per_day_min),
            "commuter_trips_per_day_max": int(commuter_trips_per_day_max),
            "commuter_avg_speed_city": int(commuter_avg_speed_city),
            "commuter_avg_speed_highway": int(commuter_avg_speed_highway),
            "commuter_home_charging_prob": float(commuter_home_charging_prob),
            "commuter_charging_threshold": float(commuter_charging_threshold),
            "commuter_weekend_factor": float(commuter_weekend_factor),
            # Rideshare profile parameters
            "rideshare_daily_km_min": int(rideshare_daily_km_min),
            "rideshare_daily_km_max": int(rideshare_daily_km_max),
            "rideshare_trips_per_day_min": int(rideshare_trips_per_day_min),
            "rideshare_trips_per_day_max": int(rideshare_trips_per_day_max),
            "rideshare_avg_speed_city": int(rideshare_avg_speed_city),
            "rideshare_avg_speed_highway": int(rideshare_avg_speed_highway),
            "rideshare_home_charging_prob": float(rideshare_home_charging_prob),
            "rideshare_charging_threshold": float(rideshare_charging_threshold),
            "rideshare_weekend_factor": float(rideshare_weekend_factor),
            # Delivery profile parameters
            "delivery_daily_km_min": int(delivery_daily_km_min),
            "delivery_daily_km_max": int(delivery_daily_km_max),
            "delivery_trips_per_day_min": int(delivery_trips_per_day_min),
            "delivery_trips_per_day_max": int(delivery_trips_per_day_max),
            "delivery_avg_speed_city": int(delivery_avg_speed_city),
            "delivery_avg_speed_highway": int(delivery_avg_speed_highway),
            "delivery_home_charging_prob": float(delivery_home_charging_prob),
            "delivery_charging_threshold": float(delivery_charging_threshold),
            "delivery_weekend_factor": float(delivery_weekend_factor),
            # Casual profile parameters
            "casual_daily_km_min": int(casual_daily_km_min),
            "casual_daily_km_max": int(casual_daily_km_max),
            "casual_trips_per_day_min": int(casual_trips_per_day_min),
            "casual_trips_per_day_max": int(casual_trips_per_day_max),
            "casual_avg_speed_city": int(casual_avg_speed_city),
            "casual_avg_speed_highway": int(casual_avg_speed_highway),
            "casual_home_charging_prob": float(casual_home_charging_prob),
            "casual_charging_threshold": float(casual_charging_threshold),
            "casual_weekend_factor": float(casual_weekend_factor),
        },
        ev_models_market_share=sliders_models if 'sliders_models' in locals() else {},
        ev_models_parameters=ui.ev_models_parameters,
        driver_profiles_proportion=sliders_profiles if 'sliders_profiles' in locals() else {},
    )
    save_overrides(new_ui)
    st.success("✅ Configuration saved successfully! Changes will apply to next data generation and optimization runs.")

st.divider()

# EV Models Configuration Section - OUTSIDE the main form to allow callbacks
st.markdown("""
<div style="background: rgba(156, 39, 176, 0.1); padding: 1rem; border-radius: 12px; border-left: 4px solid #9c27b0; margin: 1rem 0;">
    <h3>🚗 EV Models Configuration</h3>
    <p>Configure detailed parameters for each EV model in your fleet.</p>
</div>
""", unsafe_allow_html=True)

# EV Model Selection - OUTSIDE the form to allow callbacks
st.info("💡 **Note:** EV Model parameters are configured separately from the main form to allow real-time updates when changing models.")
with st.expander("🔧 EV Model Parameters", expanded=False):
    # Initialize session state if not exists
    if "current_ev_model" not in st.session_state:
        st.session_state.current_ev_model = list(EV_MODELS.keys())[0]
        st.session_state.current_ev_values = EV_MODELS[st.session_state.current_ev_model].copy()
    
    # Initialize ev_models_parameters if not exists
    if ui.ev_models_parameters is None:
        ui.ev_models_parameters = {}
        
    # Define callback function for model selection
    def on_model_change():
        selected = st.session_state.ev_model_selector
        if selected != st.session_state.current_ev_model:
            st.session_state.current_ev_model = selected
            st.session_state.current_ev_values = EV_MODELS[selected].copy()
            
            # Apply any existing overrides
            current_overrides = ui.ev_models_parameters
            if selected in current_overrides:
                st.session_state.current_ev_values.update(current_overrides[selected])
    
    selected_model = st.selectbox(
        "Select EV Model",
        list(EV_MODELS.keys()),
        format_func=lambda x: x.replace('_', ' ').title(),
        key="ev_model_selector",
        index=list(EV_MODELS.keys()).index(st.session_state.current_ev_model),
        on_change=on_model_change
    )
    
    # Always ensure session state is in sync with selected model
    if st.session_state.current_ev_model != selected_model:
        st.session_state.current_ev_model = selected_model
        st.session_state.current_ev_values = EV_MODELS[selected_model].copy()
        
        # Apply any existing overrides
        current_overrides = ui.ev_models_parameters
        if selected_model in current_overrides:
            st.session_state.current_ev_values.update(current_overrides[selected_model])
    
    # Get current values for the selected model
    current_values = st.session_state.current_ev_values
    
    # Show current model info
    st.info(f"**Current Model:** {selected_model.replace('_', ' ').title()}")
    st.write(f"**Battery:** {current_values.get('battery_capacity', 'N/A')} kWh | **Range:** {current_values.get('wltp_range', 'N/A')} km | **Efficiency:** {current_values.get('efficiency', 'N/A')} kWh/100km")
    
    # EV Model Parameters Form - SEPARATE from main config form
    with st.form(f"ev_model_form_{selected_model}"):
        col1, col2 = st.columns(2)
        with col1:
            battery_capacity = st.number_input(
                "Battery Capacity (kWh)",
                min_value=10.0,
                max_value=200.0,
                value=float(current_values['battery_capacity']),
                step=0.1
            )
            wltp_range = st.number_input(
                "WLTP Range (km)",
                min_value=100,
                max_value=1000,
                value=int(current_values['wltp_range'])
            )
            efficiency = st.number_input(
                "Efficiency (kWh/100km)",
                min_value=10.0,
                max_value=30.0,
                value=float(current_values['efficiency']),
                step=0.1
            )
            max_charging_speed = st.number_input(
                "Max Charging Speed (kW)",
                min_value=20,
                max_value=500,
                value=int(current_values['max_charging_speed'])
            )
        with col2:
            weight = st.number_input(
                "Weight (kg)",
                min_value=1000,
                max_value=3000,
                value=int(current_values['weight'])
            )
            drag_coefficient = st.number_input(
                "Drag Coefficient",
                min_value=0.1,
                max_value=0.5,
                value=float(current_values['drag_coefficient']),
                step=0.01
            )
            frontal_area = st.number_input(
                "Frontal Area (m²)",
                min_value=1.5,
                max_value=3.5,
                value=float(current_values['frontal_area']),
                step=0.01
            )
            heat_pump = st.checkbox(
                "Has Heat Pump",
                value=bool(current_values['heat_pump'])
            )
        
        # Submit button for EV model parameters
        ev_model_submitted = st.form_submit_button(f"💾 Save {selected_model.replace('_', ' ').title()} Parameters")
        
        if ev_model_submitted:
            # Store the modified values
            if selected_model not in ui.ev_models_parameters:
                ui.ev_models_parameters[selected_model] = {}
            
            ui.ev_models_parameters[selected_model].update({
                'battery_capacity': battery_capacity,
                'wltp_range': wltp_range,
                'efficiency': efficiency,
                'max_charging_speed': max_charging_speed,
                'weight': weight,
                'drag_coefficient': drag_coefficient,
                'frontal_area': frontal_area,
                'heat_pump': heat_pump
            })
            
            # Save immediately
            save_overrides(ui)
            st.success(f"✅ {selected_model.replace('_', ' ').title()} parameters saved!")

# EV Models Market Share Section - OUTSIDE the main form
st.markdown("""
<div style="background: rgba(156, 39, 176, 0.1); padding: 1rem; border-radius: 12px; border-left: 4px solid #9c27b0; margin: 1rem 0;">
    <h3>📊 EV Models Market Share</h3>
    <p>Adjust EV model market shares for realistic Bay Area fleet simulation.</p>
</div>
""", unsafe_allow_html=True)

with st.expander("📊 EV Models Market Share (must sum to 1.0)", expanded=False):
    model_keys = list(EV_MODELS.keys())
    current_shares = ui.ev_models_market_share or {}
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

# Driver Profiles Proportion Section - OUTSIDE the main form
with st.expander("👥 Driver Profiles Proportion (must sum to 1.0)", expanded=False):
    prof_keys = list(DRIVER_PROFILES.keys())
    current_props = ui.driver_profiles_proportion or {}
    sliders_profiles = {}
    
    for pk in prof_keys:
        default = float(current_props.get(pk, DRIVER_PROFILES[pk].get('proportion', 0.0)))
        sliders_profiles[pk] = st.slider(f"{pk.replace('_', ' ').title()}", 0.0, 1.0, default, 0.01)
    
    total_prof = sum(sliders_profiles.values())
    st.metric("Total Proportion", f"{total_prof:.2f}")
    if abs(total_prof - 1.0) > 1e-6:
        st.warning("⚠️ Total proportion should be 1.0 — values will be normalized on save.")

st.divider()

# Runtime Configuration Display
st.markdown("""
<div style="background: rgba(76, 175, 80, 0.1); padding: 1rem; border-radius: 12px; border-left: 4px solid #4caf50; margin: 1rem 0;">
    <h3>🔧 Effective Runtime Configuration</h3>
    <p>Current merged configuration (UI overrides + Python defaults):</p>
</div>
""", unsafe_allow_html=True)

st.json(merged_runtime_config())

