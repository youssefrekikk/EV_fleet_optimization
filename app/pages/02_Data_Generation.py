import os, sys
import streamlit as st
import time
import asyncio
import json
import pandas as pd
from datetime import datetime
from pathlib import Path

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
from services.data_service import data_service
from services.cache_manager import cache_manager
from services.async_task_service import async_task_service, submit_data_generation_task
from services.data_analytics_service import data_analytics_service

st.set_page_config(page_title="Data Generation", page_icon="🧪", layout="wide")

st.title("🧪 Data Generation Studio")
st.caption("Build the road network and generate synthetic datasets with high-performance processing")

# Performance indicator
st.markdown("""
<div style="background: rgba(0, 255, 166, 0.1); padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
    <h4>⚡ High-Performance Architecture</h4>
    <p>This page uses advanced caching, chunked data processing, and async task management for optimal performance.</p>
</div>
""", unsafe_allow_html=True)

cfg = merged_runtime_config()

# Get cached dataset stats and analytics
@st.cache_data(ttl=300)
def get_dataset_stats():
    return data_service.get_dataset_stats()

@st.cache_data(ttl=300)
def get_analytics_data():
    return {
        'summary': data_analytics_service.get_dataset_summary(),
        'fleet_analytics': data_analytics_service.get_fleet_analytics(),
        'quality_report': data_analytics_service.get_data_quality_report(),
        'visualizations': data_analytics_service.create_visualizations(),
        'schema': data_analytics_service.get_dataset_schema()
    }

dataset_stats = get_dataset_stats()
analytics_data = get_analytics_data()

col1, col2 = st.columns(2)

with col1:
    st.subheader("🛣️ Road Network")
    
    # Check if network already exists
    network_file = Path("data/networks/bay_area_network.pkl.gz")
    if network_file.exists():
        st.success("✅ Bay Area network already exists")
        st.write(f"File size: {network_file.stat().st_size / 1024 / 1024:.1f} MB")
    else:
        st.info("📁 Network file not found")
    
    if st.button("Build / Load Bay Area Network", use_container_width=True, type="primary"):
        with st.spinner("Building network..."):
            try:
                # Start network build task
                task_id = f"network_build_{int(time.time())}"
                task_manager.start_task(
                    task_id=task_id,
                    task_type="network_build",
                    params={"description": "Building Bay Area road network"}
                )
                st.success(f"Network build started! Task ID: {task_id}")
                st.rerun()
            except Exception as e:
                st.error(f"Error starting network build: {e}")

with col2:
    st.subheader("📊 Synthetic Data")
    
    # Show current dataset info
    if "error" not in dataset_stats:
        st.write(f"**Current Dataset:** {dataset_stats.get('total_files', 0)} files, {dataset_stats.get('total_size_mb', 0):.1f} MB")
        st.write(f"**Fleet Size:** {dataset_stats.get('fleet_size', 0)} vehicles")
    else:
        st.warning("No dataset found")
    
    # Configuration
    fleet_size = cfg["fleet"]["fleet_size"]
    sim_days = cfg["fleet"]["simulation_days"]
    
    st.write("**Configuration:**")
    st.write(f"- Fleet Size: {fleet_size}")
    st.write(f"- Simulation Days: {sim_days}")
    st.write("- Using market shares and driver proportions from UI overrides if set.")
    
    if st.button(f"Generate Data (fleet={fleet_size}, days={sim_days})", use_container_width=True, type="primary"):
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

# Task Status with Enhanced UI
st.subheader("📈 Task Status")

# Get both old and new task systems
running_tasks = task_manager.get_running_tasks()
async_tasks = async_task_service.get_running_tasks()

if running_tasks or async_tasks:
    # Show running tasks
    for task_id, task_info in running_tasks.items():
        with st.expander(f"🔄 Task {task_id} - {task_info.get('type', 'Unknown')}", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**Status:** {task_info.get('status', 'Unknown')}")
                st.write(f"**Started:** {task_info.get('started_at', 'Unknown')[:19]}")
            
            with col2:
                progress = float(task_info.get('progress', 0))
                st.progress(progress / 100)
                st.write(f"**Progress:** {progress}%")
                if task_info.get('message'):
                    st.write(f"**Message:** {task_info.get('message')}")
            
            with col3:
                if task_info.get('params'):
                    st.write("**Parameters:**")
                    st.json(task_info.get('params', {}))
    
    # Show async tasks
    for task_id, task_result in async_tasks.items():
        with st.expander(f"⚡ Async Task {task_id} - {task_result.metadata.get('type', 'Unknown')}", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**Status:** {task_result.status.value}")
                st.write(f"**Started:** {task_result.started_at.strftime('%Y-%m-%d %H:%M:%S')}")
            
            with col2:
                progress = float(task_result.progress.percentage)
                st.progress(progress / 100)
                st.write(f"**Progress:** {progress:.1f}%")
                st.write(f"**Message:** {task_result.progress.message}")
            
            with col3:
                if task_result.metadata:
                    st.write("**Metadata:**")
                    st.json(task_result.metadata)
else:
    st.info("No tasks currently running.")

# Performance Metrics
with st.expander("⚡ Performance Metrics", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Cache Performance**")
        cache_stats = cache_manager.get_stats()
        if "error" not in cache_stats:
            try:
                memory_usage = cache_stats.get('memory_usage_mb', 0)
                memory_limit = cache_stats.get('memory_limit_mb', 1024)
                
                # Handle pandas Series
                if hasattr(memory_usage, 'sum'):
                    memory_usage = float(memory_usage.sum())
                else:
                    memory_usage = float(memory_usage)
                    
                if hasattr(memory_limit, 'sum'):
                    memory_limit = float(memory_limit.sum())
                else:
                    memory_limit = float(memory_limit)
                    
                cache_percent = (memory_usage / memory_limit) * 100
                
                st.progress(float(cache_percent) / 100)
            except Exception as e:
                st.error(f"Cache calculation error: {str(e)}")
                st.progress(0.0)
                memory_usage = 0.0
                memory_limit = 1024.0
                cache_percent = 0.0
            
            st.write(f"Memory: {memory_usage:.1f} MB / {memory_limit} MB ({cache_percent:.1f}%)")
            st.write(f"Items: {cache_stats.get('total_items', 0)}")
            st.write(f"Files: {cache_stats.get('disk_files', 0)}")
        else:
            st.error(f"Cache Error: {cache_stats['error']}")
    
    with col2:
        st.markdown("**Data Performance**")
        if "error" not in dataset_stats:
            st.write(f"Files: {dataset_stats.get('total_files', 0)}")
            st.write(f"Size: {dataset_stats.get('total_size_mb', 0):.1f} MB")
            st.write(f"Dates: {len(dataset_stats.get('available_dates', []))}")
            st.write(f"Fleet: {dataset_stats.get('fleet_size', 0)}")
        else:
            st.error(f"Data Error: {dataset_stats['error']}")

# Quick Actions
st.subheader("🚀 Quick Actions")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔄 Refresh Cache", use_container_width=True):
        cache_manager.clear()
        st.success("Cache cleared!")
        st.rerun()

with col2:
    if st.button("📊 Update Stats", use_container_width=True):
        # Clear cache to force refresh
        st.cache_data.clear()
        st.success("Statistics updated!")
        st.rerun()

with col3:
    if st.button("🧹 Clear Old Tasks", use_container_width=True):
        task_manager.clear_completed_tasks()
        async_task_service.clear_completed_tasks()
        st.success("Old tasks cleared!")
        st.rerun()

# Add analytics refresh button
col4, col5, col8 = st.columns(3)
with col4:
    if st.button("📊 Refresh Analytics", use_container_width=True):
        # Clear analytics cache to force refresh
        st.cache_data.clear()
        st.success("Analytics refreshed!")
        st.rerun()

with col5:
    if st.button("🎯 Refresh KPIs", use_container_width=True):
        # Clear analytics cache to force refresh
        st.cache_data.clear()
        st.success("KPIs refreshed!")
        st.rerun()

with col8:
    if st.button("📋 Export Analytics Report", use_container_width=True):
        # Create a comprehensive report with KPIs
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'dataset_summary': analytics_data['summary'],
            'fleet_analytics': analytics_data['fleet_analytics'],
            'quality_report': analytics_data['quality_report']
        }
        
        # Add KPIs if available
        if 'error' not in analytics_data['fleet_analytics']:
            fleet_analytics = analytics_data['fleet_analytics']
            
            # Calculate KPIs
            total_energy_kwh = 0
            total_cost_usd = 0
            total_distance_km = 0
            total_trips = 0
            total_charging_sessions = 0
            avg_efficiency = 0
            fleet_size = 0
            
            if 'vehicle_performance' in fleet_analytics:
                perf = fleet_analytics['vehicle_performance']
                total_energy_kwh = perf.get('total_consumption_kwh', 0)
                total_distance_km = perf.get('total_distance_km', 0)
                total_trips = perf.get('total_trips', 0)
                total_charging_sessions = perf.get('total_charging_sessions', 0)
                avg_efficiency = perf.get('avg_efficiency_kwh_per_100km', 0)
            
            if 'charging_analysis' in fleet_analytics:
                charging = fleet_analytics['charging_analysis']
                total_cost_usd = charging.get('total_cost_usd', 0)
            
            if 'fleet_composition' in fleet_analytics:
                comp = fleet_analytics['fleet_composition']
                fleet_size = comp.get('total_vehicles', 0)
            
            # Add KPIs to report
            report_data['kpis'] = {
                'total_energy_kwh': total_energy_kwh,
                'total_cost_usd': total_cost_usd,
                'total_distance_km': total_distance_km,
                'total_trips': total_trips,
                'total_charging_sessions': total_charging_sessions,
                'avg_efficiency_kwh_per_100km': avg_efficiency,
                'fleet_size': fleet_size,
                'avg_energy_per_trip': total_energy_kwh / max(total_trips, 1),
                'avg_cost_per_trip': total_cost_usd / max(total_trips, 1),
                'avg_distance_per_trip': total_distance_km / max(total_trips, 1),
                'energy_cost_per_kwh': total_cost_usd / max(total_energy_kwh, 1)
            }
        
        # Save report
        report_file = f"data/analysis_results/analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs(os.path.dirname(report_file), exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        st.success(f"Analytics report with KPIs exported to: {report_file}")

# Add KPI export button
col6, col7 = st.columns(2)
with col6:
    if st.button("🎯 Export KPIs Only", use_container_width=True):
        if 'error' not in analytics_data['fleet_analytics']:
            # Export just the KPIs
            fleet_analytics = analytics_data['fleet_analytics']
            
            # Calculate KPIs
            total_energy_kwh = 0
            total_cost_usd = 0
            total_distance_km = 0
            total_trips = 0
            total_charging_sessions = 0
            avg_efficiency = 0
            fleet_size = 0
            
            if 'vehicle_performance' in fleet_analytics:
                perf = fleet_analytics['vehicle_performance']
                total_energy_kwh = perf.get('total_consumption_kwh', 0)
                total_distance_km = perf.get('total_distance_km', 0)
                total_trips = perf.get('total_trips', 0)
                total_charging_sessions = perf.get('total_charging_sessions', 0)
                avg_efficiency = perf.get('avg_efficiency_kwh_per_100km', 0)
            
            if 'charging_analysis' in fleet_analytics:
                charging = fleet_analytics['charging_analysis']
                total_cost_usd = charging.get('total_cost_usd', 0)
            
            if 'fleet_composition' in fleet_analytics:
                comp = fleet_analytics['fleet_composition']
                fleet_size = comp.get('total_vehicles', 0)
            
            kpi_data = {
                'timestamp': datetime.now().isoformat(),
                'kpis': {
                    'total_energy_kwh': total_energy_kwh,
                    'total_cost_usd': total_cost_usd,
                    'total_distance_km': total_distance_km,
                    'total_trips': total_trips,
                    'total_charging_sessions': total_charging_sessions,
                    'avg_efficiency_kwh_per_100km': avg_efficiency,
                    'fleet_size': fleet_size,
                    'avg_energy_per_trip': total_energy_kwh / max(total_trips, 1),
                    'avg_cost_per_trip': total_cost_usd / max(total_trips, 1),
                    'avg_distance_per_trip': total_distance_km / max(total_trips, 1),
                    'energy_cost_per_kwh': total_cost_usd / max(total_energy_kwh, 1)
                }
            }
            
            # Save KPI report
            kpi_file = f"data/analysis_results/kpi_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            os.makedirs(os.path.dirname(kpi_file), exist_ok=True)
            
            with open(kpi_file, 'w') as f:
                json.dump(kpi_data, f, indent=2, default=str)
            
            st.success(f"KPI report exported to: {kpi_file}")
        else:
            st.error("No KPI data available to export")

with col7:
    if st.button("📊 Export to CSV", use_container_width=True):
        if 'error' not in analytics_data['fleet_analytics']:
            # Export KPIs to CSV
            fleet_analytics = analytics_data['fleet_analytics']
            
            # Calculate KPIs
            total_energy_kwh = 0
            total_cost_usd = 0
            total_distance_km = 0
            total_trips = 0
            total_charging_sessions = 0
            avg_efficiency = 0
            fleet_size = 0
            
            if 'vehicle_performance' in fleet_analytics:
                perf = fleet_analytics['vehicle_performance']
                total_energy_kwh = perf.get('total_consumption_kwh', 0)
                total_distance_km = perf.get('total_distance_km', 0)
                total_trips = perf.get('total_trips', 0)
                total_charging_sessions = perf.get('total_charging_sessions', 0)
                avg_efficiency = perf.get('avg_efficiency_kwh_per_100km', 0)
            
            if 'charging_analysis' in fleet_analytics:
                charging = fleet_analytics['charging_analysis']
                total_cost_usd = charging.get('total_cost_usd', 0)
            
            if 'fleet_composition' in fleet_analytics:
                comp = fleet_analytics['fleet_composition']
                fleet_size = comp.get('total_vehicles', 0)
            
            # Create DataFrame for CSV export
            kpi_df = pd.DataFrame({
                'Metric': [
                    'Total Energy (kWh)', 'Total Cost (USD)', 'Total Distance (km)',
                    'Total Trips', 'Total Charging Sessions', 'Fleet Size',
                    'Avg Efficiency (kWh/100km)', 'Avg Energy per Trip (kWh)',
                    'Avg Cost per Trip (USD)', 'Avg Distance per Trip (km)',
                    'Energy Cost Rate (USD/kWh)'
                ],
                'Value': [
                    total_energy_kwh, total_cost_usd, total_distance_km,
                    total_trips, total_charging_sessions, fleet_size,
                    avg_efficiency, total_energy_kwh / max(total_trips, 1),
                    total_cost_usd / max(total_trips, 1), total_distance_km / max(total_trips, 1),
                    total_cost_usd / max(total_energy_kwh, 1)
                ]
            })
            
            # Save CSV
            csv_file = f"data/analysis_results/kpi_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            os.makedirs(os.path.dirname(csv_file), exist_ok=True)
            
            kpi_df.to_csv(csv_file, index=False)
            st.success(f"KPI CSV exported to: {csv_file}")
        else:
            st.error("No KPI data available to export")

# Key Performance Indicators (KPIs)
st.markdown("---")
st.subheader("🎯 Key Performance Indicators (KPIs)")

# Calculate KPIs from analytics data
if 'error' not in analytics_data['fleet_analytics']:
    fleet_analytics = analytics_data['fleet_analytics']
    
    # Initialize KPI values
    total_energy_kwh = 0
    total_cost_usd = 0
    total_distance_km = 0
    total_trips = 0
    total_charging_sessions = 0
    avg_efficiency = 0
    fleet_size = 0
    
    # Extract values from analytics
    if 'vehicle_performance' in fleet_analytics:
        perf = fleet_analytics['vehicle_performance']
        total_energy_kwh = perf.get('total_consumption_kwh', 0)
        total_distance_km = perf.get('total_distance_km', 0)
        total_trips = perf.get('total_trips', 0)
        total_charging_sessions = perf.get('total_charging_sessions', 0)
        avg_efficiency = perf.get('avg_efficiency_kwh_per_100km', 0)
    
    if 'charging_analysis' in fleet_analytics:
        charging = fleet_analytics['charging_analysis']
        total_cost_usd = charging.get('total_cost_usd', 0)
    
    if 'fleet_composition' in fleet_analytics:
        comp = fleet_analytics['fleet_composition']
        fleet_size = comp.get('total_vehicles', 0)
    
    # Calculate derived KPIs
    avg_energy_per_trip = total_energy_kwh / max(total_trips, 1)
    avg_cost_per_trip = total_cost_usd / max(total_trips, 1)
    avg_distance_per_trip = total_distance_km / max(total_trips, 1)
    energy_cost_per_kwh = total_cost_usd / max(total_energy_kwh, 1)
    
    # Display KPIs in a grid
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "⚡ Total Energy Consumed",
            f"{total_energy_kwh:,.1f} kWh",
            help="Total energy consumption across all vehicles and trips"
        )
        st.metric(
            "💰 Total Cost",
            f"${total_cost_usd:,.2f}",
            help="Total cost of all charging sessions"
        )
    
    with col2:
        st.metric(
            "🛣️ Total Distance",
            f"{total_distance_km:,.0f} km",
            help="Total distance traveled by all vehicles"
        )
        st.metric(
            "🚗 Fleet Size",
            f"{fleet_size:,}",
            help="Total number of vehicles in the fleet"
        )
    
    with col3:
        st.metric(
            "📊 Total Trips",
            f"{total_trips:,}",
            help="Total number of trips completed"
        )
        st.metric(
            "🔌 Charging Sessions",
            f"{total_charging_sessions:,}",
            help="Total number of charging sessions"
        )
    
    with col4:
        st.metric(
            "⚡ Avg Efficiency",
            f"{avg_efficiency:.1f} kWh/100km",
            help="Average energy efficiency across all vehicles"
        )
        st.metric(
            "📈 Avg Energy/Trip",
            f"{avg_energy_per_trip:.2f} kWh",
            help="Average energy consumption per trip"
        )
    
    # Additional KPI rows
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "💵 Avg Cost/Trip",
            f"${avg_cost_per_trip:.2f}",
            help="Average cost per trip"
        )
    
    with col2:
        st.metric(
            "🛣️ Avg Distance/Trip",
            f"{avg_distance_per_trip:.1f} km",
            help="Average distance per trip"
        )
    
    with col3:
        st.metric(
            "⚡ Energy Cost Rate",
            f"${energy_cost_per_kwh:.3f}/kWh",
            help="Average cost per kWh of energy"
        )
    
    with col4:
        if total_distance_km > 0:
            trips_per_km = total_trips / total_distance_km
            st.metric(
                "📊 Trip Density",
                f"{trips_per_km:.3f} trips/km",
                help="Trips per kilometer traveled"
            )
        else:
            st.metric("📊 Trip Density", "N/A")
    
    # KPI Summary Cards
    st.markdown("---")
    st.subheader("📋 KPI Summary Cards")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
            <h3>⚡ Energy Metrics</h3>
            <p><strong>Total:</strong> {:.1f} kWh</p>
            <p><strong>Per Trip:</strong> {:.2f} kWh</p>
            <p><strong>Efficiency:</strong> {:.1f} kWh/100km</p>
        </div>
        """.format(total_energy_kwh, avg_energy_per_trip, avg_efficiency), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
            <h3>💰 Financial Metrics</h3>
            <p><strong>Total Cost:</strong> ${:.2f}</p>
            <p><strong>Per Trip:</strong> ${:.2f}</p>
            <p><strong>Per kWh:</strong> ${:.3f}</p>
        </div>
        """.format(total_cost_usd, avg_cost_per_trip, energy_cost_per_kwh), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
            <h3>🚗 Operational Metrics</h3>
            <p><strong>Fleet Size:</strong> {:,}</p>
            <p><strong>Total Trips:</strong> {:,}</p>
            <p><strong>Avg Distance:</strong> {:.1f} km</p>
        </div>
        """.format(fleet_size, total_trips, avg_distance_per_trip), unsafe_allow_html=True)
    
    # KPI Trends (if date data is available)
    if 'visualizations' in analytics_data and 'daily_activity' in analytics_data['visualizations']:
        st.markdown("---")
        st.subheader("📈 KPI Trends Over Time")
        st.plotly_chart(analytics_data['visualizations']['daily_activity'], use_container_width=True, key="kpi_trends_chart")
        st.caption("Daily trends for trips, distance, and energy consumption")
    
    # Additional KPI Visualizations
    st.markdown("---")
    st.subheader("📊 KPI Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Energy vs Cost Scatter
        if total_energy_kwh > 0 and total_cost_usd > 0:
            # Create a simple scatter plot showing energy vs cost relationship
            import plotly.graph_objects as go
            
            fig_energy_cost = go.Figure()
            fig_energy_cost.add_trace(go.Scatter(
                x=[total_energy_kwh],
                y=[total_cost_usd],
                mode='markers',
                marker=dict(size=20, color='red'),
                name='Current Fleet',
                text=[f'Energy: {total_energy_kwh:.1f} kWh<br>Cost: ${total_cost_usd:.2f}'],
                hovertemplate='<b>%{text}</b><extra></extra>'
            ))
            
            fig_energy_cost.update_layout(
                title='Energy Consumption vs Cost',
                xaxis_title='Total Energy (kWh)',
                yaxis_title='Total Cost (USD)',
                height=400
            )
            
            st.plotly_chart(fig_energy_cost, use_container_width=True, key="energy_cost_scatter")
    
    with col2:
        # Efficiency Distribution
        if avg_efficiency > 0:
            # Create a gauge chart for efficiency
            fig_efficiency = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = avg_efficiency,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Average Efficiency (kWh/100km)"},
                delta = {'reference': 20},  # Reference value
                gauge = {
                    'axis': {'range': [None, 30]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 15], 'color': "lightgreen"},
                        {'range': [15, 25], 'color': "yellow"},
                        {'range': [25, 30], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 25
                    }
                }
            ))
            
            fig_efficiency.update_layout(height=400)
            st.plotly_chart(fig_efficiency, use_container_width=True, key="efficiency_gauge")
    
    # KPI Comparison Table
    st.markdown("---")
    st.subheader("📋 KPI Summary Table")
    
    kpi_summary = pd.DataFrame({
        'Category': ['Energy', 'Financial', 'Operational', 'Efficiency'],
        'Total Energy (kWh)': [f"{total_energy_kwh:,.1f}", '', '', ''],
        'Total Cost (USD)': ['', f"${total_cost_usd:,.2f}", '', ''],
        'Total Distance (km)': ['', '', f"{total_distance_km:,.0f}", ''],
        'Avg Efficiency (kWh/100km)': ['', '', '', f"{avg_efficiency:.1f}"],
        'Per Trip': [f"{total_energy_kwh / max(total_trips, 1):.2f} kWh", 
                     f"${total_cost_usd / max(total_trips, 1):.2f}", 
                     f"{total_distance_km / max(total_trips, 1):.1f} km", 
                     f"{avg_efficiency:.1f} kWh/100km"]
    })
    
    st.dataframe(kpi_summary, use_container_width=True)
    


else:
    st.error("❌ Unable to calculate KPIs - Fleet analytics data not available")
    st.info("Generate some data first to see KPIs")

# Comprehensive Data Analytics
st.markdown("---")
st.subheader("📊 Data Analytics & Insights")

# Data Quality Report
with st.expander("🔍 Data Quality Report", expanded=False):
    quality_report = analytics_data['quality_report']
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Overall Quality Score", f"{quality_report['overall_score']}/100")
    with col2:
        st.metric("Files Analyzed", quality_report['files_analyzed'])
    with col3:
        st.metric("Total Issues", quality_report['total_issues'])
    with col4:
        if quality_report['overall_score'] >= 90:
            st.success("✅ Excellent Quality")
        elif quality_report['overall_score'] >= 70:
            st.warning("⚠️ Good Quality")
        else:
            st.error("❌ Needs Attention")
    
    # Show file-specific quality reports
    for filename, report in quality_report['file_reports'].items():
        with st.expander(f"📄 {filename} (Score: {report['score']}/100)", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Size:** {report['file_size_mb']:.2f} MB")
                st.write(f"**Rows:** {report['total_rows']:,}")
            with col2:
                st.write(f"**Columns:** {report['total_columns']}")
                st.write(f"**Issues:** {len(report['issues'])}")
            with col3:
                if report['score'] >= 90:
                    st.success("✅ Excellent")
                elif report['score'] >= 70:
                    st.warning("⚠️ Good")
                else:
                    st.error("❌ Poor")
            
            if report['issues']:
                st.write("**Issues Found:**")
                for issue in report['issues']:
                    st.write(f"• {issue}")

# Dataset Summary
with st.expander("📈 Dataset Summary", expanded=False):
    summary = analytics_data['summary']
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Files", summary['total_files'])
    with col2:
        st.metric("Total Rows", f"{summary['total_rows']:,}")
    with col3:
        st.metric("Total Size", f"{summary['total_size_mb']:.2f} MB")
    with col4:
        if summary['total_files'] > 0:
            avg_rows = summary['total_rows'] / summary['total_files']
            st.metric("Avg Rows/File", f"{avg_rows:,.0f}")
    
    # Show detailed file information
    st.subheader("📋 File Details")
    for filename, file_info in summary['files_info'].items():
        if file_info.get('exists', False):
            with st.expander(f"📄 {filename}", expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.write(f"**Size:** {file_info['file_size_mb']:.2f} MB")
                    st.write(f"**Memory:** {file_info['memory_usage_mb']:.2f} MB")
                with col2:
                    st.write(f"**Rows:** {file_info['total_rows']:,}")
                    st.write(f"**Columns:** {file_info['total_columns']}")
                with col3:
                    st.write(f"**Modified:** {file_info['last_modified']}")
                with col4:
                    if 'error' in file_info:
                        st.error(f"Error: {file_info['error']}")
                    else:
                        st.success("✅ Valid")
                
                # Show columns
                if 'columns' in file_info:
                    st.write("**Columns:**")
                    cols_display = []
                    for col in file_info['columns']:
                        dtype = file_info['data_types'].get(col, 'unknown')
                        cols_display.append(f"`{col}` ({dtype})")
                    
                    # Display in a grid
                    col_count = 3
                    for i in range(0, len(cols_display), col_count):
                        cols = st.columns(col_count)
                        for j, col_display in enumerate(cols_display[i:i+col_count]):
                            with cols[j]:
                                st.code(col_display, language=None)

# Fleet Analytics
with st.expander("🚗 Fleet Analytics", expanded=False):
    fleet_analytics = analytics_data['fleet_analytics']
    
    if 'error' not in fleet_analytics:
        # Fleet Composition
        if 'fleet_composition' in fleet_analytics:
            st.subheader("🏗️ Fleet Composition")
            comp = fleet_analytics['fleet_composition']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Vehicles", comp['total_vehicles'])
            with col2:
                home_charging = comp['home_charging_availability']
                st.metric("Home Charging", f"{home_charging['with_home_charging']}/{comp['total_vehicles']}")
            with col3:
                st.metric("Vehicle Models", len(comp['vehicle_models']))
            with col4:
                st.metric("Driver Profiles", len(comp['driver_profiles']))
            
            # Vehicle models breakdown
            if comp['vehicle_models']:
                st.write("**Vehicle Models:**")
                model_df = pd.DataFrame(list(comp['vehicle_models'].items()), 
                                      columns=['Model', 'Count'])
                st.dataframe(model_df, use_container_width=True)
        
        # Vehicle Performance
        if 'vehicle_performance' in fleet_analytics:
            st.subheader("⚡ Vehicle Performance")
            perf = fleet_analytics['vehicle_performance']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Distance", f"{perf['total_distance_km']:,.0f} km")
            with col2:
                st.metric("Total Consumption", f"{perf['total_consumption_kwh']:,.1f} kWh")
            with col3:
                st.metric("Avg Efficiency", f"{perf['avg_efficiency_kwh_per_100km']:.1f} kWh/100km")
            with col4:
                st.metric("Total Trips", f"{perf['total_trips']:,}")
        
        # Route Analysis
        if 'route_analysis' in fleet_analytics:
            st.subheader("🛣️ Route Analysis")
            route = fleet_analytics['route_analysis']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Routes", f"{route['total_routes']:,}")
            with col2:
                st.metric("Avg Distance", f"{route['avg_distance_km']:.1f} km")
            with col3:
                st.metric("Avg Duration", f"{route['avg_duration_minutes']:.1f} min")
            with col4:
                st.metric("Avg Consumption", f"{route['avg_consumption_kwh']:.2f} kWh")
            
            # Distance distribution
            if route['distance_distribution']:
                st.write("**Distance Distribution:**")
                dist_df = pd.DataFrame(list(route['distance_distribution'].items()),
                                     columns=['Trip Type', 'Count'])
                st.dataframe(dist_df, use_container_width=True)
        
        # Charging Analysis
        if 'charging_analysis' in fleet_analytics:
            st.subheader("🔌 Charging Analysis")
            charging = fleet_analytics['charging_analysis']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Sessions", f"{charging['total_sessions']:,}")
            with col2:
                st.metric("Home Charging", f"{charging['home_charging_sessions']:,}")
            with col3:
                st.metric("Total Energy", f"{charging['total_energy_delivered_kwh']:,.1f} kWh")
            with col4:
                st.metric("Total Cost", f"${charging['total_cost_usd']:,.2f}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg Session Duration", f"{charging['avg_session_duration_minutes']:.1f} min")
            with col2:
                st.metric("Avg Energy/Session", f"{charging['avg_energy_per_session_kwh']:.2f} kWh")
            with col3:
                st.metric("Avg Cost/Session", f"${charging['avg_cost_per_session_usd']:.2f}")
    else:
        st.error(f"Error loading fleet analytics: {fleet_analytics['error']}")

# Visualizations
with st.expander("📊 Data Visualizations", expanded=False):
    viz_data = analytics_data['visualizations']
    
    if 'error' not in viz_data:
        # Fleet Composition
        if 'fleet_models' in viz_data:
            st.subheader("🚗 Fleet Composition by Vehicle Model")
            st.plotly_chart(viz_data['fleet_models'], use_container_width=True)
        
        if 'driver_profiles' in viz_data:
            st.subheader("👥 Driver Profile Distribution")
            st.plotly_chart(viz_data['driver_profiles'], use_container_width=True)
        
        # Performance Visualizations
        if 'efficiency_distribution' in viz_data:
            st.subheader("⚡ Vehicle Energy Efficiency Distribution")
            st.plotly_chart(viz_data['efficiency_distribution'], use_container_width=True)
        
        if 'distance_consumption_scatter' in viz_data:
            st.subheader("📈 Distance vs Energy Consumption")
            st.plotly_chart(viz_data['distance_consumption_scatter'], use_container_width=True)
        
        if 'charging_analysis' in viz_data:
            st.subheader("🔌 Charging Session Analysis")
            st.plotly_chart(viz_data['charging_analysis'], use_container_width=True)
        
        if 'daily_activity' in viz_data:
            st.subheader("📅 Daily Activity Patterns")
            st.plotly_chart(viz_data['daily_activity'], use_container_width=True)
    else:
        st.error(f"Error creating visualizations: {viz_data['error']}")

# Dataset Schema
with st.expander("📋 Dataset Schema", expanded=False):
    st.subheader("🔍 Dataset Column Information")
    
    schema_data = analytics_data['schema']
    
    for filename, schema_info in schema_data.items():
        with st.expander(f"📄 {filename}", expanded=False):
            if 'error' in schema_info:
                st.error(f"Error: {schema_info['error']}")
            elif 'columns' in schema_info:
                st.write(f"**Total Columns:** {len(schema_info['columns'])}")
                st.write("**Columns:**")
                
                # Display columns in a grid
                cols_display = []
                for col in schema_info['columns']:
                    dtype = schema_info['dtypes'].get(col, 'unknown')
                    cols_display.append(f"`{col}` ({dtype})")
                
                col_count = 3
                for i in range(0, len(cols_display), col_count):
                    cols = st.columns(col_count)
                    for j, col_display in enumerate(cols_display[i:i+col_count]):
                        with cols[j]:
                            st.code(col_display, language=None)
            else:
                st.warning("No schema information available")

# Column Analysis
with st.expander("🔍 Detailed Column Analysis", expanded=False):
    st.subheader("📋 Column-Level Analysis")
    
    # File selector
    available_files = [f for f, info in analytics_data['summary']['files_info'].items() 
                      if info.get('exists', False) and 'error' not in info]
    
    if available_files:
        selected_file = st.selectbox("Select file for detailed analysis:", available_files)
        
        if selected_file:
            column_analysis = data_analytics_service.get_column_analysis(selected_file)
            
            if 'error' not in column_analysis:
                st.write(f"**File:** {column_analysis['filename']}")
                st.write(f"**Total Rows:** {column_analysis['total_rows']:,}")
                st.write(f"**Total Columns:** {column_analysis['total_columns']}")
                
                # Show column analysis
                for col_name, analysis in column_analysis['column_analysis'].items():
                    with st.expander(f"📊 {col_name} ({analysis['dtype']})", expanded=False):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.write("**Basic Stats:**")
                            st.write(f"• Total: {analysis['total_count']:,}")
                            st.write(f"• Non-null: {analysis['non_null_count']:,}")
                            st.write(f"• Null: {analysis['null_count']:,}")
                            st.write(f"• Null %: {analysis['null_percentage']}%")
                        
                        with col2:
                            st.write("**Data Quality:**")
                            if analysis['null_percentage'] == 0:
                                st.success("✅ No null values")
                            elif analysis['null_percentage'] < 5:
                                st.info("ℹ️ Low null values")
                            elif analysis['null_percentage'] < 20:
                                st.warning("⚠️ Moderate null values")
                            else:
                                st.error("❌ High null values")
                        
                        with col3:
                            # Type-specific analysis
                            if analysis['dtype'] in ['int64', 'float64']:
                                st.write("**Numeric Stats:**")
                                if analysis.get('min') is not None:
                                    st.write(f"• Min: {analysis['min']:.2f}")
                                    st.write(f"• Max: {analysis['max']:.2f}")
                                    st.write(f"• Mean: {analysis['mean']:.2f}")
                                    st.write(f"• Std: {analysis['std']:.2f}")
                                st.write(f"• Unique: {analysis['unique_values']:,}")
                            
                            elif analysis['dtype'] == 'object':
                                st.write("**Text Stats:**")
                                st.write(f"• Unique: {analysis['unique_values']:,}")
                                st.write(f"• Avg Length: {analysis['avg_length']:.1f}")
                                if analysis.get('most_common'):
                                    st.write("**Most Common:**")
                                    for value, count in list(analysis['most_common'].items())[:3]:
                                        st.write(f"• {value}: {count}")
                            
                            elif analysis['dtype'] == 'bool':
                                st.write("**Boolean Stats:**")
                                st.write(f"• True: {analysis['true_count']:,}")
                                st.write(f"• False: {analysis['false_count']:,}")
                                st.write(f"• True %: {analysis['true_percentage']}%")
            else:
                st.error(f"Error analyzing {selected_file}: {column_analysis['error']}")
    else:
        st.info("No data files available for analysis")

# Debug Information
with st.expander("🐛 Debug Information", expanded=False):
    st.subheader("🔍 Troubleshooting Data Issues")
    
    # Show actual column names for key files
    st.write("**Key Files Column Names:**")
    
    for key_file in ['routes.csv', 'charging_sessions.csv', 'vehicle_states.csv', 'fleet_info.csv']:
        filepath = Path(f"data/synthetic/{key_file}")
        if filepath.exists():
            try:
                df_sample = pd.read_csv(filepath, nrows=1)
                st.write(f"**{key_file}:** {list(df_sample.columns)}")
            except Exception as e:
                st.error(f"Error reading {key_file}: {e}")
        else:
            st.warning(f"{key_file} not found")
    
    # Show analytics errors if any
    st.write("**Analytics Status:**")
    if 'error' in analytics_data['fleet_analytics']:
        st.error(f"Fleet Analytics Error: {analytics_data['fleet_analytics']['error']}")
    else:
        st.success("Fleet Analytics: ✅ Working")
    
    if 'error' in analytics_data['visualizations']:
        st.error(f"Visualizations Error: {analytics_data['visualizations']['error']}")
    else:
        st.success("Visualizations: ✅ Working")

# Tips and Information
st.info("""
💡 **Performance Tips:**
- Tasks run in the background - you can navigate to other pages while they're running
- Check the Home page for global status updates
- The system uses advanced caching to improve performance
- Large datasets are processed in chunks to avoid memory issues
- Use the performance metrics to monitor system health
- Analytics are cached for 5 minutes - refresh to see latest data
- Use the Debug Information section to troubleshoot data issues
""")

