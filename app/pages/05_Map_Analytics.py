import os, sys
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import pydeck as pdk
import folium
from streamlit_folium import st_folium

CURRENT_DIR = os.path.dirname(__file__)
APP_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, '..'))
for p in [APP_DIR, PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from services.data_analytics_service import data_analytics_service

st.set_page_config(page_title="Map Analytics", page_icon="🗺️", layout="wide")

st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, rgba(10,18,31,0.9), rgba(25,36,56,0.9)); border-radius: 20px; border: 2px solid rgba(0, 255, 166, 0.3); margin-bottom: 2rem;">
    <h1>🗺️ Bay Area EV Fleet Map Analytics</h1>
    <p style="color: #a0aec0; font-size: 1.1rem;">Interactive visualization of routes, charging stations, and fleet performance across the San Francisco Bay Area</p>
</div>
""", unsafe_allow_html=True)

# Check for required data
data_dir = Path("data/synthetic")
charging_infra_dir = Path("data/charging_infrastructure")

# Check for routes and fleet data
required_files = ["routes.csv", "fleet_info.csv"]
missing_files = [f for f in required_files if not (data_dir / f).exists()]

if missing_files:
    st.error(f"""
    ❌ **Missing required data files:**
    
    {', '.join(missing_files)}
    
    Please generate synthetic data first in the **Data Generation** page.
    """)
    st.stop()

# Check for charging station data with fallback options
charging_files = [
    data_dir / "charging_infrastructure.csv",
    data_dir / "real_charging_stations.csv", 
    charging_infra_dir / "real_stations_bay_area.csv",
    charging_infra_dir / "mock_stations_persistent.csv"
]

charging_file = None
for file_path in charging_files:
    if file_path.exists():
        charging_file = file_path
        break

if not charging_file:
    st.warning("""
    ⚠️ **No charging station data found.**
    
    Available options:
    - Generate synthetic data in **Data Generation** page
    - Use existing files: charging_infrastructure.csv, real_charging_stations.csv, real_stations_bay_area.csv, or mock_stations_persistent.csv
    
    Map will show routes and fleet data without charging stations.
    """)
    charging_file = None
else:
    st.success(f"✅ Using charging station data: {charging_file.name}")


# --- DATA SOURCES TRACE ---
st.info("🔎 Loading all data ONLY from: routes.csv, charging_sessions.csv, fleet_info.csv in data/synthetic/")

@st.cache_data(show_spinner=True)
def load_map_data():
    """Load and prepare data for map visualization."""
    try:
        # ROUTES: from routes.csv
        routes_df = pd.read_csv("data/synthetic/routes.csv")
        st.info(f"✅ Routes loaded from data/synthetic/routes.csv: {len(routes_df)} rows")
        # FLEET: from fleet_info.csv
        fleet_df = pd.read_csv("data/synthetic/fleet_info.csv")
        st.info(f"✅ Fleet info loaded from data/synthetic/fleet_info.csv: {len(fleet_df)} rows")
        # CHARGING SESSIONS: from charging_sessions.csv
        charging_sessions_df = pd.read_csv("data/synthetic/charging_sessions.csv")
        st.info(f"✅ Charging sessions loaded from data/synthetic/charging_sessions.csv: {len(charging_sessions_df)} rows")
        # Weather (optional, not used for map)
        weather_df = None
        if (data_dir / "weather.csv").exists():
            weather_df = pd.read_csv("data/synthetic/weather.csv")
        return routes_df, fleet_df, charging_sessions_df, weather_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None

routes_df, fleet_df, charging_sessions_df, weather_df = load_map_data()

if routes_df is None:
    st.stop()


# --- DATA PREPROCESSING TRACE ---
@st.cache_data(show_spinner=True)
def prepare_map_data(routes_df, fleet_df, charging_sessions_df):
    """Prepare data for map visualization."""
    # ROUTE COORDINATES: from routes.csv columns origin_lat, origin_lon, destination_lat, destination_lon
    st.info("🛣️ Route coordinates extracted from routes.csv: origin_lat, origin_lon, destination_lat, destination_lon")
    routes_df['start_lat'] = pd.to_numeric(routes_df['origin_lat'], errors='coerce')
    routes_df['start_lon'] = pd.to_numeric(routes_df['origin_lon'], errors='coerce')
    routes_df['end_lat'] = pd.to_numeric(routes_df['destination_lat'], errors='coerce')
    routes_df['end_lon'] = pd.to_numeric(routes_df['destination_lon'], errors='coerce')
    # Filter out invalid coordinates
    routes_df = routes_df.dropna(subset=['start_lat', 'start_lon', 'end_lat', 'end_lon'])

    # CHARGING STATIONS: extracted ONLY from charging_sessions.csv 'location' column
    st.info("🔌 Charging station coordinates extracted from charging_sessions.csv 'location' column")
    def extract_coordinates(location_str):
        try:
            coords = location_str.strip('()').split(', ')
            return float(coords[0]), float(coords[1])
        except:
            return None, None
    if 'location' in charging_sessions_df.columns:
        charging_sessions_df[['station_lat', 'station_lon']] = charging_sessions_df['location'].apply(
            lambda x: pd.Series(extract_coordinates(x))
        )
        charging_sessions_df = charging_sessions_df.dropna(subset=['station_lat', 'station_lon'])
        # Bay Area bounding box
        bay_area_bounds = {
            'lat_min': 37.0,
            'lat_max': 38.5,
            'lon_min': -123.0,
            'lon_max': -121.5
        }
        charging_sessions_df = charging_sessions_df[
            (charging_sessions_df['station_lat'] >= bay_area_bounds['lat_min']) &
            (charging_sessions_df['station_lat'] <= bay_area_bounds['lat_max']) &
            (charging_sessions_df['station_lon'] >= bay_area_bounds['lon_min']) &
            (charging_sessions_df['station_lon'] <= bay_area_bounds['lon_max'])
        ]
        # Aggregate by station_id
        station_usage = charging_sessions_df.groupby(['station_id', 'station_lat', 'station_lon']).agg({
            'session_id': 'count',
            'energy_delivered_kwh': 'sum',
            'station_operator': 'first'
        }).reset_index()
        st.info("🔋 Charging station usage and energy aggregated from charging_sessions.csv")
    else:
        station_usage = pd.DataFrame()

    # ENERGY: from routes.csv (total_consumption_kwh or energy_kwh)
    st.info("⚡ Route energy values taken from routes.csv: total_consumption_kwh or energy_kwh")
    route_lines = []
    for _, route in routes_df.iterrows():
        distance = route.get('distance_km', route.get('total_distance_km', 0))
        energy = route.get('energy_kwh', route.get('total_consumption_kwh', 0))
        # Filter: skip routes with distance == 0 or energy == 0 (allow very small nonzero values)
        if (
            pd.notna(route['start_lat']) and pd.notna(route['end_lat']) and
            distance is not None and energy is not None and
            distance > 1e-6 and energy > 1e-6
        ):
            route_lines.append({
                'start_lat': route['start_lat'],
                'start_lon': route['start_lon'],
                'end_lat': route['end_lat'],
                'end_lon': route['end_lon'],
                'energy_kwh': energy,
                'duration_minutes': route.get('duration_minutes', route.get('total_time_minutes', 0)),
                'distance_km': distance,
                'vehicle_id': route.get('vehicle_id', 'Unknown'),
                'trip_id': route.get('trip_id', 'Unknown')
            })
    # Ensure DataFrame always has expected columns, even if empty
    route_lines_df = pd.DataFrame(route_lines)
    expected_cols = ['start_lat', 'start_lon', 'end_lat', 'end_lon', 'energy_kwh', 'duration_minutes', 'distance_km', 'vehicle_id', 'trip_id']
    for col in expected_cols:
        if col not in route_lines_df.columns:
            route_lines_df[col] = np.nan
    return route_lines_df, station_usage

route_lines_df, station_usage = prepare_map_data(routes_df, fleet_df, charging_sessions_df)

# Map Configuration


st.subheader("🗺️ Interactive Bay Area Map (OpenStreetMap)")

# Show data source information
st.info(f"📊 **Data Source:** Using charging_sessions.csv with {len(station_usage)} unique charging stations")

# Map controls
col1, col2, col3 = st.columns(3)
with col1:
    map_center_lat = st.number_input("Map Center Latitude", value=37.7749, format="%.4f", help="Center of the map view")
    map_center_lon = st.number_input("Map Center Longitude", value=-122.4194, format="%.4f", help="Center of the map view")
with col2:
    zoom_level = st.slider("Zoom Level", min_value=8, max_value=15, value=10, help="Map zoom level")
with col3:
    show_heatmap = st.checkbox("Show Energy Consumption Heatmap", value=True, help="Display energy consumption as heatmap overlay")

# Layer detail control (for marker/line thickness)
layer_detail = st.slider("Layer Detail Level", min_value=1, max_value=10, value=3, help="Lower values show less detail (thicker lines, fewer points)")

# Layer controls
col1, col2, col3 = st.columns(3)
with col1:
    show_routes = st.checkbox("Show Routes", value=False, help="Display vehicle routes")
with col2:
    show_charging = st.checkbox("Show Charging Stations", value=False, help="Display charging infrastructure")
with col3:
    show_fleet = st.checkbox("Show Fleet Locations", value=False, help="Display current fleet positions")



# Vehicle Route Selector (for route visualization only)
vehicle_ids = route_lines_df['vehicle_id'].dropna().unique().tolist()
selected_vehicle = st.selectbox("Select Vehicle ID to show route", vehicle_ids)

# --- SEGMENT DATA LOADING ---
segment_file = "data/synthetic/segments.csv"
segment_cols = [
    "vehicle_id", "trip_id", "segment_id", "start_lat", "start_lon", "end_lat", "end_lon", "energy_kwh", "distance_m"
]
try:
    segments_df_full = pd.read_csv(segment_file, usecols=[
        "vehicle_id", "trip_id", "segment_id", "start_lat", "start_lon", "end_lat", "end_lon", "energy_kwh", "distance_m"
    ])
    segments_df_full = segments_df_full.dropna(subset=["start_lat", "start_lon", "end_lat", "end_lon", "energy_kwh", "distance_m"])
except Exception as e:
    segments_df_full = pd.DataFrame(columns=segment_cols)
    st.warning(f"Could not load segments.csv: {e}")

# Filter routes for selected vehicle (only for map visualization)
if show_routes:
    vehicle_routes_df = route_lines_df[route_lines_df['vehicle_id'] == selected_vehicle]
    segments_df = segments_df_full[segments_df_full["vehicle_id"] == selected_vehicle]
else:
    vehicle_routes_df = route_lines_df.copy()
    segments_df = segments_df_full.copy()

# --- RANDOM 10% ROUTE SAMPLING ---
if not vehicle_routes_df.empty:
    sample_frac = 0.1
    vehicle_routes_df = vehicle_routes_df.sample(frac=sample_frac, random_state=42)
    segments_df = segments_df.sample(frac=sample_frac, random_state=42)

# Create map layers

# Create Folium map
m = folium.Map(location=[map_center_lat, map_center_lon], zoom_start=zoom_level)

## Add real route segments for selected vehicle
if show_routes and not segments_df.empty:
    # Group by trip, plot segments as polylines
    for trip_id, trip_segments in segments_df.groupby("trip_id"):
        segment_points = trip_segments.sort_values("segment_id")
        # Build full route by connecting segment start/end points
        locations = []
        for _, seg in segment_points.iterrows():
            locations.append([seg["start_lat"], seg["start_lon"]])
        # Add last segment's end point
        if not segment_points.empty:
            last_seg = segment_points.iloc[-1]
            locations.append([last_seg["end_lat"], last_seg["end_lon"]])
        folium.PolyLine(
            locations=locations,
            color="#ff00a6",
            weight=layer_detail,
            opacity=0.7,
            tooltip=f"Vehicle: {selected_vehicle}<br>Trip: {trip_id}<br>Segments: {len(segment_points)}"
        ).add_to(m)
# Fallback: if no segments, show straight lines
elif show_routes and not vehicle_routes_df.empty:
    for _, route in vehicle_routes_df.iterrows():
        folium.PolyLine(
            locations=[
                [route['start_lat'], route['start_lon']],
                [route['end_lat'], route['end_lon']]
            ],
            color="#ff00a6",  # Neon pink
            weight=layer_detail,
            opacity=0.7,
            tooltip=f"Vehicle: {selected_vehicle}<br>Trip: {route['trip_id']}<br>Energy: {route['energy_kwh']:.2f} kWh<br>Distance: {route['distance_km']:.2f} km<br>Duration: {route['duration_minutes']:.1f} min"
        ).add_to(m)

# Add charging stations
if show_charging and not station_usage.empty:
    # Limit number of stations for performance
    max_stations = 500
    if len(station_usage) > max_stations:
        station_usage_sample = station_usage.sample(max_stations, random_state=42)
    else:
        station_usage_sample = station_usage
    # Color map for operators
    operators = station_usage_sample['station_operator'].unique()
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen']
    operator_colors = dict(zip(operators, colors[:len(operators)]))
    for _, station in station_usage_sample.iterrows():
        folium.CircleMarker(
            location=[station['station_lat'], station['station_lon']],
            radius=min(station['session_id'] / 2, 20),  # Size based on usage
            popup=f"Station ID: {station['station_id']}<br>Operator: {station['station_operator']}<br>Sessions: {station['session_id']}<br>Energy: {station['energy_delivered_kwh']:.1f} kWh",
            color=operator_colors.get(station['station_operator'], 'gray'),
            fill=True,
            fillOpacity=0.7
        ).add_to(m)

# Add fleet locations (end points)
if show_fleet and not vehicle_routes_df.empty:
    fleet_positions = vehicle_routes_df[['end_lat', 'end_lon', 'vehicle_id', 'energy_kwh']].copy()
    fleet_positions.columns = ['latitude', 'longitude', 'vehicle_id', 'energy_kwh']
    for _, fleet in fleet_positions.iterrows():
        folium.CircleMarker(
            location=[fleet['latitude'], fleet['longitude']],
            radius=layer_detail,
            color="#00d4ff",
            fill=True,
            fill_color="#00d4ff",
            fill_opacity=0.8,
            tooltip=f"Vehicle: {fleet['vehicle_id']}<br>Energy Used: {fleet['energy_kwh']:.2f} kWh"
        ).add_to(m)

# Energy consumption heatmap (midpoints)
if show_heatmap and not vehicle_routes_df.empty:
    from folium.plugins import HeatMap
    heatmap_data = []
    for _, route in vehicle_routes_df.iterrows():
        mid_lat = (route['start_lat'] + route['end_lat']) / 2
        mid_lon = (route['start_lon'] + route['end_lon']) / 2
        heatmap_data.append([mid_lat, mid_lon, route['energy_kwh']])
    if heatmap_data:
        HeatMap(heatmap_data, radius=layer_detail*2, blur=layer_detail*2, min_opacity=0.3).add_to(m)

st_folium(m, width=900, height=600)

# Map Statistics
st.markdown("---")
st.subheader("📊 Map Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Routes", len(vehicle_routes_df))
    
with col2:
    if not vehicle_routes_df.empty:
        avg_energy = vehicle_routes_df['energy_kwh'].mean()
        st.metric("Avg Energy per Route", f"{avg_energy:.2f} kWh")
    
with col3:
    if not vehicle_routes_df.empty:
        total_distance = vehicle_routes_df['distance_km'].sum()
        st.metric("Total Distance", f"{total_distance:.1f} km")
    
with col4:
    if not station_usage.empty:
        st.metric("Charging Stations", len(station_usage))

# Data Summary Tables
st.markdown("---")
st.subheader("📋 Data Summary")

col1, col2 = st.columns(2)

with col1:
    st.write("**Route Energy Distribution**")
    if not vehicle_routes_df.empty:
        energy_stats = vehicle_routes_df['energy_kwh'].describe()
        st.dataframe(energy_stats, use_container_width=True)
    
with col2:
    st.write("**Distance Distribution**")
    if not vehicle_routes_df.empty:
        distance_stats = vehicle_routes_df['distance_km'].describe()
        st.dataframe(distance_stats, use_container_width=True)

# Charging Station Analysis
if not station_usage.empty:
    st.markdown("---")
    st.subheader("🔌 Charging Station Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Operator Distribution**")
        if 'station_operator' in station_usage.columns:
            operator_counts = station_usage['station_operator'].value_counts()
            st.bar_chart(operator_counts)
        else:
            st.info("No operator information available")
    with col2:
        st.write("**Sessions per Station Distribution**")
        if 'session_id' in station_usage.columns:
            session_stats = station_usage['session_id'].describe()
            st.dataframe(session_stats, use_container_width=True)
        else:
            st.info("No session information available")
    # Show charging station data sample
    st.markdown("---")
    st.subheader("📋 Charging Station Data Sample")
    st.write(f"Showing first 10 of {len(station_usage)} charging stations:")
    st.dataframe(station_usage.head(10), use_container_width=True)
else:
    st.markdown("---")
    st.subheader("🔌 Charging Station Analysis")
    st.info("""
    **No charging station data available for analysis.**
    
    To see charging station analysis:
    1. Ensure you have charging_sessions.csv in data/synthetic/
    2. Or generate synthetic data in the **Data Generation** page
    """)

# Route Performance Analysis
if not vehicle_routes_df.empty:
    st.markdown("---")
    st.subheader("🚗 Route Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Energy vs Distance Correlation**")
        if len(vehicle_routes_df) > 1:
            correlation = vehicle_routes_df['energy_kwh'].corr(vehicle_routes_df['distance_km'])
            st.metric("Correlation", f"{correlation:.3f}")
            
            # Scatter plot
            chart_data = pd.DataFrame({
                'Distance (km)': vehicle_routes_df['distance_km'],
                'Energy (kWh)': vehicle_routes_df['energy_kwh']
            })
            st.line_chart(chart_data)
    
    with col2:
        st.write("**Efficiency Distribution**")
        if 'distance_km' in vehicle_routes_df.columns and 'energy_kwh' in vehicle_routes_df.columns:
            vehicle_routes_df['efficiency'] = vehicle_routes_df['energy_kwh'] / vehicle_routes_df['distance_km']
            efficiency_stats = vehicle_routes_df['efficiency'].describe()
            st.dataframe(efficiency_stats, use_container_width=True)

st.info("💡 **Tip:** Use the map controls to adjust the view, filters to focus on specific data ranges, and hover over map elements for detailed information.")

