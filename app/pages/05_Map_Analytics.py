import os, sys, json
from pathlib import Path

import pandas as pd
import streamlit as st

# Add project root to path for imports
CURRENT_DIR = os.path.dirname(__file__)
APP_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, '..'))
for p in [APP_DIR, PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

st.set_page_config(page_title="Map & Analytics", page_icon="🗺️", layout="wide")

st.title("🗺️ Map & Analytics — SF Bay Area")
st.caption("Golden Gate • East Bay • South Bay • Peninsula • North Bay")

# Check for required data
data_dir = Path("data/synthetic")
routes_path = data_dir / "routes.csv"
segments_path = data_dir / "segments.csv"

# Data availability check
col1, col2, col3 = st.columns(3)
with col1:
    routes_available = routes_path.exists()
    st.metric("Routes Data", "✅ Available" if routes_available else "❌ Missing", 
              help="GPS route traces for visualization")
    
with col2:
    segments_available = segments_path.exists()
    st.metric("Segments Data", "✅ Available" if segments_available else "❌ Missing",
              help="Individual road segments with energy data")
    
with col3:
    stations_available = (Path("data/analysis_results/real_stations.parquet").exists() or 
                         Path("data/analysis_results/mock_stations.parquet").exists())
    st.metric("Charging Stations", "✅ Available" if stations_available else "❌ Missing",
              help="Charging infrastructure locations")

if not routes_available:
    st.warning("""
    📊 **No routes data found yet!**
    
    To see the map visualization:
    1. Go to **Data Generation** page
    2. Generate synthetic data (this creates routes.csv)
    3. Come back to this page
    """)
    st.stop()

# Load data with progress indicators
with st.spinner("Loading route data..."):
    try:
        routes = pd.read_csv(routes_path)
        st.success(f"✅ Loaded {len(routes):,} routes")
    except Exception as e:
        st.error(f"❌ Failed to load routes: {e}")
        st.stop()

# Show data overview
st.subheader("📊 Data Overview")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Routes", f"{len(routes):,}")
    
with col2:
    if "vehicle_id" in routes.columns:
        unique_vehicles = routes["vehicle_id"].nunique()
        st.metric("Unique Vehicles", f"{unique_vehicles:,}")
    else:
        st.metric("Unique Vehicles", "N/A")

with col3:
    if "date" in routes.columns:
        unique_dates = routes["date"].nunique()
        st.metric("Simulation Days", f"{unique_dates:,}")
    else:
        st.metric("Simulation Days", "N/A")

with col4:
    if "total_distance_km" in routes.columns:
        total_distance = routes["total_distance_km"].sum()
        st.metric("Total Distance", f"{total_distance:,.0f} km")
    else:
        st.metric("Total Distance", "N/A")

# Show sample data
st.subheader("🔍 Sample Route Data")
if len(routes) > 0:
    st.dataframe(routes.head(10), use_container_width=True)

# Map visualization (simplified for now)
st.subheader("🗺️ Route Visualization")
st.info("""
**Map Features Coming Soon:**
- Interactive route visualization with PyDeck
- Energy consumption heatmaps
- Charging station locations
- Real-time fleet tracking

For now, use the data overview above to analyze your routes.
""")

# Analytics section
st.subheader("📈 Route Analytics")
if len(routes) > 0:
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Route Distribution by Date**")
        if "date" in routes.columns:
            date_counts = routes["date"].value_counts().sort_index()
            st.line_chart(date_counts)
        else:
            st.write("No date information available")
    
    with col2:
        st.write("**Distance Distribution**")
        if "total_distance_km" in routes.columns:
            # Use bar_chart instead of histogram_chart (which doesn't exist)
            distance_counts = routes["total_distance_km"].value_counts(bins=10).sort_index()
            st.bar_chart(distance_counts)
        else:
            st.write("No distance information available")

# Quick actions
st.subheader("🚀 Quick Actions")
col1, col2 = st.columns(2)

with col1:
    if st.button("🔄 Refresh Data", help="Reload data from files"):
        st.rerun()

with col2:
    if st.button("📊 Export Summary", help="Download route summary as CSV"):
        if len(routes) > 0:
            csv = routes.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="routes_summary.csv",
                mime="text/csv"
            )

