import os, sys
import json
from pathlib import Path
import base64
import streamlit as st

# Add project root to path for imports
CURRENT_DIR = os.path.dirname(__file__)
APP_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, '..'))
for p in [APP_DIR, PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from services.task_manager import task_manager
from services.data_service import data_service
from services.cache_manager import cache_manager
from services.async_task_service import async_task_service


with open("app/pictures/San_Francisco.jpg", "rb") as f:
    img = base64.b64encode(f.read()).decode()
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background: url("data:image/jpg;base64,{img}") no-repeat center center fixed;
    background-size: cover;
    position: relative;
}}

[data-testid="stAppViewContainer"]::before {{
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: rgba(0,0,0,0.45); /* 👈 overlay darkness */
    z-index: 0;
}}

.block-container {{
    position: relative;
    z-index: 1; /* makes sure content is above overlay */
    color: #f5f5f5; /* soft white text */
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)



st.set_page_config(
    page_title="EV Fleet Optimization Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

image_path = os.path.join(CURRENT_DIR, 'pictures', 'San_Francisco.jpg')



# Custom CSS for futuristic Bay Area theme
css = """
<style>

/* Remove gradient text */
.hero h1 {
    background: none !important;
    -webkit-background-clip: unset !important;
    -webkit-text-fill-color: black !important;
    text-shadow: none !important;
}

/* Buttons text black */
.stButton > button {
    color: black !important;
}

/* Futuristic Bay Area Theme */
.hero {
    border-radius: 20px; 
    padding: 2rem 2.5rem; 
    background-image: url("app/pictures/San_Francisco.jpg");
    color: #e6f1ff; 
    border: 2px solid rgba(0, 255, 166, 0.3);
    box-shadow: 0 8px 32px rgba(0, 255, 166, 0.1);
    position: relative;
    overflow: hidden;
}

.hero::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(45deg, transparent 30%, rgba(0, 255, 166, 0.1) 50%, transparent 70%);
    animation: shimmer 3s infinite;
}

@keyframes shimmer {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
}

.hero h1 {
    margin-bottom: 0.5rem;
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(45deg, #00ffa6, #00d4ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 0 20px rgba(0, 255, 166, 0.3);
}

.hero small {
    opacity: 0.9;
    font-size: 1.1rem;
    display: block;
    margin-bottom: 0.5rem;
}

.glass-card {
    background: linear-gradient(135deg, rgba(10,18,31,0.8), rgba(25,36,56,0.8)); 
    border-radius: 16px; 
    border: 1px solid rgba(0, 255, 166, 0.2); 
    padding: 1.5rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    backdrop-filter: blur(10px);
}

.metric-green {color: #00ffa6 !important; font-weight: 600;}
.metric-red {color: #ff6b6b !important; font-weight: 600;}

.bay-area-accent {
    background: linear-gradient(90deg, #00ffa6, #00d4ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 600;
}

.stButton > button {
    background: linear-gradient(90deg, #00ffa6, #00d4ff);
    color: #0d1b2a;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(0, 255, 166, 0.3);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0, 255, 166, 0.4);
}

.stSelectbox > div > div {
    background: rgba(10,18,31,0.8);
    border: 1px solid rgba(0, 255, 166, 0.3);
    border-radius: 8px;
}

.stNumberInput > div > div > input {
    background: rgba(10,18,31,0.8);
    border: 1px solid rgba(0, 255, 166, 0.3);
    border-radius: 8px;
    color: #e6f1ff;
}

.stSlider > div > div > div > div {
    background: linear-gradient(90deg, #00ffa6, #00d4ff);
}

.stCheckbox > div > label {
    color: #e6f1ff;
}

.stExpander > div > div > div {
    background: rgba(10,18,31,0.6);
    border: 1px solid rgba(0, 255, 166, 0.2);
    border-radius: 12px;
}

.stForm > div {
    background: rgba(10,18,31,0.4);
    border-radius: 16px;
    padding: 1.5rem;
    border: 1px solid rgba(0, 255, 166, 0.2);
}

.stSubheader {
    color: #00ffa6;
    font-weight: 600;
    border-bottom: 2px solid rgba(0, 255, 166, 0.3);
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}

.stCaption {
    color: #a0aec0;
    font-style: italic;
}

.stInfo {
    background: rgba(0, 255, 166, 0.1);
    border: 1px solid rgba(0, 255, 166, 0.3);
    border-radius: 8px;
}

.stSuccess {
    background: rgba(0, 255, 166, 0.1);
    border: 1px solid rgba(0, 255, 166, 0.3);
    border-radius: 8px;
}

.stWarning {
    background: rgba(255, 107, 107, 0.1);
    border: 1px solid rgba(255, 107, 107, 0.3);
    border-radius: 8px;
}

/* Performance indicators */
.performance-indicator {
    display: inline-block;
    padding: 4px 8px;
    border-radius: 12px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-left: 8px;
}

.performance-fast { background: rgba(0, 255, 166, 0.2); color: #00ffa6; }
.performance-medium { background: rgba(255, 193, 7, 0.2); color: #ffc107; }
.performance-slow { background: rgba(255, 107, 107, 0.2); color: #ff6b6b; }

/* Global training status indicator */
.training-status {
    position: fixed;
    top: 20px;
    right: 20px;
    z-index: 1000;
    background: rgba(0, 255, 166, 0.9);
    color: #0d1b2a;
    padding: 10px 15px;
    border-radius: 25px;
    font-weight: 600;
    box-shadow: 0 4px 15px rgba(0, 255, 166, 0.3);
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.05); }
    100% { transform: scale(1); }
}

/* Loading animations */
.loading-spinner {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 3px solid rgba(0, 255, 166, 0.3);
    border-radius: 50%;
    border-top-color: #00ffa6;
    animation: spin 1s ease-in-out infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}
</style>
"""

st.markdown(css, unsafe_allow_html=True)

# Performance optimizations with new caching
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_cached_running_tasks():
    """Cache running tasks to avoid repeated calls."""
    try:
        return task_manager.get_running_tasks()
    except:
        return {}

@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_cached_analysis_files():
    """Cache analysis files to avoid repeated file system calls."""
    analysis_dir = Path("data/analysis_results")
    if analysis_dir.exists():
        recent_files = list(analysis_dir.glob("*.json"))
        recent_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return recent_files
    return []

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def get_cached_dataset_stats():
    """Cache dataset statistics."""
    try:
        return data_service.get_dataset_stats()
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_cached_cache_stats():
    """Cache cache manager statistics."""
    try:
        return cache_manager.get_stats()
    except Exception as e:
        return {"error": str(e)}

# Global training status indicator
running_tasks = get_cached_running_tasks()
if running_tasks:
    task_count = len(running_tasks)
    task_types = [task.get('type', 'Unknown') for task in running_tasks.values()]
    
    # Show floating status indicator
    st.markdown(f"""
    <div class="training-status">
        🔄 {task_count} Task{'s' if task_count > 1 else ''} Running
        <br><small>{', '.join(set(task_types))}</small>
    </div>
    """, unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="hero">
    <h1>⚡ EV Fleet Optimization Studio</h1>
    <small>San Francisco Bay Area • Intelligent Energy Management • ML-Powered Routing</small>
    <p style="margin-top: 1rem; opacity: 0.9;">
        Optimize your electric vehicle fleet's energy consumption, routes, and charging strategies 
        using advanced machine learning and real-time data analysis.
    </p>
</div>
""", unsafe_allow_html=True)

# Performance Overview
st.subheader("🚀 Performance Overview")

# Get cached statistics
dataset_stats = get_cached_dataset_stats()
cache_stats = get_cached_cache_stats()

col1, col2, col3, col4 = st.columns(4)

with col1:
    if "error" not in dataset_stats:
        st.metric(
            "Dataset Size", 
            f"{dataset_stats.get('total_size_mb', 0):.1f} MB",
            delta="Optimized"
        )
    else:
        st.metric("Dataset Size", "N/A", delta="Error")

with col2:
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
                
            st.metric(
                "Cache Memory", 
                f"{memory_usage:.1f} MB",
                delta=f"{memory_usage/memory_limit*100:.1f}% used"
            )
        except Exception as e:
            st.metric("Cache Memory", "Error", delta=str(e)[:20])
    else:
        st.metric("Cache Memory", "N/A", delta="Error")

with col3:
    if "error" not in dataset_stats:
        fleet_size = dataset_stats.get('fleet_size', 0)
        st.metric("Fleet Size", fleet_size, delta="Active")
    else:
        st.metric("Fleet Size", "N/A", delta="Error")

with col4:
    if "error" not in cache_stats:
        cache_items = cache_stats.get('total_items', 0)
        st.metric("Cache Items", cache_items, delta="Cached")
    else:
        st.metric("Cache Items", "N/A", delta="Error")

# Workflow Explanation
st.markdown("""
<div class="glass-card">
    <h3>🚀 Workflow Overview</h3>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin-top: 1rem;">
        <div style="background: rgba(0, 255, 166, 0.1); padding: 1rem; border-radius: 8px;">
            <h4>1. 📊 Data Generation </h4>
            <p> We build a road network graph from OpenStreetMap (OSM), where intersections are nodes and road segments are edges. Public charging stations from the Open Charge Map API and synthetic home chargers are added for realism. On this network, we simulate EV fleets with physics-based models, assigning each vehicle traits like battery size, charging rate, and driving style to capture diverse usage patterns at scale. </p>
        </div>
        <div style="background: rgba(0, 212, 255, 0.1); padding: 1rem; border-radius: 8px;">
            <h4>2. 🤖 Model Training </h4>
            <p>From the synthetic driving data, we train machine learning models to predict the energy use of EVs on each road segment. Features such as road type, speed limit, traffic, and weather feed into a regression-based model that outputs the predicted kWh per segment. This segment-level approach captures local variations, enabling accurate energy forecasting across the full network.
            </p>
        </div>
        <div style="background: rgba(255, 193, 7, 0.1); padding: 1rem; border-radius: 8px;">
            <h4>3. 🛰️ Route Optimization </h4>
            <p>Using predicted segment energies, we compute energy-aware routes with algorithms like Dijkstra and A*. To account for EV limits, we extend routing with state-of-charge (SOC) planning, balancing energy, time, and charging stops. At the fleet level, we optimize across many vehicles, integrating factors like traffic, weather, and vehicle specs. Techniques such as linear and dynamic programming enable scalable, efficient fleet routing and charging schedules.
            </p>
        </div>
        <div style="background: rgba(156, 39, 176, 0.1); padding: 1rem; border-radius: 8px;">
            <h4>4. 📈 Analytics </h4>
            <p>Visualize results and analyze fleet performance</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Recent Activity
st.subheader("📊 Recent Activity")

# Check for recent runs
analysis_files = get_cached_analysis_files()
if analysis_files:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Analysis Files", len(analysis_files))
    
    with col2:
        latest_file = analysis_files[0]
        st.metric("Latest Run", latest_file.name)
    
    with col3:
        if running_tasks:
            st.metric("Active Tasks", len(running_tasks), delta="Running")
        else:
            st.metric("Active Tasks", 0)
    
    # Show recent runs
    st.write("**Recent Analysis Results:**")
    for file in analysis_files[:5]:  # Show last 5
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                status = data.get('status', 'unknown')
                status_color = {
                    'completed': '✅',
                    'running': '🔄',
                    'failed': '❌',
                    'error': '❌'
                }.get(status, '❓')
                
                st.write(f"{status_color} {file.name} - {status}")
        except:
            st.write(f"❓ {file.name} - Unable to read")
else:
    st.info("No analysis results found yet. Start by generating data and training models!")

# System Performance
with st.expander("⚡ System Performance", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Cache Performance**")
        if "error" not in cache_stats:
            cache_usage = cache_stats.get('memory_usage_mb', 0)
            cache_limit = cache_stats.get('memory_limit_mb', 1024)
            cache_percent = (cache_usage / cache_limit) * 100
            
            st.progress(cache_percent / 100)
            st.write(f"Memory Usage: {cache_usage:.1f} MB / {cache_limit} MB ({cache_percent:.1f}%)")
            st.write(f"Cached Items: {cache_stats.get('total_items', 0)}")
            st.write(f"Disk Files: {cache_stats.get('disk_files', 0)}")
        else:
            st.error(f"Cache Error: {cache_stats['error']}")
    
    with col2:
        st.markdown("**Data Performance**")
        if "error" not in dataset_stats:
            st.write(f"Total Files: {dataset_stats.get('total_files', 0)}")
            st.write(f"Total Size: {dataset_stats.get('total_size_mb', 0):.1f} MB")
            st.write(f"Available Dates: {len(dataset_stats.get('available_dates', []))}")
            st.write(f"Fleet Size: {dataset_stats.get('fleet_size', 0)}")
        else:
            st.error(f"Data Error: {dataset_stats['error']}")

# Configuration Overview
with st.expander("⚙️ Configuration Overview", expanded=False):
    st.markdown("""
    <div class="glass-card">
        <h4>🔧 Key Configuration Variables</h4>
        <p>This dashboard allows you to configure all aspects of the EV fleet optimization system:</p>
        <ul>
            <li><strong>Fleet Configuration:</strong> Size, simulation days, geographic bounds</li>
            <li><strong>Charging Infrastructure:</strong> Home/public charging, power levels</li>
            <li><strong>Route Optimization:</strong> Algorithms, time/energy trade-offs</li>
            <li><strong>Advanced Settings:</strong> SOC planning, horizon planning, cost weights</li>
            <li><strong>Fleet Composition:</strong> EV model distribution, driver profiles</li>
        </ul>
        <p><strong>💡 Tip:</strong> See <a href="docs/config_variables_explained.md" target="_blank">detailed explanations</a> for all config variables.</p>
    </div>
    """, unsafe_allow_html=True)

# Quick Start Guide
with st.expander("🚀 Quick Start Guide", expanded=False):
    st.markdown("""
    <div class="glass-card">
        <h4>🎯 Get Started in 4 Steps</h4>
        <ol>
            <li><strong>Configure Fleet:</strong> Set fleet size, region, and simulation parameters</li>
            <li><strong>Generate Data:</strong> Create synthetic EV fleet data with realistic patterns</li>
            <li><strong>Train Models:</strong> Train ML models for energy consumption prediction</li>
            <li><strong>Optimize Routes:</strong> Run energy-aware route optimization</li>
        </ol>
        <p><strong>⚡ Performance Tip:</strong> Start with small fleet sizes (10-20 vehicles) for testing, then scale up.</p>
        <p><strong>🔧 Architecture:</strong> This system uses advanced caching, chunked data processing, and async task management for optimal performance.</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #a0aec0; padding: 1rem;">
    <p>⚡ EV Fleet Optimization Studio • Powered by Machine Learning • San Francisco Bay Area</p>
    <p><small>Built with Streamlit • Advanced Energy Modeling • Route Optimization • High-Performance Architecture</small></p>
</div>
""", unsafe_allow_html=True)

