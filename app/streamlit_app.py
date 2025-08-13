import os, sys
import json
from pathlib import Path

import streamlit as st

# Add project root to path for imports
CURRENT_DIR = os.path.dirname(__file__)
APP_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, '..'))
for p in [APP_DIR, PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from services.task_manager import task_manager

st.set_page_config(
    page_title="EV Fleet Optimization Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for futuristic Bay Area theme
css = """
<style>
.main .block-container {padding-top: 1rem;}

/* Futuristic Bay Area Theme */
.hero {
    border-radius: 20px; 
    padding: 2rem 2.5rem; 
    background: linear-gradient(135deg, #0d1b2a 0%, #1b263b 25%, #2d3748 50%, #4a5568 75%, #718096 100%);
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
</style>
"""

st.markdown(css, unsafe_allow_html=True)

# Performance optimizations
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

# Workflow Explanation
st.markdown("""
<div class="glass-card">
    <h3>🚀 Workflow Overview</h3>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin-top: 1rem;">
        <div style="background: rgba(0, 255, 166, 0.1); padding: 1rem; border-radius: 8px;">
            <h4>1. 📊 Data Generation</h4>
            <p>Create synthetic EV fleet data with realistic Bay Area patterns</p>
        </div>
        <div style="background: rgba(0, 212, 255, 0.1); padding: 1rem; border-radius: 8px;">
            <h4>2. 🤖 Model Training</h4>
            <p>Train ML models to predict energy consumption per road segment</p>
        </div>
        <div style="background: rgba(255, 193, 7, 0.1); padding: 1rem; border-radius: 8px;">
            <h4>3. 🛰️ Route Optimization</h4>
            <p>Find energy-optimal routes using trained models</p>
        </div>
        <div style="background: rgba(156, 39, 176, 0.1); padding: 1rem; border-radius: 8px;">
            <h4>4. 📈 Analytics</h4>
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
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #a0aec0; padding: 1rem;">
    <p>⚡ EV Fleet Optimization Studio • Powered by Machine Learning • San Francisco Bay Area</p>
    <p><small>Built with Streamlit • Advanced Energy Modeling • Route Optimization</small></p>
</div>
""", unsafe_allow_html=True)

