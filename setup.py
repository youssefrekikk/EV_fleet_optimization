"""
EV Fleet Optimization Studio
A comprehensive platform for optimizing electric vehicle fleet operations
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="ev-fleet-optimization",
    version="1.0.0",
    author="Youssef Rekik",
    description="A comprehensive platform for optimizing electric vehicle fleet operations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scipy>=1.9.0",
        "scikit-learn>=1.1.0",
        "dask>=2023.1.0",
        "dask[dataframe]>=2023.1.0",
        "xgboost>=1.7.0",
        "optuna>=3.2.0",
        "ortools>=9.4.0",
        "pulp>=2.6.0",
        "plotly>=5.10.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "streamlit>=1.33.0",
        "geopandas>=0.13.0",
        "osmnx>=1.6.0",
        "geopy>=2.3.0",
        "folium>=0.14.0",
        "pydeck>=0.8.0",
        "shapely>=2.0.0",
        "networkx>=3.1.0",
        "requests>=2.28.0",
        "beautifulsoup4>=4.11.0",
        "python-dotenv>=1.0.0",
        "pydantic>=1.10.0",
        "pyyaml>=6.0",
        "jupyter>=1.0.0",
        "pytest>=7.0.0",
        "black>=22.0.0",
        "tqdm>=4.65.0",
        "joblib>=1.3.0"
    ],
    entry_points={
        "console_scripts": [
            "ev-fleet-optimize=app.streamlit_app:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
