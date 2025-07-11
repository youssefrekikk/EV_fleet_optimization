"""
Main configuration file for EV fleet optimization project
Combines all configuration parameters and provides easy access
"""

from .ev_models import EV_MODELS
from .driver_profiles import DRIVER_PROFILES
from .physics_constants import PHYSICS_CONSTANTS, TEMPERATURE_EFFICIENCY

# Fleet Configuration
FLEET_CONFIG = {
    'fleet_size': 10,
    'simulation_days': 7,
    'start_date': '2024-01-01',
    'region': 'bay_area',
    'depot_location': (37.7749, -122.4194),  # San Francisco
    'operating_hours': {
        'start': 6,  # 6 AM
        'end': 22,   # 10 PM
    }
}

MAJOR_LOCATIONS = {
    'downtown_sf': (37.7749, -122.4194),
    'silicon_valley': (37.3861, -122.0839),
    'oakland': (37.8044, -122.2712),
    'berkeley': (37.8715, -122.2730),
    'san_jose': (37.3382, -122.0922),
    'palo_alto': (37.4419, -122.1430),
    'fremont': (37.5485, -122.9886),
    'daly_city': (37.6879, -122.4702),
    'hayward': (37.6688, -122.0808),
    'mountain_view': (37.3861, -122.0839),
    'sunnyvale': (37.3688, -122.0363),
    'santa_clara': (37.3541, -122.0322)
}

# Simulation Parameters
SIMULATION_CONFIG = {
    'time_step_minutes': 5,
    'weather_update_hours': 1,
    'charging_decision_frequency': 15,  # minutes
    'route_recalculation_threshold': 0.1,  # 10% battery remaining
    'max_daily_distance_km': 500,
    'min_charging_session_minutes': 15,
}

# Geographic Bounds (Bay Area)
GEOGRAPHIC_BOUNDS = {
    'north': 37.95,   # Tightened from 38.0
    'south': 37.25,   # Tightened from 37.2
    'east': -121.85,  # Tightened from -121.5
    'west': -122.35   # Moved east from -122.8 to avoid Pacific Ocean
}

# Charging Infrastructure
CHARGING_CONFIG = {
    'enable_home_charging': True,
    'home_charging_availability': 0.8,  # Add this - 80% of drivers have access
    'home_charging_power': 7.4,  # kW (Level 2)
    'public_fast_charging_power': 150,  # kW
    'charging_efficiency': 0.9,  # 90% efficiency
    'peak_hours': [(7,9),(16, 21)],  # 5 PM - 9 PM and 7 am - 10 am 
    'peak_pricing_multiplier': 1.5,
    'base_electricity_cost': 0.15,  # USD per kWh
    'use_real_charging_data': True,
    
    'charging_station_search_radius': 10,  # km
    'max_charging_stations_per_search': 15
}

# Weather Configuration
WEATHER_CONFIG = {
    'base_temperature': 20,  # Celsius
    'temperature_variation': 10,  # +/- degrees
    'seasonal_amplitude': 8,  # seasonal temperature swing
    'rain_probability': 0.15,  # 15% chance of rain
    'wind_speed_avg': 15,  # km/h
    'humidity_avg': 0.65,  # 65%
}

# Data Generation Settings
DATA_GENERATION = {
    'output_directory': 'data/synthetic',
    'file_formats': ['csv', 'parquet'],
    'compression': False,
    'include_noise': True,
    'noise_level': 0.02,  # 2% random noise
    'missing_data_rate': 0.001,  # 0.1% missing data points
}

# Optimization Parameters
OPTIMIZATION_CONFIG = {
    'route_optimization_algorithm': 'dijkstra',
    'charging_optimization_method': 'greedy',
    'prediction_horizon_hours': 24,
    'reoptimization_frequency_hours': 4,
    'battery_buffer_percentage': 0.15,  # Keep 15% buffer
    'max_detour_for_charging_km': 5,
}

# Export all configurations
__all__ = [
    'EV_MODELS',
    'DRIVER_PROFILES', 
    'PHYSICS_CONSTANTS',
    'TEMPERATURE_EFFICIENCY',
    'FLEET_CONFIG',
    'SIMULATION_CONFIG',
    'GEOGRAPHIC_BOUNDS',
    'CHARGING_CONFIG',
    'WEATHER_CONFIG',
    'DATA_GENERATION',
    'OPTIMIZATION_CONFIG',
    'MAJOR_LOCATIONS'
]
