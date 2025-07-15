"""
Charging Infrastructure Configuration
Separate from main EV config to focus on infrastructure-specific settings
"""

# Bay Area Geographic Bounds (9-county region)
BAY_AREA_BOUNDS = {
    'north': 38.5,   # Sonoma County
    'south': 36.9,   # South of San Jose
    'east': -121.2,  # Central Valley edge
    'west': -123.0   # Pacific Coast
}

# Comprehensive Bay Area regions for thorough station coverage
BAY_AREA_REGIONS = {
    'san_francisco': {
        'name': 'San Francisco',
        'center_lat': 37.7749, 
        'center_lon': -122.4194, 
        'radius_km': 15,
        'priority': 'high',  # Dense urban area
        'expected_station_density': 0.15  # stations per km²
    },
    'oakland_berkeley': {
        'name': 'Oakland/Berkeley',
        'center_lat': 37.8044, 
        'center_lon': -122.2712, 
        'radius_km': 20,
        'priority': 'high',
        'expected_station_density': 0.12
    },
    'san_jose': {
        'name': 'San Jose/South Bay',
        'center_lat': 37.3382, 
        'center_lon': -121.8863, 
        'radius_km': 25,
        'priority': 'high',
        'expected_station_density': 0.10
    },
    'peninsula': {
        'name': 'Peninsula (Palo Alto/Mountain View)',
        'center_lat': 37.4419, 
        'center_lon': -122.1430, 
        'radius_km': 20,
        'priority': 'medium',
        'expected_station_density': 0.08
    },
    'north_bay': {
        'name': 'North Bay (Santa Rosa/Napa)',
        'center_lat': 38.2975, 
        'center_lon': -122.2869, 
        'radius_km': 30,
        'priority': 'medium',
        'expected_station_density': 0.05
    },
    'east_bay': {
        'name': 'East Bay (Fremont/Hayward)',
        'center_lat': 37.6688, 
        'center_lon': -121.7680, 
        'radius_km': 25,
        'priority': 'medium',
        'expected_station_density': 0.07
    },
    'south_bay': {
        'name': 'South Bay (Gilroy/Morgan Hill)',
        'center_lat': 37.2431, 
        'center_lon': -121.7680, 
        'radius_km': 20,
        'priority': 'low',
        'expected_station_density': 0.04
    },
    'west_bay': {
        'name': 'West Bay (San Mateo/Redwood City)',
        'center_lat': 37.5630, 
        'center_lon': -122.3255, 
        'radius_km': 15,
        'priority': 'medium',
        'expected_station_density': 0.09
    }
}

# Infrastructure Scenarios (persist across simulations)
INFRASTRUCTURE_SCENARIOS = {

        "name": "Current Reality",
        "description": "Real stations + minimal mock gap-filling",
        "real_stations_enabled": True,
        "mock_stations_enabled": True,
        "home_stations_enabled": True,
        "target_density_per_km2": 0.08,
        "mock_gap_filling_threshold_km": 20,  # Add mock if no real station within 20km
        "max_mock_stations_per_gap": 2,
        "station_reliability": 0.95,  # 95% uptime
        "capacity_per_station": 4  # Average charging ports per station
}


# Mock Station Generation Rules (for critical gaps only)
MOCK_STATION_CONFIG = {
    'generation_strategy': 'critical_gaps_only',
    'gap_analysis_grid_size_km': 5,  # Analyze coverage every 5km
    'critical_gap_threshold_km': 20,  # Gap is critical if >20km to nearest station
    'mock_station_types': {
        'highway_rest_stop': {
            'probability': 0.4,
            'power_kw_range': [150, 350],  # DC Fast charging
            'connector_types': ['CCS', 'CHAdeMO'],
            'cost_per_kwh_range': [0.35, 0.50],
            'capacity_ports': 8
        },
        'shopping_center': {
            'probability': 0.3,
            'power_kw_range': [50, 150],  # Medium DC
            'connector_types': ['CCS', 'Type 2'],
            'cost_per_kwh_range': [0.30, 0.45],
            'capacity_ports': 6
        },
        'urban_level2': {
            'probability': 0.3,
            'power_kw_range': [7, 22],  # AC Level 2
            'connector_types': ['Type 2', 'J1772'],
            'cost_per_kwh_range': [0.25, 0.40],
            'capacity_ports': 4
        }
    },
    'realistic_operators': [
        'ChargePoint', 'EVgo', 'Electrify America', 'Blink', 
        'Tesla Supercharger', 'Shell Recharge', 'Volta'
    ]
}

# Station Capacity and Availability Modeling
CAPACITY_MODELING = {
    'enable_capacity_constraints': True,
    'enable_queuing': True,
    'enable_random_outages': True,
    
    # Capacity distribution (realistic port counts)
    'capacity_distribution': {
        'small_station': {'ports': 2, 'probability': 0.4},
        'medium_station': {'ports': 4, 'probability': 0.35},
        'large_station': {'ports': 8, 'probability': 0.20},
        'hub_station': {'ports': 12, 'probability': 0.05}
    },
    
    # Availability patterns
    'availability_patterns': {
        'peak_hours': [7, 8, 17, 18, 19],  # Rush hours
        'peak_utilization': 0.8,  # 80% of ports busy during peak
        'off_peak_utilization': 0.3,  # 30% busy during off-peak
        'weekend_factor': 0.6  # Weekend utilization vs weekday
    },
    
    # Random outages
    'outage_modeling': {
        'daily_outage_probability': 0.02,  # 2% chance per station per day
        'average_outage_duration_hours': 4,
        'maintenance_outage_probability': 0.005,  # Longer maintenance
        'maintenance_duration_hours': 24
    }
}

# Home Charging Configuration (generated per simulation)
HOME_CHARGING_CONFIG = {
    'default_power_kw': 7.4,  # Standard Level 2
    'power_distribution': {
        'level1_120v': {'power_kw': 1.4, 'probability': 0.1},
        'level2_standard': {'power_kw': 7.4, 'probability': 0.7},
        'level2_high': {'power_kw': 11.0, 'probability': 0.15},
        'level2_premium': {'power_kw': 19.2, 'probability': 0.05}
    },
    'cost_per_kwh': 0.15,  # Residential electricity rate
    'availability': 0.98,  # Home charging rarely unavailable
    'installation_rate_by_profile': {
        'commuter': 0.85,  # Commuters most likely to have home charging
        'rideshare': 0.60,  # Rideshare drivers less likely
        'delivery': 0.70,   # Delivery drivers moderate
        'casual': 0.75     # Casual users moderate-high
    }
}

# API Configuration
API_CONFIG = {
    'openchargemap': {
        'base_url': 'https://api.openchargemap.io/v3',
        'rate_limit_seconds': 0.5,
        'max_results_per_request': 100,
        'timeout_seconds': 15,
        'retry_attempts': 3
    }
}

# File Paths
DATA_PATHS = {
    'base_dir': 'data/charging_infrastructure',
    'real_stations_csv': 'real_stations_bay_area.csv',
    'real_stations_pkl': 'real_stations_bay_area.pkl',
    'mock_stations_csv': 'mock_stations_persistent.csv',
    'mock_stations_pkl': 'mock_stations_persistent.pkl',
    'scenarios_config': 'infrastructure_scenarios.json',
    'coverage_analysis': 'coverage_analysis.json'
}
