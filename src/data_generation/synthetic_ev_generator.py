"""
Comprehensive synthetic EV fleet data generator
Creates realistic GPS traces, consumption patterns, and charging behavior
"""

import numpy as np
import pandas as pd
import networkx as nx
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import os
from geopy.distance import geodesic
from scipy.signal import savgol_filter

# Import our configurations
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config.ev_config import *
from config.logging_config import *
from src.data_processing.openchargemap_api2 import ChargingInfrastructureManager
from dotenv import load_dotenv
from src.data_generation.road_network_db import NetworkDatabase
from src.data_generation.advanced_energy_model import AdvancedEVEnergyModel
from src.utils.logger import setup_logger, get_logger, info, warning, error, debug, print_summary, log_route_failure

# Set up logging
load_dotenv()

# Initialize logger with current configuration
logger_config = get_logging_config()
logger = setup_logger(**logger_config)

# Key locations in Bay Area for route generation


# Driving styles (separate from driver profiles)
DRIVING_STYLES = {
    'eco_friendly': {
        'efficiency_modifier': 0.85,  # 15% better efficiency
        'speed_modifier': 0.9,       # 10% slower speeds
        'acceleration_modifier': 0.7, # Gentle acceleration
        'charging_threshold': 0.4,    # Charge when battery < 40%
        'proportion': 0.25
    },
    'normal': {
        'efficiency_modifier': 1.0,   # Baseline efficiency
        'speed_modifier': 1.0,        # Normal speeds
        'acceleration_modifier': 1.0, # Normal acceleration
        'charging_threshold': 0.3,    # Charge when battery < 30%
        'proportion': 0.5
    },
    'aggressive': {
        'efficiency_modifier': 1.2,   # 20% worse efficiency
        'speed_modifier': 1.1,        # 10% faster speeds
        'acceleration_modifier': 1.4, # Hard acceleration
        'charging_threshold': 0.25,   # Charge when battery < 25%
        'proportion': 0.25
    }
}

# 🔧 FIX: Driver personality traits with realistic emergency thresholds
DRIVER_PERSONALITIES = {
    'anxious': {
        'charge_threshold': 0.5,      # Charge at 50% SOC
        'target_soc_home': 0.95,      # Charge to 95% at home
        'target_soc_public': 0.85,    # Charge to 85% at public stations
        'cost_sensitivity': 0.3,      # Low cost sensitivity (0-1 scale)
        'convenience_weight': 0.7,    # High convenience preference
        'max_detour_km': 2.0,        # Won't go far for charging
        'emergency_threshold': 0.20,  # 🔧 FIX: Panic mode at 20% (was 30%)
        'proportion': 0.2
    },
    'optimizer': {
        'charge_threshold': 0.25,     # Wait until 25%
        'target_soc_home': 0.85,      # Optimal home charging
        'target_soc_public': 0.80,    # Don't overcharge in public
        'cost_sensitivity': 0.8,      # High cost sensitivity
        'convenience_weight': 0.2,    # Will travel for savings
        'max_detour_km': 8.0,        # Will detour for better prices
        'emergency_threshold': 0.12,  # 🔧 FIX: Comfortable with low SOC (was 15%)
        'proportion': 0.3
    },
    'convenience': {
        'charge_threshold': 0.35,     # Moderate threshold
        'target_soc_home': 0.90,      # High home charging
        'target_soc_public': 0.85,    # High public charging
        'cost_sensitivity': 0.2,      # Low cost sensitivity
        'convenience_weight': 0.8,    # Prioritize convenience
        'max_detour_km': 3.0,        # Minimal detour tolerance
        'emergency_threshold': 0.18,  # 🔧 FIX: Moderate emergency threshold (was 25%)
        'proportion': 0.25
    },
    'procrastinator': {
        'charge_threshold': 0.15,     # Waits until 15%!
        'target_soc_home': 0.95,      # Then charges fully
        'target_soc_public': 0.90,    # Charges fully in public too
        'cost_sensitivity': 0.5,      # Moderate cost sensitivity
        'convenience_weight': 0.6,    # Moderate convenience preference
        'max_detour_km': 5.0,        # Moderate detour tolerance
        'emergency_threshold': 0.10,  # 🔧 FIX: Very low emergency threshold (realistic minimum)
        'proportion': 0.25
    }
}

class SyntheticEVGenerator:
    def __init__(self, config_override: Optional[Dict] = None):
        """Initialize the synthetic EV data generator"""
        self.config = self._merge_config(config_override)

        info("🏗️ Initializing charging infrastructure...", "synthetic_ev_generator")
        self.infrastructure_manager = ChargingInfrastructureManager()
        info("🔧 Auto-building charging infrastructure...", "synthetic_ev_generator")
        try:
            # Build real stations database
            real_stats = self.infrastructure_manager.build_real_stations_database(force_refresh=False)
            info(f"Real stations built: {real_stats}", "synthetic_ev_generator")
            
            # Build mock stations for gaps
            mock_stats = self.infrastructure_manager.build_mock_stations_for_gaps(force_refresh=False)
            info(f"Mock stations built: {mock_stats}", "synthetic_ev_generator")
            
            # Get final stats
            final_stats = self.infrastructure_manager.get_infrastructure_statistics()
            info(f"Infrastructure ready: {final_stats.get('total_stations', 0)} total stations", "synthetic_ev_generator")
            
        except Exception as e:
            error(f"Error building infrastructure: {e}", "synthetic_ev_generator")
            info("Continuing with limited infrastructure...", "synthetic_ev_generator")

        self.fleet_vehicles = []

        self.network_db = NetworkDatabase()
        self.energy_model = AdvancedEVEnergyModel()


        
        # Initialize random seed for reproducibility and testing remove when generating final data
        np.random.seed(42)
        
        info("Synthetic EV Generator initialized", "synthetic_ev_generator")

    def _merge_config(self, override: Optional[Dict]) -> Dict:
        """Merge configuration with overrides"""
        base_config = {
            'fleet': FLEET_CONFIG,
            'simulation': SIMULATION_CONFIG,
            'geography': GEOGRAPHIC_BOUNDS,
            'charging': CHARGING_CONFIG,
            'weather': WEATHER_CONFIG,
            'data_gen': DATA_GENERATION,
            'optimization': OPTIMIZATION_CONFIG
        }
        
        if override:
            # Deep merge configurations
            for key, value in override.items():
                if key in base_config and isinstance(base_config[key], dict):
                    base_config[key].update(value)
                else:
                    base_config[key] = value
        
        return base_config

    # Add this method after _select_driving_style() (around line 180)

    def _select_driver_personality(self) -> str:
        """Select driver personality based on proportions"""
        personalities = list(DRIVER_PERSONALITIES.keys())
        weights = [DRIVER_PERSONALITIES[personality]['proportion'] for personality in personalities]
        return np.random.choice(personalities, p=weights)


    def generate_fleet_vehicles(self) -> List[Dict]:
        """Generate fleet of vehicles with realistic characteristics"""
        info("Generating fleet vehicles...", "synthetic_ev_generator")
        
        fleet_size = self.config['fleet']['fleet_size']
        vehicles = []
        home_charging_enabled = self.config['charging'].get('enable_home_charging', True)
        home_charging_availability = self.config['charging'].get('home_charging_availability', 0.8)
        for vehicle_id in range(fleet_size):
            # Select vehicle model based on market share
            model_name = self._select_vehicle_model()
            model_specs = EV_MODELS[model_name]
            
            # Select driver profile
            driver_profile = self._select_driver_profile()
            # Select driving style (behavioral pattern) - correlated with profile for realism
            driving_style = self._select_driving_style(driver_profile)
            # Determine home charging access
            has_home_charging = (
                home_charging_enabled and 
                np.random.random() < home_charging_availability
            )
            # 🔧 FIX: Generate schedule preferences BEFORE using them
            preferred_start_hour = self._generate_preferred_start_hour(driver_profile)
            schedule_variability = self._generate_schedule_variability(driver_profile)
            
            # 🔧 FIX: More realistic starting SOC distribution
            starting_soc = self._generate_realistic_starting_soc(driver_profile, has_home_charging)

            driver_personality = self._select_driver_personality()

            # 🔧 FIX: Apply driving style efficiency modifier for realistic consumption patterns
            driving_style_modifier = DRIVING_STYLES[driving_style]['efficiency_modifier']
            base_efficiency = model_specs['efficiency'] * np.random.normal(1.0, 0.05)  # 5% variation
            
            # Apply driving style modifier
            final_efficiency = base_efficiency * driving_style_modifier
            
            # Generate vehicle-specific characteristics
            vehicle = {
                'vehicle_id': f'EV_{vehicle_id:03d}',
                'model': model_name,
                'battery_capacity': model_specs['battery_capacity'],
                'efficiency': final_efficiency,  # Now includes driving style impact
                'max_charging_speed': model_specs['max_charging_speed'],
                'driver_profile': driver_profile,
                'driving_style': driving_style,
                'driver_personality': driver_personality,
                'has_home_charging': has_home_charging,
                'home_location': self._generate_home_location(),
                'current_battery_soc': starting_soc,  # Start with random charge
                'odometer': np.random.uniform(0, 50000),  # km
                'last_service': datetime.now() - timedelta(days=np.random.randint(0, 365)),
                'preferred_start_hour': preferred_start_hour,
                'schedule_variability': schedule_variability
            }
            
            vehicles.append(vehicle)
        
        self.fleet_vehicles = vehicles
        info(f"Generated {len(vehicles)} fleet vehicles", "synthetic_ev_generator")
        info(f"Home charging enabled: {home_charging_enabled}", "synthetic_ev_generator")
        info(f"Vehicles with home charging: {sum(1 for v in vehicles if v['has_home_charging'])}", "synthetic_ev_generator")
        # NEW: Generate home charging stations for the fleet
        if home_charging_enabled:
            info("🏠 Setting up home charging infrastructure...", "synthetic_ev_generator")
            home_stations = self.infrastructure_manager.generate_home_stations_for_fleet(vehicles)
            info(f"✅ Generated {len(home_stations)} home charging stations", "synthetic_ev_generator")
        return vehicles

    def _generate_realistic_starting_soc(self, driver_profile: str, has_home_charging: bool) -> float:
        """Generate realistic starting SOC based on driver profile and charging access"""
        
        if has_home_charging:
            # Home charging users typically start with higher SOC
            soc_options = [0.2, 0.4, 0.6, 0.8, 0.9]
            soc_weights = [0.05, 0.15, 0.25, 0.35, 0.20]  # Bias toward higher SOC
        else:
            # Non-home charging users more variable
            soc_options = [0.2, 0.3, 0.4, 0.6, 0.8]
            soc_weights = [0.15, 0.25, 0.30, 0.20, 0.10]  # More spread out
        
        # Adjust based on driver profile
        if driver_profile == 'delivery':
            # Delivery drivers start day with full charge
            soc_options = [0.6, 0.8, 0.9, 0.95]
            soc_weights = [0.1, 0.3, 0.4, 0.2]
        elif driver_profile == 'rideshare':
            # Rideshare drivers need good charge to start
            soc_options = [0.4, 0.6, 0.8, 0.9]
            soc_weights = [0.1, 0.3, 0.4, 0.2]
        
        return np.random.choice(soc_options, p=soc_weights)

    def _generate_preferred_start_hour(self, driver_profile: str) -> int:
        """Generate preferred daily start hour based on driver profile"""
        
        if driver_profile == 'commuter':
            # Commuters start early, consistent times
            return np.random.choice([6, 7, 8, 9], p=[0.2, 0.4, 0.3, 0.1])
        elif driver_profile == 'delivery':
            # Delivery starts early for efficiency
            return np.random.choice([5, 6, 7, 8], p=[0.1, 0.4, 0.4, 0.1])
        elif driver_profile == 'rideshare':
            # Rideshare varies - some early, some late
            return np.random.choice([6, 7, 8, 9, 10, 11], p=[0.1, 0.2, 0.2, 0.2, 0.2, 0.1])
        else:  # casual
            # Casual drivers start later, more variable
            return np.random.choice([8, 9, 10, 11, 12], p=[0.1, 0.2, 0.3, 0.3, 0.1])

    def _generate_schedule_variability(self, driver_profile: str) -> float:
        """Generate schedule variability (hours) based on driver profile"""
        
        if driver_profile == 'commuter':
            return np.random.uniform(0.5, 1.5)  # Very consistent
        elif driver_profile == 'delivery':
            return np.random.uniform(0.5, 1.0)  # Consistent for efficiency
        elif driver_profile == 'rideshare':
            return np.random.uniform(1.0, 3.0)  # More variable
        else:  # casual
            return np.random.uniform(2.0, 4.0)  # Most variable

    
    def _select_vehicle_model(self) -> str:
        """Select vehicle model based on market share"""
        models = list(EV_MODELS.keys())
        weights = [EV_MODELS[model]['market_share'] for model in models]
        return np.random.choice(models, p=weights)
    
    def _select_driver_profile(self) -> str:
        """Select driver profile based on proportions"""
        profiles = list(DRIVER_PROFILES.keys())
        weights = [DRIVER_PROFILES[profile]['proportion'] for profile in profiles]
        return np.random.choice(profiles, p=weights)
    
    def _select_driving_style(self, driver_profile: str = None) -> str:
        """Select driving style based on proportions and driver profile correlation"""
        
        # 🔧 FIX: Correlate driving style with driver profile for realistic efficiency patterns
        if driver_profile:
            if driver_profile == 'casual':
                # Casual drivers tend to be more eco-friendly, less aggressive
                style_probs = {'eco_friendly': 0.4, 'normal': 0.5, 'aggressive': 0.1}
            elif driver_profile == 'commuter':
                # Commuters are mixed but tend to be efficient for cost savings
                style_probs = {'eco_friendly': 0.3, 'normal': 0.6, 'aggressive': 0.1}
            elif driver_profile == 'delivery':
                # Delivery drivers are time-pressured, more aggressive driving
                style_probs = {'eco_friendly': 0.1, 'normal': 0.4, 'aggressive': 0.5}
            elif driver_profile == 'rideshare':
                # Rideshare drivers balance efficiency with time pressure
                style_probs = {'eco_friendly': 0.2, 'normal': 0.6, 'aggressive': 0.2}
            else:
                # Default to original proportions
                styles = list(DRIVING_STYLES.keys())
                weights = [DRIVING_STYLES[style]['proportion'] for style in styles]
                return np.random.choice(styles, p=weights)
            
            styles = list(style_probs.keys())
            weights = list(style_probs.values())
            return np.random.choice(styles, p=weights)
        else:
            # Original method for backward compatibility
            styles = list(DRIVING_STYLES.keys())
            weights = [DRIVING_STYLES[style]['proportion'] for style in styles]
            return np.random.choice(styles, p=weights)


    def _generate_home_location(self) -> Tuple[float, float]:
        """Generate realistic home location on land (not in water bodies)"""
        
        max_attempts = 50  # Prevent infinite loops
        
        for attempt in range(max_attempts):
            # Generate random location within Bay Area bounds
            lat = np.random.uniform(
                self.config['geography']['south'], 
                self.config['geography']['north']
            )
            lon = np.random.uniform(
                self.config['geography']['west'], 
                self.config['geography']['east']
            )
            
            # Check if location is on land (not in water)
            if self._is_location_on_land(lat, lon):
                return (lat, lon)
        
        # Fallback to known safe locations if all attempts fail
        warning("Could not find land location after 50 attempts, using fallback", "synthetic_ev_generator")
        safe_locations = [
            (37.7749, -122.4194),  # San Francisco
            (37.3382, -122.0922),  # San Jose
            (37.8044, -122.2712),  # Oakland
            (37.4419, -122.1430),  # Palo Alto
            (37.5630, -122.3255),  # San Mateo
        ]
        return safe_locations[np.random.randint(0, len(safe_locations))]

    def _is_location_on_land(self, lat: float, lon: float) -> bool:
        """Check if a location is on land (not in water bodies)"""
        
        # San Francisco Bay water body bounds (approximate)
        bay_water_zones = [
            # Main SF Bay
            {
                'lat_min': 37.45, 'lat_max': 37.85,
                'lon_min': -122.35, 'lon_max': -122.05
            },
            # San Pablo Bay
            {
                'lat_min': 37.85, 'lat_max': 38.15,
                'lon_min': -122.35, 'lon_max': -122.15
            },
            # South Bay water
            {
                'lat_min': 37.35, 'lat_max': 37.55,
                'lon_min': -122.15, 'lon_max': -121.95
            }
        ]
        
        # Check if location is in any water zone
        for zone in bay_water_zones:
            if (zone['lat_min'] <= lat <= zone['lat_max'] and 
                zone['lon_min'] <= lon <= zone['lon_max']):
                return False
        
        # Check if too close to Pacific Ocean (west of certain longitude)
        if lon < -122.5:  # Too far west = Pacific Ocean
            return False
        
        # Additional checks for known water areas
        # Suisun Bay area
        if (38.0 <= lat <= 38.2 and -122.1 <= lon <= -121.8):
            return False
        
        # Assume location is on land if it passes all water checks
        return True



    def _generate_trip_destination(self, origin: Tuple[float, float], target_distance: float, 
                              driver_profile: str, is_weekend: bool) -> Tuple[float, float]:
        """Generate realistic trip destination based on driver profile and distance"""
        
        # Driver profile influences destination type
        if driver_profile == 'commuter' and not is_weekend:
            # Commuters go to work areas (downtown, business districts)
            work_areas = [
                'downtown_sf', 'silicon_valley', 'oakland', 'palo_alto'
            ]
            if np.random.random() < 0.7:  # 70% chance to go to work area
                base_location = MAJOR_LOCATIONS[np.random.choice(work_areas)]
            else:
                base_location = self._get_random_destination_within_distance(origin, target_distance)
        
        elif driver_profile == 'delivery':
            # Delivery drivers go to various commercial areas
            commercial_areas = list(MAJOR_LOCATIONS.keys())
            base_location = MAJOR_LOCATIONS[np.random.choice(commercial_areas)]
        
        elif driver_profile == 'rideshare':
            # Rideshare drivers go to popular areas
            popular_areas = [
                'downtown_sf', 'silicon_valley', 'oakland', 'berkeley', 'san_jose'
            ]
            base_location = MAJOR_LOCATIONS[np.random.choice(popular_areas)]
        
        else:  # casual
            # Casual drivers have more random destinations
            if is_weekend:
                # Weekend: recreational areas
                weekend_areas = [
                    'berkeley', 'mountain_view', 'palo_alto', 'santa_clara'
                ]
                base_location = MAJOR_LOCATIONS[np.random.choice(weekend_areas)]
            else:
                base_location = self._get_random_destination_within_distance(origin, target_distance)
        
        # Add some randomness around the base location
        if 'base_location' in locals():
            lat_offset = np.random.normal(0, 0.01)  # ~1km variation
            lon_offset = np.random.normal(0, 0.01)
            destination = (base_location[0] + lat_offset, base_location[1] + lon_offset)
        else:
            destination = self._get_random_destination_within_distance(origin, target_distance)
        
        # Ensure destination is within geographic bounds
        bounds = self.config['geography']
        destination = (
            np.clip(destination[0], bounds['south'], bounds['north']),
            np.clip(destination[1], bounds['west'], bounds['east'])
        )
        
        return destination

    def _get_random_destination_within_distance(self, origin: Tuple[float, float], 
                                            target_distance: float) -> Tuple[float, float]:
        """Generate random destination within target distance"""
        # Convert distance to approximate lat/lon degrees
        # Rough approximation: 1 degree ≈ 111 km
        max_offset = target_distance / 111.0
        
        # Generate random direction and distance
        angle = np.random.uniform(0, 2 * np.pi)
        distance_factor = np.random.uniform(0.7, 1.3)  # 70-130% of target distance
        actual_offset = (target_distance * distance_factor) / 111.0
        
        lat_offset = actual_offset * np.cos(angle)
        lon_offset = actual_offset * np.sin(angle)
        
        destination = (origin[0] + lat_offset, origin[1] + lon_offset)
        
        return destination



    def generate_daily_routes(self, vehicle: Dict, date: datetime) -> List[Dict]:
        """Generate realistic daily routes for a vehicle with proper timing"""
        driver_profile = DRIVER_PROFILES[vehicle['driver_profile']]
        driving_style = DRIVING_STYLES[vehicle['driving_style']]
        routes = []
        
        # Determine number of trips for the day
        is_weekend = date.weekday() >= 5
        weekend_factor = driver_profile['weekend_factor']
        
        if is_weekend:
            daily_km_range = (
                driver_profile['daily_km'][0] * weekend_factor,
                driver_profile['daily_km'][1] * weekend_factor
            )
            trips_range = (
                max(1, int(driver_profile['trips_per_day'][0] * weekend_factor)),
                max(2, int(driver_profile['trips_per_day'][1] * weekend_factor))
            )
        else:
            daily_km_range = driver_profile['daily_km']
            trips_range = driver_profile['trips_per_day']
        
        total_daily_km = np.random.uniform(*daily_km_range)
        num_trips = np.random.randint(*trips_range)
        
        # 🔧 FIX: Generate realistic daily start time
        daily_start_time = self._generate_daily_start_time(vehicle, date, is_weekend)
        
        # Generate individual trips with proper timing
        current_location = vehicle['home_location']
        current_time = daily_start_time
        remaining_km = total_daily_km
        
        for trip_id in range(num_trips):
            if remaining_km <= 5:  # Not enough distance for meaningful trip
                break
            
            # Determine trip distance
            if trip_id == num_trips - 1:  # Last trip
                trip_distance = remaining_km
            else:
                max_trip_distance = min(remaining_km * 0.7, remaining_km - (num_trips - trip_id - 1) * 5)
                trip_distance = np.random.uniform(5, max_trip_distance)
            
            # Generate destination
            destination = self._generate_trip_destination(
                current_location, trip_distance, vehicle['driver_profile'], is_weekend
            )
            
            # 🔧 FIX: Generate route with proper timing
            route = self._generate_route_with_timing(
                current_location, destination, vehicle, current_time, trip_id
            )
            
            if route:
                routes.append(route)
                current_location = destination
                remaining_km -= trip_distance
                
                # 🔧 FIX: Update current time based on route duration + break
                route_duration = timedelta(minutes=route['total_time_minutes'])
                break_duration = self._generate_break_duration(vehicle['driver_profile'], trip_id, num_trips)
                current_time = current_time + route_duration + break_duration
        
        return routes

    def _generate_daily_start_time(self, vehicle: Dict, date: datetime, is_weekend: bool) -> datetime:
        """Generate realistic daily start time for vehicle"""
        
        preferred_hour = vehicle['preferred_start_hour']
        variability_hours = vehicle['schedule_variability']
        
        # Weekend adjustment
        if is_weekend:
            preferred_hour += np.random.randint(0, 3)  # Start later on weekends
        
        # Add random variability
        actual_start_hour = preferred_hour + np.random.normal(0, variability_hours)
        
        # Ensure reasonable bounds (5 AM to 11 AM for most starts)
        actual_start_hour = np.clip(actual_start_hour, 5, 11)
        
        # Convert to datetime
        start_hour = int(actual_start_hour)
        start_minute = int((actual_start_hour - start_hour) * 60)
        
        return date.replace(
            hour=start_hour,
            minute=start_minute,
            second=0,
            microsecond=0
        )

    def _generate_break_duration(self, driver_profile: str, trip_id: int, total_trips: int) -> timedelta:
        """Generate realistic break duration between trips"""
        
        if driver_profile == 'delivery':
            # Short breaks for delivery
            break_minutes = np.random.uniform(10, 30)
        elif driver_profile == 'rideshare':
            # Variable breaks for rideshare
            break_minutes = np.random.uniform(5, 45)
        elif driver_profile == 'commuter':
            # Longer breaks (work day)
            if trip_id == 0:  # After first trip (going to work)
                break_minutes = np.random.uniform(300, 600)  # 5-10 hour work day
            else:
                break_minutes = np.random.uniform(30, 120)
        else:  # casual
            # Variable casual breaks
            break_minutes = np.random.uniform(60, 180)
        
        return timedelta(minutes=break_minutes)









    def _generate_route_with_timing(self, origin: Tuple[float, float], destination: Tuple[float, float],
                               vehicle: Dict, start_time: datetime, trip_id: int) -> Dict:
        """Generate realistic route using OSM road network with proper timing"""
        
        # Ensure road network is loaded
        if self.network_db.network is None:
            warning("Road network not loaded. Loading now...", "synthetic_ev_generator")
            self.network_db.load_or_create_network()
        
        # Double check that we have a network
        if self.network_db.network is None:
            error("Failed to load any road network", "synthetic_ev_generator")
            return None
        
        try:
            # Find nearest nodes in road network
            origin_node = self.network_db._find_nearest_node(origin[0], origin[1])
            dest_node = self.network_db._find_nearest_node(destination[0], destination[1])
            
            if origin_node is None or dest_node is None:
                error("Could not find nearest nodes", "synthetic_ev_generator")
                log_route_failure(origin, destination, origin_node, dest_node, self.network_db.network, "No nearest node found", vehicle_id=vehicle['vehicle_id'])
                return self._generate_fallback_route_with_timing(origin, destination, vehicle, start_time, trip_id)
            
            # Calculate shortest path
            try:
                path = nx.shortest_path(
                    self.network_db.network, origin_node, dest_node, weight='travel_time'
                )
            except nx.NetworkXNoPath:
                warning(f"No path found between {origin} and {destination}", "synthetic_ev_generator")
                log_route_failure(origin, destination, origin_node, dest_node, self.network_db.network, "No path with weight", vehicle_id=vehicle['vehicle_id'])
                # Try without weight
                try:
                    path = nx.shortest_path(self.network_db.network, origin_node, dest_node)
                except nx.NetworkXNoPath:
                    warning("No path found even without weight - using fallback", "synthetic_ev_generator")
                    log_route_failure(origin, destination, origin_node, dest_node, self.network_db.network, "No path without weight", vehicle_id=vehicle['vehicle_id'])
                    return self._generate_fallback_route_with_timing(origin, destination, vehicle, start_time, trip_id)
            
            # Extract route details
            route_coords = []
            route_speeds = []
            route_elevations = []
            total_distance = 0
            total_time = 0
            
            for i in range(len(path) - 1):
                node1, node2 = path[i], path[i + 1]
                
                # Get node coordinates
                node1_data = self.network_db.network.nodes[node1]
                node2_data = self.network_db.network.nodes[node2]
                
                coord1 = (node1_data['y'], node1_data['x'])
                coord2 = (node2_data['y'], node2_data['x'])
                
                route_coords.extend([coord1, coord2])
                
                # Get edge data (handle both OSM and mock networks)
                try:
                    edge_data = self.network_db.network.edges[node1, node2, 0]
                except KeyError:
                    # Fallback for mock network
                    try:
                        edge_data = list(self.network_db.network[node1][node2].values())[0]
                    except (KeyError, IndexError):
                        # Create default edge data
                        edge_data = {
                            'length': geodesic(coord1, coord2).meters,
                            'speed_kph': 50,
                            'travel_time': geodesic(coord1, coord2).meters / (50 * 1000 / 3600)
                        }
                
                # Speed and distance
                speed_kmh = edge_data.get('speed_kph', 50)  # Default 50 km/h
                distance_m = edge_data.get('length', geodesic(coord1, coord2).meters)
                
                route_speeds.append(speed_kmh)
                total_distance += distance_m / 1000  # Convert to km
                total_time += edge_data.get('travel_time', distance_m / (speed_kmh * 1000 / 3600))
            
            # 🔧 FIX: Generate elevation profile AFTER we have all coordinates
            route_elevations = self._generate_simple_elevation_profile(route_coords)
            # 🔧 FIX: Generate realistic GPS trace with proper start time
            gps_trace = self._generate_gps_trace_with_timing(
                route_coords, route_speeds, route_elevations, start_time, trip_id
            )
            
            # Calculate energy consumption
            consumption_data = self._calculate_energy_consumption(
                gps_trace, vehicle, start_time.date()
            )
            
            route_data = {
                'vehicle_id': vehicle['vehicle_id'],
                'trip_id': f"{vehicle['vehicle_id']}_{start_time.strftime('%Y%m%d')}_{trip_id:02d}",
                'date': start_time.strftime('%Y-%m-%d'),
                'origin': origin,
                'destination': destination,
                'total_distance_km': total_distance,
                'total_time_minutes': total_time / 60,
                'start_time': start_time.isoformat(),  # 🔧 ADD: Actual start time
                'end_time': (start_time + timedelta(seconds=total_time)).isoformat(),  # 🔧 ADD: End time
                'gps_trace': gps_trace,
                'consumption_data': consumption_data,
                'driver_profile': vehicle['driver_profile'],
                'route_type': 'osm_network'
            }
            
            return route_data
            
        except Exception as e:
            error(f"Error generating route: {e}", "synthetic_ev_generator")
            return self._generate_fallback_route_with_timing(origin, destination, vehicle, start_time, trip_id)


    def _generate_simple_elevation_profile(self, route_coords: List[Tuple[float, float]]) -> List[float]:
        """Generate realistic elevation profile for Bay Area routes"""
        
        if not route_coords:
            return []
        
        elevations = []
        
        for lat, lon in route_coords:
            # Use realistic Bay Area elevation
            elevation = self._generate_realistic_bay_area_elevation(lat, lon)
            elevations.append(elevation)
        
        # --- SMOOTHING: Apply Savitzky-Golay filter for realistic elevation profile ---
        if len(elevations) >= 5:
            # window_length must be odd and <= len(elevations)
            window_length = min(11, len(elevations) if len(elevations) % 2 == 1 else len(elevations)-1)
            if window_length < 5:
                window_length = 5 if len(elevations) >= 5 else len(elevations) | 1
            polyorder = 2 if window_length > 2 else 1
            try:
                smoothed = savgol_filter(elevations, window_length=window_length, polyorder=polyorder, mode='nearest')
                return smoothed.tolist()
            except Exception as e:
                warning(f"Smoothing failed: {e}, returning raw elevations.", "synthetic_ev_generator")
                return elevations
        else:
            return elevations

    def _generate_realistic_bay_area_elevation(self, lat: float, lon: float) -> float:
        """Generate realistic elevation based on Bay Area geography"""
        
        # Bay Area elevation zones
        if 37.75 <= lat <= 37.80 and -122.45 <= lon <= -122.40:
            # San Francisco hills
            base_elevation = 100
            variation = 80
        elif 37.70 <= lat <= 37.90 and -122.25 <= lon <= -122.05:
            # East Bay hills
            base_elevation = 200
            variation = 150
        elif 37.40 <= lat <= 37.60 and -122.35 <= lon <= -122.15:
            # Peninsula hills
            base_elevation = 150
            variation = 100
        elif 37.25 <= lat <= 37.45 and -122.15 <= lon <= -121.85:
            # South Bay (flatter)
            base_elevation = 30
            variation = 20
        else:
            # Default Bay Area
            base_elevation = 50
            variation = 40
        
        # Add realistic variation
        elevation = base_elevation + np.random.normal(0, variation/3)
        return max(0, min(800, elevation))  # Bay Area bounds: 0-800m



    def _generate_fallback_route_with_timing(self, origin: Tuple[float, float], destination: Tuple[float, float],
                                        vehicle: Dict, start_time: datetime, trip_id: int) -> Dict:
        """Generate fallback route when OSM routing fails with proper timing"""
        
        # Calculate straight-line distance
        straight_distance = geodesic(origin, destination).kilometers
        
        # Add realistic detour factor (roads aren't straight lines)
        detour_factor = np.random.uniform(1.2, 1.5)  # 20-50% longer than straight line
        total_distance = straight_distance * detour_factor
        
        # Estimate travel time based on average speed
        avg_speed_kmh = 35  # Average city driving speed
        total_time_minutes = (total_distance / avg_speed_kmh) * 60
        
        # Generate realistic GPS trace with proper speed and elevation data
        gps_trace = self._generate_fallback_gps_trace_with_timing(
            origin, destination, start_time, avg_speed_kmh, total_time_minutes
        )
        
        # Calculate energy consumption using the GPS trace
        consumption_data = self._calculate_energy_consumption(gps_trace, vehicle, start_time.date())
        
        # FIXED: Ensure consumption is calculated even for fallback routes
        if consumption_data['total_consumption_kwh'] == 0 and total_distance > 0:
            # Force minimum consumption calculation
            fallback_consumption = self._calculate_fallback_consumption(total_distance, avg_speed_kmh, vehicle)
            consumption_data = {
                'total_consumption_kwh': fallback_consumption,
                'total_distance_km': total_distance,
                'efficiency_kwh_per_100km': (fallback_consumption / total_distance) * 100,
                'temperature_celsius': 20,  # Default temperature
                'temperature_efficiency_factor': 1.0,
                'consumption_breakdown': {
                    'rolling_resistance': fallback_consumption * 0.4,
                    'aerodynamic_drag': fallback_consumption * 0.2,
                    'elevation_change': fallback_consumption * 0.1,
                    'acceleration': fallback_consumption * 0.1,
                    'hvac': fallback_consumption * 0.1,
                    'auxiliary': fallback_consumption * 0.1,
                    'regenerative_braking': 0.0,
                    'battery_thermal_loss': 0.0
                },
                'weather_conditions': {
                    'temperature': 20,
                    'is_raining': False,
                    'wind_speed_kmh': 10,
                    'humidity': 0.6,
                    'season': 'spring'
                }
            }
        
        route_data = {
            'vehicle_id': vehicle['vehicle_id'],
            'trip_id': f"{vehicle['vehicle_id']}_{start_time.strftime('%Y%m%d')}_{trip_id:02d}",
            'date': start_time.strftime('%Y-%m-%d'),
            'origin': origin,
            'destination': destination,
            'total_distance_km': total_distance,
            'total_time_minutes': total_time_minutes,
            'start_time': start_time.isoformat(),
            'end_time': (start_time + timedelta(minutes=total_time_minutes)).isoformat(),
            'gps_trace': gps_trace,
            'consumption_data': consumption_data,
            'driver_profile': vehicle['driver_profile'],
            'route_type': 'fallback_direct'
        }
        
        return route_data
    
    def _calculate_fallback_consumption(self, distance_km: float, avg_speed_kmh: float, vehicle: Dict) -> float:
        """Calculate fallback energy consumption when GPS trace fails"""
        
        # Get vehicle efficiency (kWh/100km)
        base_efficiency = vehicle.get('efficiency', 18.0)  # Default 18 kWh/100km
        
        # Apply speed correction (city driving is less efficient)
        if avg_speed_kmh < 40:
            speed_factor = 1.2  # 20% higher consumption in city
        elif avg_speed_kmh > 80:
            speed_factor = 1.3  # 30% higher consumption at highway speeds
        else:
            speed_factor = 1.0
        
        # Calculate base consumption
        base_consumption = (base_efficiency / 100) * distance_km * speed_factor
        
        # Add auxiliary consumption (time-based)
        travel_time_hours = distance_km / avg_speed_kmh
        aux_consumption = 0.5 * travel_time_hours  # 0.5 kW auxiliary load
        
        total_consumption = base_consumption + aux_consumption
        
        # Ensure minimum consumption
        min_consumption = distance_km * 0.08  # Minimum 8 kWh/100km
        
        return max(total_consumption, min_consumption)



    def _generate_gps_trace_with_timing(self, route_coords: List[Tuple[float, float]], 
                                   route_speeds: List[float], route_elevations: List[float],
                                   start_time: datetime, trip_id: int) -> List[Dict]:
        """Generate realistic GPS trace with proper timestamps starting from start_time"""
        
        gps_trace = []
        current_time = start_time  # 🔧 FIX: Use actual start time
        
        # Add realistic variations to speed and position
        for i, (coord, speed_kmh, elevation) in enumerate(zip(route_coords, route_speeds, route_elevations)):
            
            # Add realistic speed variations
            actual_speed = speed_kmh * np.random.normal(1.0, 0.1)  # 10% speed variation
            actual_speed = np.clip(actual_speed, 5, speed_kmh * 1.2)  # Reasonable bounds
            
            # Add GPS noise (typical GPS accuracy ~3-5 meters)
            lat_noise = np.random.normal(0, 0.00003)  # ~3m accuracy
            lon_noise = np.random.normal(0, 0.00003)
            
            gps_point = {
                'timestamp': current_time.isoformat(),
                'latitude': coord[0] + lat_noise,
                'longitude': coord[1] + lon_noise,
                'speed_kmh': actual_speed,
                'elevation_m': elevation + np.random.normal(0, 2),  # Elevation noise
                'heading': self._calculate_heading(i, route_coords),
                'accuracy_m': np.random.uniform(2, 8)  # GPS accuracy
            }
            
            gps_trace.append(gps_point)
            
            # Update time based on speed and distance
            if i < len(route_coords) - 1:
                next_coord = route_coords[i + 1]
                distance_m = geodesic(coord, next_coord).meters
                time_seconds = (distance_m / 1000) / (actual_speed / 3600)  # Convert to seconds
                current_time += timedelta(seconds=time_seconds)
        
        return gps_trace

    def _generate_fallback_gps_trace_with_timing(self, origin: Tuple[float, float], destination: Tuple[float, float],
                                                start_time: datetime, avg_speed_kmh: float, 
                                                total_time_minutes: float) -> List[Dict]:
        """Generate simplified GPS trace for fallback routes with proper timing"""
        
        # Calculate straight-line distance
        distance_km = geodesic(origin, destination).kilometers
        
        # FIX: Handle cases where origin and destination are too close or identical
        if distance_km < 0.001:  # Less than 1 meter
            warning(f"Origin and destination too close ({distance_km:.6f} km) - generating minimal trace", "synthetic_ev_generator")
            # Generate a minimal trace with at least 2 points
            gps_trace = []
            current_time = start_time
            
            # Add origin point
            gps_trace.append({
                'timestamp': current_time.isoformat(),
                'latitude': origin[0],
                'longitude': origin[1],
                'speed_kmh': 0.0,
                'elevation_m': 50.0,
                'heading': 0.0,
                'accuracy_m': 3.0
            })
            
            # Add destination point (slightly offset if identical)
            if distance_km == 0:
                # If identical, add small offset to destination
                dest_lat = destination[0] + 0.000001
                dest_lon = destination[1] + 0.000001
            else:
                dest_lat = destination[0]
                dest_lon = destination[1]
            
            current_time += timedelta(seconds=30)  # 30 second gap
            gps_trace.append({
                'timestamp': current_time.isoformat(),
                'latitude': dest_lat,
                'longitude': dest_lon,
                'speed_kmh': 0.0,
                'elevation_m': 50.0,
                'heading': 0.0,
                'accuracy_m': 3.0
            })
            
            return gps_trace
        
        # Number of GPS points (roughly every 30 seconds for better energy calculation)
        travel_time_hours = total_time_minutes / 60
        num_points = max(10, int(travel_time_hours * 120))  # 2 points per minute, minimum 10
        
        # FIX: Ensure minimum number of points for very short distances
        if distance_km < 0.1:  # Less than 100 meters
            num_points = max(5, num_points)  # At least 5 points for short distances
        
        gps_trace = []
        current_time = start_time
        
        for i in range(num_points):
            # Linear interpolation between origin and destination
            progress = i / (num_points - 1) if num_points > 1 else 0
            
            lat = origin[0] + (destination[0] - origin[0]) * progress
            lon = origin[1] + (destination[1] - origin[1]) * progress
            
            # Add some random variation to make it more realistic
            lat += np.random.normal(0, 0.0001)  # Small GPS noise
            lon += np.random.normal(0, 0.0001)
            
            # Realistic speed variation with acceleration/deceleration
            if i == 0:  # Starting
                speed = avg_speed_kmh * 0.3  # Start slow
            elif i == num_points - 1:  # Ending
                speed = avg_speed_kmh * 0.2  # End slow
            elif i < 3:  # Accelerating
                speed = avg_speed_kmh * (0.3 + 0.2 * i)
            elif i > num_points - 4:  # Decelerating
                speed = avg_speed_kmh * (1.0 - 0.2 * (num_points - i - 1))
            else:  # Cruising with variation
                speed = avg_speed_kmh * np.random.normal(1.0, 0.15)  # 15% variation
            
            speed = np.clip(speed, 5, avg_speed_kmh * 1.3)  # Reasonable bounds
            
            # Realistic elevation (add some variation)
            base_elevation = 50  # Bay Area average
            elevation_variation = 20 * np.sin(progress * np.pi * 2)  # Some hills
            elevation = base_elevation + elevation_variation + np.random.normal(0, 5)
            
            gps_point = {
                'timestamp': current_time.isoformat(),
                'latitude': lat,
                'longitude': lon,
                'speed_kmh': speed,
                'elevation_m': elevation,
                'heading': self._calculate_bearing(origin, destination),
                'accuracy_m': np.random.uniform(3, 8)
            }
            
            gps_trace.append(gps_point)
            
            # Update time
            if i < num_points - 1:
                time_increment = (total_time_minutes * 60) / (num_points - 1)
                current_time += timedelta(seconds=time_increment)
        
        # FIX: Final validation - ensure we have at least 2 points
        if len(gps_trace) < 2:
            warning(f"Generated GPS trace has only {len(gps_trace)} points - adding fallback points", "synthetic_ev_generator")
            # Add a second point if we only have one
            if len(gps_trace) == 1:
                gps_trace.append({
                    'timestamp': (start_time + timedelta(seconds=30)).isoformat(),
                    'latitude': destination[0],
                    'longitude': destination[1],
                    'speed_kmh': 0.0,
                    'elevation_m': 50.0,
                    'heading': 0.0,
                    'accuracy_m': 3.0
                })
        
        return gps_trace








    def _calculate_bearing(self, origin: Tuple[float, float], destination: Tuple[float, float]) -> float:
        """Calculate bearing between two points"""
        lat1, lon1 = np.radians(origin)
        lat2, lon2 = np.radians(destination)
        
        dlon = lon2 - lon1
        
        y = np.sin(dlon) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        
        bearing = np.degrees(np.arctan2(y, x))
        return (bearing + 360) % 360

    def _calculate_heading(self, index: int, route_coords: List[Tuple[float, float]]) -> float:
        """Calculate heading/bearing between GPS points"""
        if index >= len(route_coords) - 1:
            return 0.0
        
        lat1, lon1 = route_coords[index]
        lat2, lon2 = route_coords[index + 1]
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        dlon = lon2 - lon1
        
        y = np.sin(dlon) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        
        heading = np.degrees(np.arctan2(y, x))
        return (heading + 360) % 360  # Normalize to 0-360
    

    def _calculate_energy_consumption(self, gps_trace: List[Dict], 
                                vehicle: Dict, date: datetime) -> Dict:
        """Calculate realistic energy consumption using advanced physics model"""
        
        # Validate GPS trace
        if not gps_trace or len(gps_trace) < 2:
            warning(f"Invalid GPS trace for {vehicle['vehicle_id']}: {len(gps_trace) if gps_trace else 0} points", "synthetic_ev_generator")
            return self._create_zero_consumption_result()
        
       
        # Enable debugging for first few vehicles
        if hasattr(self.energy_model, 'debug_mode'):
            vehicle_num = int(vehicle['vehicle_id'].split('_')[1])
            self.energy_model.debug_mode = (vehicle_num < 5)
    

        # Get weather conditions for the day
        weather = self._get_weather_conditions(date)
        
        # Use the advanced energy model
        try:
            consumption_result = self.energy_model.calculate_energy_consumption(
                gps_trace, vehicle, weather
            )
            
            # Check if consumption calculation failed
            if (consumption_result['total_consumption_kwh'] == 0 and 
                consumption_result['total_distance_km'] > 0):
                
                warning(f"Zero consumption detected for {vehicle['vehicle_id']} "
                       f"with {consumption_result['total_distance_km']:.2f}km distance - using fallback", "synthetic_ev_generator")
                
                # Use fallback calculation
                fallback_consumption = self._calculate_fallback_consumption_simple(
                    consumption_result['total_distance_km'], vehicle
                )
                
                # Update the result with fallback values
                consumption_result['total_consumption_kwh'] = fallback_consumption
                consumption_result['efficiency_kwh_per_100km'] = (fallback_consumption / consumption_result['total_distance_km']) * 100
                
                # Update breakdown with fallback values
                consumption_result['consumption_breakdown'] = {
                    'rolling_resistance': fallback_consumption * 0.4,
                    'aerodynamic_drag': fallback_consumption * 0.2,
                    'elevation_change': fallback_consumption * 0.1,
                    'acceleration': fallback_consumption * 0.1,
                    'hvac': fallback_consumption * 0.1,
                    'auxiliary': fallback_consumption * 0.1,
                    'regenerative_braking': 0.0,
                    'battery_thermal_loss': 0.0
                }
            
            return consumption_result
            
        except Exception as e:
            error(f"Energy model failed for {vehicle['vehicle_id']}: {e}", "synthetic_ev_generator")
            
            # Calculate distance from GPS trace
            total_distance = 0
            for i in range(len(gps_trace) - 1):
                point1 = gps_trace[i]
                point2 = gps_trace[i + 1]
                segment_distance = geodesic(
                    (point1['latitude'], point1['longitude']),
                    (point2['latitude'], point2['longitude'])
                ).kilometers
                total_distance += segment_distance
            
            # Use simple fallback calculation
            fallback_consumption = self._calculate_fallback_consumption_simple(total_distance, vehicle)
            
            return {
                'total_consumption_kwh': fallback_consumption,
                'total_distance_km': total_distance,
                'efficiency_kwh_per_100km': (fallback_consumption / total_distance * 100) if total_distance > 0 else 0,
                'temperature_celsius': weather['temperature'],
                'temperature_efficiency_factor': 1.0,
                'consumption_breakdown': {
                    'rolling_resistance': fallback_consumption * 0.4,
                    'aerodynamic_drag': fallback_consumption * 0.2,
                    'elevation_change': fallback_consumption * 0.1,
                    'acceleration': fallback_consumption * 0.1,
                    'hvac': fallback_consumption * 0.1,
                    'auxiliary': fallback_consumption * 0.1,
                    'regenerative_braking': 0.0,
                    'battery_thermal_loss': 0.0
                },
                'weather_conditions': weather
            }

    def _calculate_fallback_consumption_simple(self, distance_km: float, vehicle: Dict) -> float:
        """Simple fallback energy consumption calculation"""
        
        if distance_km <= 0:
            return 0.0
        
        # Get vehicle efficiency (kWh/100km)
        base_efficiency = vehicle.get('efficiency', 18.0)  # Default 18 kWh/100km
        
        # Apply driver profile modifier
        driver_profile = vehicle.get('driver_profile', 'casual')
        if driver_profile == 'delivery':
            efficiency_modifier = 1.2  # 20% higher consumption (stop-and-go)
        elif driver_profile == 'rideshare':
            efficiency_modifier = 1.1  # 10% higher consumption (city driving)
        elif driver_profile == 'commuter':
            efficiency_modifier = 0.95  # 5% lower consumption (highway driving)
        else:  # casual
            efficiency_modifier = 1.0
        
        # Calculate consumption
        adjusted_efficiency = base_efficiency * efficiency_modifier
        consumption = (adjusted_efficiency / 100) * distance_km
        
        # Ensure minimum consumption
        min_consumption = distance_km * 0.08  # Minimum 8 kWh/100km
        
        return max(consumption, min_consumption)


    def _create_zero_consumption_result(self) -> Dict:
        """Create a minimal but realistic consumption result with noise instead of zero"""
        
        # Generate minimal realistic consumption (equivalent to ~0.5-2km trip)
        minimal_distance_km = np.random.uniform(0.1, 0.3)
        base_efficiency = np.random.uniform(12.0, 25.0)  # Realistic EV efficiency range
        minimal_consumption = (base_efficiency / 100) * minimal_distance_km
        
        # Add noise to make it more realistic
        noise_factor = np.random.normal(1.0, 0.15)  # 15% noise
        minimal_consumption *= abs(noise_factor)  # Ensure positive
        
        # Ensure minimum bounds
        minimal_consumption = max(0.008, minimal_consumption)  # At least 8Wh
        minimal_distance_km = max(0.1, minimal_distance_km)   # At least 100m
        
        # Calculate efficiency
        efficiency = (minimal_consumption / minimal_distance_km) * 100
        
        # Create realistic breakdown with noise
        breakdown_base = {
            'rolling_resistance': 0.45,
            'aerodynamic_drag': 0.25,
            'elevation_change': 0.10,
            'acceleration': 0.08,
            'hvac': 0.07,
            'auxiliary': 0.05,
            'regenerative_braking': 0.0,
            'battery_thermal_loss': 0.0
        }
        
        # Add noise to breakdown percentages
        breakdown = {}
        for component, base_pct in breakdown_base.items():
            noise = np.random.normal(1.0, 0.2)  # 20% noise on breakdown
            actual_pct = max(0.01, base_pct * abs(noise))  # Ensure positive
            breakdown[component] = minimal_consumption * actual_pct
        
        # Normalize breakdown to sum to total consumption
        breakdown_sum = sum(breakdown.values())
        if breakdown_sum > 0:
            normalization_factor = minimal_consumption / breakdown_sum
            breakdown = {k: v * normalization_factor for k, v in breakdown.items()}
        
        return {
            'total_consumption_kwh': round(minimal_consumption, 4),
            'total_distance_km': round(minimal_distance_km, 3),
            'efficiency_kwh_per_100km': round(efficiency, 2),
            'temperature_celsius': np.random.uniform(15, 25),  # Random reasonable temp
            'temperature_efficiency_factor': np.random.uniform(0.95, 1.05),  # Slight variation
            'consumption_breakdown': {k: round(v, 4) for k, v in breakdown.items()},
            'weather_conditions': {
                'temperature': np.random.uniform(15, 25),
                'is_raining': np.random.random() < 0.1,  # 10% chance of rain
                'wind_speed_kmh': np.random.uniform(5, 20),
                'humidity': np.random.uniform(0.4, 0.8),
                'season': np.random.choice(['spring', 'summer', 'autumn', 'winter'])
            }
        }



    def _get_weather_conditions(self, date: datetime) -> Dict:
        """Generate realistic weather conditions for a given date"""
        # Seasonal temperature variation
        day_of_year = date.timetuple().tm_yday
        seasonal_factor = np.sin(2 * np.pi * (day_of_year - 80) / 365)  # Peak in summer
        
        base_temp = self.config['weather']['base_temperature']
        seasonal_amplitude = self.config['weather']['seasonal_amplitude']
        daily_variation = self.config['weather']['temperature_variation']
        
        # Calculate temperature
        seasonal_temp = base_temp + seasonal_amplitude * seasonal_factor
        daily_temp = seasonal_temp + np.random.normal(0, daily_variation / 3)
        
        # Other weather conditions
        rain_prob = self.config['weather']['rain_probability']
        is_raining = np.random.random() < rain_prob
        
        wind_speed = np.random.normal(
            self.config['weather']['wind_speed_avg'], 
            self.config['weather']['wind_speed_avg'] * 0.3
        )
        
        humidity = np.random.normal(
            self.config['weather']['humidity_avg'], 0.1
        )
        
        weather_conditions = {
            'temperature': round(daily_temp, 1),
            'is_raining': is_raining,
            'wind_speed_kmh': max(0, round(wind_speed, 1)),
            'humidity': np.clip(humidity, 0.2, 0.95),
            'season': self._get_season(date)
        }
        
        # NEW: Pass weather to infrastructure manager for availability calculations
        self.infrastructure_manager.set_current_weather(weather_conditions)
        
        return weather_conditions
    

    def log_infrastructure_status(self) -> None:
        """Log current infrastructure status for debugging"""
        
        try:
            # Get infrastructure statistics
            stats = self.infrastructure_manager.get_infrastructure_statistics()
            
            info("🏗️ INFRASTRUCTURE STATUS:", "synthetic_ev_generator")
            info(f"  Total stations: {stats.get('total_stations', 0)}", "synthetic_ev_generator")
            info(f"  Real stations: {stats.get('real_stations', 0)}", "synthetic_ev_generator")
            info(f"  Mock stations: {stats.get('mock_stations', 0)}", "synthetic_ev_generator")
            info(f"  Home stations: {stats.get('home_stations', 0)}", "synthetic_ev_generator")
            info(f"  Available ports: {stats.get('available_ports', 0)}", "synthetic_ev_generator")
            info(f"  Total capacity: {stats.get('total_capacity', 0)}", "synthetic_ev_generator")
            info(f"  Utilization rate: {stats.get('utilization_rate', 0):.1%}", "synthetic_ev_generator")
            
            # Check for potential issues
            if stats.get('total_stations', 0) == 0:
                warning("⚠️ No charging stations available!", "synthetic_ev_generator")
            elif stats.get('available_ports', 0) == 0:
                warning("⚠️ No available charging ports!", "synthetic_ev_generator")
            elif stats.get('utilization_rate', 0) > 0.9:
                warning("⚠️ Very high utilization rate - may cause charging delays", "synthetic_ev_generator")
            
        except Exception as e:
            error(f"❌ Could not get infrastructure status: {e}", "synthetic_ev_generator")

    def get_infrastructure_manager(self) -> ChargingInfrastructureManager:
        """Get the infrastructure manager instance for external access"""
        return self.infrastructure_manager

    
    def _get_season(self, date: datetime) -> str:
        """Determine season based on date"""
        month = date.month
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'autumn'
    
    # Add this method after _generate_break_duration() (around line 350)

    def _should_charge_enhanced(self, vehicle: Dict, current_soc: float, 
                            location: Tuple[float, float], current_time: datetime) -> Dict:
        """
        Enhanced charging decision based on driver personality
        Returns: {
            'should_charge': bool,
            'urgency': str,  # 'low', 'medium', 'high', 'emergency'
            'target_soc': float,
            'max_acceptable_cost': float,
            'max_detour_km': float
        }
        """
        personality = DRIVER_PERSONALITIES[vehicle['driver_personality']]
        driving_style = DRIVING_STYLES[vehicle['driving_style']]
        
        # Base charging threshold from personality
        base_threshold = personality['charge_threshold']
        emergency_threshold = personality['emergency_threshold']
        
        # Modify threshold based on context
        # Time of day factor (more anxious in evening)
        hour = current_time.hour
        if 18 <= hour <= 22:  # Evening hours
            time_modifier = 0.05  # Charge 5% earlier
        elif 22 <= hour or hour <= 6:  # Night/early morning
            time_modifier = 0.1   # Charge 10% earlier
        else:
            time_modifier = 0.0
        
        # Weekend factor (more relaxed on weekends)
        is_weekend = current_time.weekday() >= 5
        weekend_modifier = -0.05 if is_weekend else 0.0
        
        # Apply personality variation (some randomness)
        personality_variation = np.random.normal(0, 0.03)  # ±3% variation
        
        # Calculate final threshold
        final_threshold = base_threshold + time_modifier + weekend_modifier + personality_variation
        final_threshold = np.clip(final_threshold, 0.1, 0.8)  # Keep reasonable bounds
        
        # Determine if charging is needed
        should_charge = current_soc <= final_threshold
        
        # Determine urgency level
        if current_soc <= emergency_threshold:
            urgency = 'emergency'
        elif current_soc <= final_threshold - 0.05:
            urgency = 'high'
        elif current_soc <= final_threshold + 0.05:
            urgency = 'medium'
        else:
            urgency = 'low'
        
        # Determine target SOC based on location and personality
        is_at_home = self._is_near_home(location, vehicle['home_location'])
        if is_at_home:
            target_soc = personality['target_soc_home']
        else:
            target_soc = personality['target_soc_public']
        
        # Add some variation to target SOC
        target_variation = np.random.normal(0, 0.02)  # ±2% variation
        target_soc = np.clip(target_soc + target_variation, current_soc + 0.1, 1.0)
        
        # Calculate acceptable cost based on urgency and personality
        base_acceptable_cost = 0.4  # Base $/kWh
        cost_sensitivity = personality['cost_sensitivity']
        
        if urgency == 'emergency':
            max_acceptable_cost = base_acceptable_cost * 2.0  # Will pay double in emergency
        elif urgency == 'high':
            max_acceptable_cost = base_acceptable_cost * (1.5 - cost_sensitivity * 0.3)
        else:
            max_acceptable_cost = base_acceptable_cost * (1.2 - cost_sensitivity * 0.4)
        
        return {
            'should_charge': should_charge,
            'urgency': urgency,
            'target_soc': target_soc,
            'max_acceptable_cost': max_acceptable_cost,
            'max_detour_km': personality['max_detour_km'],
            'final_threshold_used': final_threshold
        }


    def generate_charging_sessions(self, vehicle: Dict, routes: List[Dict], date: datetime) -> List[Dict]:
        """Generate realistic charging sessions - SIMPLIFIED for optimization focus"""
        charging_sessions = []
        driver_profile = DRIVER_PROFILES[vehicle['driver_profile']]
        driving_style = DRIVING_STYLES[vehicle['driving_style']]
        
        # Track battery state throughout the day
        current_soc = vehicle['current_battery_soc']
        battery_capacity = vehicle['battery_capacity']
        
        # Use driving style for charging threshold
        charging_threshold = driving_style['charging_threshold']
        
        # Start of day charging (only if has home charging and battery is low)
        if (vehicle['has_home_charging'] and 
            current_soc < charging_threshold and 
            np.random.random() < 0.8):
            
            # Generate early morning home charging (5-8 AM)
            morning_time = date.replace(
                hour=np.random.randint(5, 8),
                minute=np.random.randint(0, 60),
                second=0, microsecond=0
            )
            
            home_session = self._generate_home_charging_session(
                vehicle, morning_time, current_soc
            )
            charging_sessions.append(home_session)
            current_soc = home_session['end_soc']
        
        # Process each route and check for charging needs
        current_time = date.replace(hour=6, minute=0)  # Start day at 6 AM
        
        for route_idx, route in enumerate(routes):
            # Get route timing from GPS trace
            if route['gps_trace'] and len(route['gps_trace']) > 0:
                route_start_time = datetime.fromisoformat(route['gps_trace'][0]['timestamp'])
                route_end_time = datetime.fromisoformat(route['gps_trace'][-1]['timestamp'])
            else:
                # Fallback: estimate route timing
                route_duration_hours = route['total_time_minutes'] / 60
                route_start_time = current_time
                route_end_time = route_start_time + timedelta(hours=route_duration_hours)
            
            # Update current time
            current_time = route_end_time
            
            # Consume energy for this route
            consumption_kwh = route['consumption_data']['total_consumption_kwh']
            energy_consumed_percent = consumption_kwh / battery_capacity
            current_soc -= energy_consumed_percent
            
            # 🔧 FIX: Realistic minimum SOC protection - modern EVs prevent deep discharge
            # Most EVs have BMS that prevents discharge below 10-15% to protect battery health
            min_operational_soc = 0.12  # 12% minimum - realistic BMS protection
            current_soc = max(min_operational_soc, current_soc)
            
            # If approaching critical level, force emergency charging
            if current_soc <= 0.15:
                charging_decision = {
                    'should_charge': True,
                    'urgency': 'emergency',
                    'target_soc': 0.8,  # Emergency charge to 80%
                    'max_acceptable_cost': 1.0  # Will pay any reasonable price
                }
                needs_charging = True
                urgency = 'emergency'
                target_soc = 0.8
            else:
                charging_decision = self._should_charge_enhanced(
                    vehicle, current_soc, route['destination'], route_end_time
                )
                needs_charging = charging_decision['should_charge']
                urgency = charging_decision['urgency']
                target_soc = charging_decision['target_soc']
            # 🔧 FIX: Realistic opportunistic charging with home charging preference
            # People with home charging are much less likely to use public charging opportunistically
            if vehicle['has_home_charging']:
                # Home owners prefer to wait and charge at home unless urgent
                if vehicle['driver_profile'] == 'rideshare':
                    opportunistic_prob = 0.25  # Still need flexibility for rideshare
                elif vehicle['driver_profile'] == 'delivery':
                    opportunistic_prob = 0.20  # Commercial drivers sometimes need quick top-ups
                else:
                    opportunistic_prob = 0.05  # Casual/commuters rarely do opportunistic public charging
            else:
                # Non-home charging users depend more on public infrastructure
                if vehicle['driver_profile'] == 'rideshare':
                    opportunistic_prob = 0.45
                elif vehicle['driver_profile'] == 'delivery':
                    opportunistic_prob = 0.40
                else:
                    opportunistic_prob = 0.35
            
            opportunistic_charging = (
                current_soc < 0.6 and  # Less than 60% (lower threshold)
                np.random.random() < opportunistic_prob
            )
            
            if needs_charging or opportunistic_charging:
                # Determine where we're going next
                if route_idx < len(routes) - 1:
                    next_destination = routes[route_idx + 1]['destination']
                else:
                    next_destination = vehicle['home_location']
                # 🔧 FIX: Realistic charging type decision with strong home preference
                is_at_home = self._is_near_home(route['destination'], vehicle['home_location'])
                is_near_home = self._is_near_home(route['destination'], vehicle['home_location'], threshold_km=5.0)
                
                # Decide charging type based on realistic human behavior
                use_home_charging = False
                
                if vehicle['has_home_charging']:
                    if is_at_home:
                        # At home - almost always use home charging
                        use_home_charging = np.random.random() < 0.95  # 95% probability
                    elif is_near_home and not needs_charging:
                        # Near home for opportunistic charging - strong preference for home
                        use_home_charging = np.random.random() < 0.85  # 85% go home to charge
                    elif urgency == 'low' and is_near_home:
                        # Low urgency and near home - prefer to go home
                        use_home_charging = np.random.random() < 0.70  # 70% go home
                    elif urgency == 'medium':
                        # Medium urgency - sometimes still prefer home if nearby
                        if is_near_home:
                            use_home_charging = np.random.random() < 0.40  # 40% still go home
                        else:
                            use_home_charging = np.random.random() < 0.15  # 15% drive home
                    # Emergency charging - rarely go home unless already there
                    elif urgency == 'emergency':
                        use_home_charging = is_at_home and np.random.random() < 0.80
                
                if use_home_charging:
                    # Home charging - potentially with detour time
                    if is_at_home:
                        detour_time = np.random.randint(5, 30)  # Already home
                    else:
                        detour_time = np.random.randint(20, 60)  # Time to drive home
                    
                    charging_start_time = route_end_time + timedelta(minutes=detour_time)
                    
                    charging_session = self._generate_home_charging_session(
                        vehicle, charging_start_time, current_soc
                    )
                else:
                    # Public charging - use simplified infrastructure method
                    travel_to_station_minutes = np.random.randint(10, 45)  # Time to find and reach station
                    charging_start_time = route_end_time + timedelta(minutes=travel_to_station_minutes)
                    
                    charging_session = self._generate_public_charging_session_with_infrastructure(
                        vehicle, route['destination'], charging_start_time, current_soc, needs_charging,next_destination
                    )
                
                if charging_session:
                    charging_sessions.append(charging_session)
                    current_soc = charging_session['end_soc']
                    
                    # Update current time to after charging
                    charging_end_time = datetime.fromisoformat(charging_session['end_time'])
                    current_time = charging_end_time
        
        # 🔧 FIX: Realistic end-of-day home charging timing
        if (vehicle['has_home_charging'] and 
            current_soc < 0.8 and 
            np.random.random() < 0.9):
            
            # Generate realistic evening charging time based on human behavior
            # Most people charge when they get home from work (6-11 PM)
            base_date = current_time.date()
            
            # Determine realistic evening charging hour based on driver profile
            if vehicle['driver_profile'] == 'commuter':
                # Commuters typically arrive home 6-8 PM
                evening_hour = np.random.choice([18, 19, 20, 21], p=[0.3, 0.4, 0.2, 0.1])
            elif vehicle['driver_profile'] == 'delivery':
                # Delivery drivers finish earlier but may be tired, charge 5-7 PM
                evening_hour = np.random.choice([17, 18, 19, 20], p=[0.2, 0.4, 0.3, 0.1])
            elif vehicle['driver_profile'] == 'rideshare':
                # Rideshare drivers have variable schedules, may charge before evening peak
                evening_hour = np.random.choice([16, 17, 18, 19, 20, 21, 22], p=[0.1, 0.15, 0.2, 0.2, 0.15, 0.15, 0.05])
            else:  # casual
                # Casual drivers charge when convenient, later evening
                evening_hour = np.random.choice([19, 20, 21, 22], p=[0.2, 0.3, 0.3, 0.2])
            
            # Add some variation in minutes
            evening_minute = np.random.randint(0, 60)
            
            evening_charging_time = datetime.combine(base_date, datetime.min.time()).replace(
                hour=evening_hour, minute=evening_minute
            )
            
            # If the calculated time is before current time, it means we finished routes very late
            # In this case, charge immediately but not past midnight
            if evening_charging_time <= current_time:
                if current_time.hour >= 23:
                    # Too late - charge early morning next day
                    next_day = base_date + timedelta(days=1)
                    morning_hour = np.random.choice([5, 6, 7], p=[0.2, 0.5, 0.3])
                    evening_charging_time = datetime.combine(next_day, datetime.min.time()).replace(
                        hour=morning_hour, minute=np.random.randint(0, 60)
                    )
                else:
                    # Add small delay from current time
                    evening_charging_time = current_time + timedelta(minutes=np.random.randint(15, 60))
            
            end_day_session = self._generate_home_charging_session(
                vehicle, evening_charging_time, current_soc
            )
            charging_sessions.append(end_day_session)
        
        return charging_sessions



    def _is_near_home(self, location: Tuple[float, float], home_location: Tuple[float, float], 
                    threshold_km: float = 2.0) -> bool:
        """Check if location is near home (within threshold)"""
        from geopy.distance import geodesic
        distance = geodesic(location, home_location).kilometers
        return distance <= threshold_km




    def _calculate_charging_time(self, energy_needed: float, charging_power: float,
                               start_soc: float, target_soc: float) -> float:
        """Calculate realistic charging time with charging curve"""
        
        if energy_needed <= 0:
            return 0
        
        # Simplified charging curve - charging slows down as battery fills
        avg_soc = (start_soc + target_soc) / 2
        
        if avg_soc < 0.2:
            power_factor = 1.0  # Full power at low SOC
        elif avg_soc < 0.5:
            power_factor = 0.95  # Slight reduction
        elif avg_soc < 0.8:
            power_factor = 0.8   # Moderate reduction
        else:
            power_factor = 0.5   # Significant reduction at high SOC
        
        effective_power = charging_power * power_factor
        charging_time_hours = energy_needed / effective_power
        
        # Add some randomness for real-world variations
        charging_time_hours *= np.random.normal(1.0, 0.1)  # 10% variation
        
        return max(0.25, charging_time_hours)  # Minimum 15 minutes

    
    def _generate_home_charging_session(self, vehicle: Dict, start_time: datetime, 
                                  start_soc: float) -> Dict:
        """Generate home charging session - REALISTIC human behavior"""
        
        charging_power = self.config['charging']['home_charging_power']  # 7.4 kW
        battery_capacity = vehicle['battery_capacity']
        personality = DRIVER_PERSONALITIES[vehicle['driver_personality']]
        
        # 🔧 FIX: Realistic home charging behavior
        # Many people don't charge to 100%, some stop early due to time constraints or habits
        base_target = personality['target_soc_home']
        
        # Add realistic human variation - sometimes people stop charging early
        time_of_day = start_time.hour
        day_of_week = start_time.weekday()
        
        # Night charging (10 PM - 6 AM) - people tend to charge fully
        if 22 <= time_of_day or time_of_day <= 6:
            target_soc_variation = np.random.uniform(0.9, 1.0)  # 90-100% of target
        # Evening charging (6 PM - 10 PM) - sometimes interrupted
        elif 18 <= time_of_day <= 22:
            # 30% chance of early interruption (visitors, going out, forgetting)
            if np.random.random() < 0.3:
                target_soc_variation = np.random.uniform(0.4, 0.7)  # Only partial charge
            else:
                target_soc_variation = np.random.uniform(0.8, 1.0)
        # Weekend vs weekday differences
        else:
            target_soc_variation = np.random.uniform(0.7, 1.0)
        
        # Weekend effect - more relaxed charging
        if day_of_week >= 5:  # Weekend
            if np.random.random() < 0.2:  # 20% chance of lazy charging
                target_soc_variation *= np.random.uniform(0.6, 0.8)
        
        target_soc = min(0.95, base_target * target_soc_variation)  # Cap at 95%
        target_soc = max(0.3, target_soc)  # Floor at 30% for minimum usability
        
        energy_needed = max(0, (target_soc - start_soc) * battery_capacity)
        
        # Calculate charging time (with charging curve)
        charging_time_hours = self._calculate_charging_time(
            energy_needed, charging_power, start_soc, target_soc
        )
        
        # Cost calculation - home charging uses base electricity rate
        electricity_cost = self.config['charging']['base_electricity_cost']
        total_cost = energy_needed * electricity_cost
        
        # Simple home station info
        station_location = f"({float(vehicle['home_location'][0]):.6f}, {float(vehicle['home_location'][1]):.6f})"
        station_id = f"home_{vehicle['vehicle_id']}"
        
        return {
            'session_id': f"{vehicle['vehicle_id']}_{start_time.strftime('%Y%m%d_%H%M')}_home",
            'vehicle_id': vehicle['vehicle_id'],
            'charging_type': 'home',
            'station_id': station_id,
            'station_operator': 'Home',
            'location': station_location,
            'start_time': start_time.isoformat(),
            'end_time': (start_time + timedelta(hours=charging_time_hours)).isoformat(),
            'start_soc': round(start_soc, 3),
            'end_soc': round(target_soc, 3),
            'energy_delivered_kwh': round(energy_needed, 2),
            'charging_power_kw': charging_power,
            'duration_hours': round(charging_time_hours, 2),
            'cost_usd': round(total_cost, 2),
            'cost_per_kwh': electricity_cost,
            'is_emergency_charging': False,
            'connector_type': 'Type 1 (J1772)',
            'distance_to_station_km': 0.0
        }









    def _generate_public_charging_session_with_infrastructure(self, vehicle: Dict, location: Tuple[float, float],
                                                        start_time: datetime, start_soc: float, 
                                                        is_emergency: bool,next_destination: Tuple[float, float] = None) -> Optional[Dict]:
        """Generate public charging session using infrastructure manager - SIMPLIFIED"""
        
        # Find nearby charging stations using infrastructure manager
        nearby_stations = self.infrastructure_manager.find_nearby_stations(
            location[0], location[1], 
            radius_km=self.config['charging']['charging_station_search_radius'],
            max_results=self.config['charging']['max_charging_stations_per_search']
        )
        
        if not nearby_stations:
            warning(f"No charging stations found near {location} for {vehicle['vehicle_id']}", "synthetic_ev_generator")
            return None
        
        # Simple availability filter - just operational stations
        available_stations = [
            station for station in nearby_stations 
            if station.get('is_operational', True)
        ]
        
        if not available_stations:
            warning(f"No operational charging stations near {location} for {vehicle['vehicle_id']}", "synthetic_ev_generator")
            return None

        # Create charging decision based on personality
        charging_decision = {
            'urgency': 'emergency' if is_emergency else 'medium',
            'max_acceptable_cost': 0.50 if is_emergency else 0.35,
            'max_detour_km': DRIVER_PERSONALITIES[vehicle['driver_personality']]['max_detour_km']
        }
        destination_for_detour = next_destination or vehicle['home_location']
        # Now select station with personality
        selected_station = self._select_station_with_personality(
            available_stations, vehicle, charging_decision,location,destination_for_detour
        )     

        
        
        if not selected_station:
            return None
        
        # Determine charging parameters
        personality = DRIVER_PERSONALITIES[vehicle['driver_personality']]

        # 🔧 FIX: Realistic public charging behavior
        if is_emergency:
            # Emergency charging - people are stressed and might stop earlier than optimal
            base_target = personality['target_soc_public']
            # 20% chance to stop charging early due to anxiety/cost concerns
            if np.random.random() < 0.2:
                target_soc = base_target * np.random.uniform(0.6, 0.8)  # Stop at 60-80% of target
            else:
                target_soc = base_target * np.random.uniform(0.9, 1.0)  # Nearly full charge
        else:
            # Opportunistic charging - highly variable human behavior
            base_target = personality['target_soc_public']
            
            # Time pressure affects charging behavior
            hour = start_time.hour
            if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
                # People in hurry - shorter charging sessions
                target_soc = base_target * np.random.uniform(0.3, 0.6)  # Quick top-up
            elif 12 <= hour <= 14:  # Lunch hours
                # Moderate time available
                target_soc = base_target * np.random.uniform(0.5, 0.8)
            else:
                # More time available - longer sessions
                target_soc = base_target * np.random.uniform(0.6, 0.9)
            
            # Cost sensitivity affects behavior
            cost_per_kwh = selected_station.get('cost_usd_per_kwh', 0.35)
            if cost_per_kwh > 0.40:  # Expensive station
                target_soc *= np.random.uniform(0.7, 0.9)  # Charge less due to cost
        
        # Ensure reasonable bounds
        target_soc = max(0.25, min(0.95, target_soc))
        
        battery_capacity = vehicle['battery_capacity']
        energy_needed = max(0, (target_soc - start_soc) * battery_capacity)
        # Determine charging power based on station and vehicle capabilities
        station_max_power = selected_station.get('max_power_kw', 22)  # Default 22kW
        vehicle_max_power = vehicle.get('max_charging_speed', 50)     # Vehicle limit
        charging_power = min(station_max_power, vehicle_max_power)

        # Apply urgency factor (emergency = use max power, normal = maybe slower)
        if not is_emergency:
            # Sometimes use lower power for cost savings or availability
            charging_power *= np.random.uniform(0.7, 1.0)

        
        # Calculate charging time
        charging_time_hours = self._calculate_charging_time(
            energy_needed, charging_power, start_soc, target_soc
        )
        
        # Calculate cost with proper peak/off-peak pricing
        station_base_cost = selected_station.get('estimated_cost_per_kwh', 0.30)
        
        # Peak hour pricing logic
        peak_hours = self.config['charging']['peak_hours']
        is_peak_hour = any(start <= start_time.hour <= end for start, end in peak_hours)
        
        if is_peak_hour:
            cost_per_kwh = station_base_cost * self.config['charging']['peak_pricing_multiplier']
        else:
            cost_per_kwh = station_base_cost
        
        total_cost = energy_needed * cost_per_kwh
        
        # Create charging session
        charging_session = {
            'session_id': f"{vehicle['vehicle_id']}_{start_time.strftime('%Y%m%d_%H%M')}_public",
            'vehicle_id': vehicle['vehicle_id'],
            'charging_type': 'public',
            'station_id': selected_station.get('station_id', 'unknown'),
            'station_operator': selected_station.get('operator', 'Unknown'),
            'location': f"({float(selected_station.get('latitude', 0)):.6f}, {float(selected_station.get('longitude', 0)):.6f})",
            'start_time': start_time.isoformat(),
            'end_time': (start_time + timedelta(hours=charging_time_hours)).isoformat(),
            'start_soc': round(start_soc, 3),
            'end_soc': round(target_soc, 3),
            'energy_delivered_kwh': round(energy_needed, 2),
            'charging_power_kw': charging_power,
            'duration_hours': round(charging_time_hours, 2),
            'cost_usd': round(total_cost, 2),
            'cost_per_kwh': round(cost_per_kwh, 3),
            'is_emergency_charging': is_emergency,
            'connector_type': selected_station.get('connector_types', ['Unknown'])[0] if selected_station.get('connector_types') else 'Unknown',
            'distance_to_station_km': selected_station.get('distance_km', 0)
        }
        
        return charging_session

    def _select_station_simple(self, stations: List[Dict], vehicle: Dict, is_emergency: bool) -> Optional[Dict]:
        """Simple but realistic station selection focused on optimization"""
        
        if not stations:
            return None
        
        if is_emergency:
            # Emergency: prioritize fast charging and proximity
            fast_stations = [s for s in stations if s.get('max_power_kw', 0) >= 50]
            if fast_stations:
                # Choose closest fast charger
                return min(fast_stations, key=lambda x: x.get('distance_km', 999))
            else:
                # No fast chargers, choose closest available
                return min(stations, key=lambda x: x.get('distance_km', 999))
        else:
            # Normal charging: balance distance and cost
            def simple_score(station):
                distance_km = station.get('distance_km', 10)
                cost_per_kwh = station.get('estimated_cost_per_kwh', 0.30)
                
                # Simple scoring: minimize distance + cost penalty
                # Distance penalty: 1 point per km
                # Cost penalty: 10 points per $0.10/kWh above $0.25
                distance_penalty = distance_km
                cost_penalty = max(0, (cost_per_kwh - 0.25) * 100)  # Penalty for expensive stations
                
                return distance_penalty + cost_penalty
            
            # Return station with lowest penalty score
            return min(stations, key=simple_score)


    def _calculate_actual_detour(self, current_location: Tuple[float, float], 
                           destination: Tuple[float, float], 
                           station_location: Tuple[float, float]) -> float:
        """
        Calculate actual detour distance for charging station
        Returns: detour in km (0 if station is on the way)
        """
        from geopy.distance import geodesic
        
        # Direct distance to destination
        direct_distance = geodesic(current_location, destination).kilometers
        
        # Distance via charging station
        to_station = geodesic(current_location, station_location).kilometers
        from_station = geodesic(station_location, destination).kilometers
        via_station_distance = to_station + from_station
        
        # Actual detour (can be negative if station is very close to direct path)
        detour = via_station_distance - direct_distance
        
        # Return 0 if detour is minimal (within 0.5km tolerance)
        return max(0, detour - 0.5)


    # Update _select_station_with_personality() to use proper detour calculation:

    def _select_station_with_personality(self, stations: List[Dict], vehicle: Dict, 
                                    charging_decision: Dict, 
                                    current_location: Tuple[float, float] = None,
                                    destination: Tuple[float, float] = None) -> Optional[Dict]:
        """
        Select charging station based on driver personality and charging urgency
        """
        if not stations:
            return None
        
        personality = DRIVER_PERSONALITIES[vehicle['driver_personality']]
        urgency = charging_decision['urgency']
        max_cost = charging_decision['max_acceptable_cost']
        max_detour = charging_decision['max_detour_km']
        
        # Filter stations by detour and cost constraints
        acceptable_stations = []
        for station in stations:
            # Calculate actual detour if we have route information
            if current_location and destination:
                station_location = (station.get('latitude', 0), station.get('longitude', 0))
                actual_detour = self._calculate_actual_detour(current_location, destination, station_location)
            else:
                # Fallback to simple distance
                actual_detour = station.get('distance_km', 0)
            
            cost_per_kwh = station.get('estimated_cost_per_kwh', 0.30)
            
            # Check detour constraint
            if actual_detour > max_detour:
                continue
            
            # Check cost constraint (unless emergency)
            if urgency != 'emergency' and cost_per_kwh > max_cost:
                continue
            
            # Add calculated detour to station data for scoring
            station['actual_detour_km'] = actual_detour
            acceptable_stations.append(station)
        
        if not acceptable_stations:
            # If no stations meet criteria, relax constraints based on urgency
            if urgency in ['high', 'emergency']:
                # Accept any station within reasonable distance
                for station in stations:
                    if current_location and destination:
                        station_location = (station.get('latitude', 0), station.get('longitude', 0))
                        actual_detour = self._calculate_actual_detour(current_location, destination, station_location)
                    else:
                        actual_detour = station.get('distance_km', 0)
                    
                    if actual_detour <= max_detour * 1.5:
                        station['actual_detour_km'] = actual_detour
                        acceptable_stations.append(station)
            
            if not acceptable_stations:
                # Last resort: closest station
                return min(stations, key=lambda x: x.get('distance_km', 999))
        
        # Score remaining stations based on personality
        def calculate_station_score(station):
            detour_km = station.get('actual_detour_km', station.get('distance_km', 0))
            cost_per_kwh = station.get('estimated_cost_per_kwh', 0.30)
            max_power_kw = station.get('max_power_kw', 22)
            
            # Detour penalty (higher weight for convenience-focused personalities)
            detour_penalty = detour_km * personality['convenience_weight'] * 3
            
            # Cost penalty (higher weight for cost-sensitive personalities)
            cost_penalty = (cost_per_kwh - 0.25) * personality['cost_sensitivity'] * 10
            
            # Speed bonus (faster charging is generally preferred)
            speed_bonus = min(max_power_kw / 50, 1.0) * 2
            
            # Urgency modifier
            if urgency == 'emergency':
                detour_penalty *= 2
                cost_penalty *= 0.5
                speed_bonus *= 2
            elif urgency == 'high':
                detour_penalty *= 1.5
                cost_penalty *= 0.7
            
            total_score = detour_penalty + cost_penalty - speed_bonus
            return total_score
        
        # Select station with best score
        best_station = min(acceptable_stations, key=calculate_station_score)
        return best_station



    def generate_complete_dataset(self, num_days: int = 30) -> Dict[str, pd.DataFrame]:
        """Generate complete synthetic EV fleet dataset - SIMPLIFIED"""
        
        info(f"🚀 Generating complete dataset for {num_days} days...", "synthetic_ev_generator")
        
        # Initialize data structures
        all_routes = []
        all_charging_sessions = []
        all_vehicle_states = []
        all_weather_data = []
        
        # Load road network ONCE at the beginning
        info("📍 Loading road network...", "synthetic_ev_generator")
        self.network_db.load_or_create_network()
        
        # Generate fleet vehicles (this will also create home charging stations)
        if not self.fleet_vehicles:
            info("🚗 Generating fleet vehicles...", "synthetic_ev_generator")
            self.generate_fleet_vehicles()
        
        # Log initial infrastructure status
        info("🏗️ Infrastructure status:", "synthetic_ev_generator")
        self.log_infrastructure_status()
        
        # Generate data for each day
        start_date = datetime.strptime(self.config['fleet']['start_date'], '%Y-%m-%d')
        
        for day in range(num_days):
            current_date = start_date + timedelta(days=day)
            info(f"📅 Generating data for day {day + 1}/{num_days}: {current_date.strftime('%Y-%m-%d')}", "synthetic_ev_generator")
            
            # Generate weather for the day
            daily_weather = self._get_weather_conditions(current_date)
            daily_weather['date'] = current_date.strftime('%Y-%m-%d')
            all_weather_data.append(daily_weather)
            
            # Generate data for each vehicle
            for vehicle_idx, vehicle in enumerate(self.fleet_vehicles):
                try:
                    # Generate daily routes
                    daily_routes = self.generate_daily_routes(vehicle, current_date)
                    
                    # Filter out None routes and log statistics
                    valid_routes = [route for route in daily_routes if route is not None]
                    failed_routes = len(daily_routes) - len(valid_routes)
                    
                    if failed_routes > 0:
                        debug(f"Vehicle {vehicle['vehicle_id']}: {len(valid_routes)} successful routes, {failed_routes} fallback routes", "synthetic_ev_generator")
                    
                    all_routes.extend(valid_routes)
                    
                    # Generate charging sessions
                    charging_sessions = self.generate_charging_sessions(
                        vehicle, valid_routes, current_date
                    )
                    
                    # Simple validation - just check for required fields
                    valid_charging_sessions = []
                    for session in charging_sessions:
                        if self._validate_charging_session_simple(session):
                            valid_charging_sessions.append(session)
                        else:
                            warning(f"Invalid charging session for {vehicle['vehicle_id']}: {session.get('session_id', 'unknown')}", "synthetic_ev_generator")
                    
                    all_charging_sessions.extend(valid_charging_sessions)
                    
                    # Track vehicle state
                    total_distance = sum(route['total_distance_km'] for route in valid_routes)
                    total_consumption = sum(
                        route['consumption_data']['total_consumption_kwh'] 
                        for route in valid_routes
                    )
                    
                    vehicle_state = {
                        'vehicle_id': vehicle['vehicle_id'],
                        'date': current_date.strftime('%Y-%m-%d'),
                        'total_distance_km': round(total_distance, 2),
                        'total_consumption_kwh': round(total_consumption, 3),
                        'efficiency_kwh_per_100km': round(
                            (total_consumption / total_distance * 100) if total_distance > 0 else 0, 2
                        ),
                        'num_trips': len(valid_routes),
                        'num_charging_sessions': len(valid_charging_sessions),
                        'driver_profile': vehicle['driver_profile'],
                        'vehicle_model': vehicle['model'],
                        'has_home_charging': vehicle['has_home_charging']
                    }
                    all_vehicle_states.append(vehicle_state)
                    
                except Exception as e:
                    error(f"Error generating data for vehicle {vehicle['vehicle_id']} on {current_date}: {e}", "synthetic_ev_generator")
                    continue
            
            # Log daily progress
            if (day + 1) % 7 == 0 or day == num_days - 1:
                info(f"✅ Completed {day + 1}/{num_days} days", "synthetic_ev_generator")
                info(f"   Routes generated: {len(all_routes):,}", "synthetic_ev_generator")
                info(f"   Charging sessions: {len(all_charging_sessions):,}", "synthetic_ev_generator")
        
        # Convert to DataFrames
        datasets = {
            'routes': self._create_routes_dataframe(all_routes),
            'charging_sessions': pd.DataFrame(all_charging_sessions),
            'vehicle_states': pd.DataFrame(all_vehicle_states),
            'weather': pd.DataFrame(all_weather_data),
            'fleet_info': pd.DataFrame(self.fleet_vehicles)
        }
        
        # Add infrastructure datasets (simplified)
        try:
            infrastructure_datasets = self._create_infrastructure_datasets_simple()
            datasets.update(infrastructure_datasets)
        except Exception as e:
            warning(f"Could not create infrastructure datasets: {e}", "synthetic_ev_generator")
        
        info("✅ Dataset generation complete!", "synthetic_ev_generator")
        self._print_dataset_summary(datasets)
        
        return datasets

    def _validate_charging_session_simple(self, session: Dict) -> bool:
        """Simple validation for charging session data"""
        
        required_fields = [
            'session_id', 'vehicle_id', 'charging_type', 'start_time', 'end_time',
            'start_soc', 'end_soc', 'energy_delivered_kwh', 'cost_usd'
        ]
        
        # Check required fields exist and are not None
        for field in required_fields:
            if field not in session or session[field] is None:
                return False
        
        # Basic range checks
        try:
            # SOC should be between 0 and 1
            if not (0 <= session['start_soc'] <= 1 and 0 <= session['end_soc'] <= 1):
                return False
            
            # End SOC should be greater than start SOC
            if session['end_soc'] <= session['start_soc']:
                return False
            
            # Energy and cost should be positive
            if session['energy_delivered_kwh'] <= 0 or session['cost_usd'] < 0:
                return False
            
            return True
            
        except Exception:
            return False

    def _create_infrastructure_datasets_simple(self) -> Dict[str, pd.DataFrame]:
        """Create simplified infrastructure datasets"""
        
        infrastructure_datasets = {}
        
        try:
            # Combined infrastructure
            combined_stations = self.infrastructure_manager.get_combined_infrastructure()
            if len(combined_stations) > 0:
                infrastructure_datasets['charging_infrastructure'] = combined_stations
            
            info(f"Created {len(infrastructure_datasets)} infrastructure datasets", "synthetic_ev_generator")
            
        except Exception as e:
            error(f"Error creating infrastructure datasets: {e}", "synthetic_ev_generator")
        
        return infrastructure_datasets






    def _create_routes_dataframe(self, all_routes: List[Dict]) -> pd.DataFrame:
        """Create routes DataFrame with flattened GPS and consumption data"""
        
        flattened_routes = []
        
        for route in all_routes:
            base_route = {
                'vehicle_id': route['vehicle_id'],
                'trip_id': route['trip_id'],
                'date': route['date'],
                'origin_lat': route['origin'][0],
                'origin_lon': route['origin'][1],
                'destination_lat': route['destination'][0],
                'destination_lon': route['destination'][1],
                'total_distance_km': route['total_distance_km'],
                'total_time_minutes': route['total_time_minutes'],
                'driver_profile': route['driver_profile']
            }
            
            # Add consumption data
            consumption = route['consumption_data']
            base_route.update({
                'total_consumption_kwh': consumption['total_consumption_kwh'],
                'efficiency_kwh_per_100km': consumption['efficiency_kwh_per_100km'],
                'temperature_celsius': consumption['temperature_celsius'],
                'temperature_efficiency_factor': consumption['temperature_efficiency_factor']
            })
            
            # Add consumption breakdown
            breakdown = consumption['consumption_breakdown']
            for component, value in breakdown.items():
                base_route[f'consumption_{component}_kwh'] = value
            
            # Add weather conditions
            weather = consumption['weather_conditions']
            base_route.update({
                'weather_is_raining': weather['is_raining'],
                'weather_wind_speed_kmh': weather['wind_speed_kmh'],
                'weather_humidity': weather['humidity'],
                'weather_season': weather['season']
            })
            
            flattened_routes.append(base_route)
        
        return pd.DataFrame(flattened_routes)
    
    def _print_dataset_summary(self, datasets: Dict[str, pd.DataFrame]):
        """Print summary of generated datasets"""
        
        print("\n" + "="*50)
        print("SYNTHETIC EV FLEET DATASET SUMMARY")
        print("="*50)
        
        for name, df in datasets.items():
            print(f"\n{name.upper()}:")
            print(f"  Rows: {len(df):,}")
            print(f"  Columns: {len(df.columns)}")
            
            if name == 'routes':
                total_distance = df['total_distance_km'].sum()
                total_consumption = df['total_consumption_kwh'].sum()
                avg_efficiency = df['efficiency_kwh_per_100km'].mean()
                print(f"  Total Distance: {total_distance:,.1f} km")
                print(f"  Total Consumption: {total_consumption:,.1f} kWh")
                print(f"  Average Efficiency: {avg_efficiency:.2f} kWh/100km")
            
            elif name == 'charging_sessions':
                total_energy = df['energy_delivered_kwh'].sum()
                total_cost = df['cost_usd'].sum()
                avg_session_time = df['duration_hours'].mean()
                print(f"  Total Energy Delivered: {total_energy:,.1f} kWh")
                print(f"  Total Charging Cost: ${total_cost:,.2f}")
                print(f"  Average Session Duration: {avg_session_time:.1f} hours")
            
            elif name == 'vehicle_states':
                unique_vehicles = df['vehicle_id'].nunique()
                unique_days = df['date'].nunique()
                print(f"  Unique Vehicles: {unique_vehicles}")
                print(f"  Days of Data: {unique_days}")
        
        print("\n" + "="*50)
    
    def save_datasets(self, datasets: Dict[str, pd.DataFrame], 
                 output_dir: str = None) -> Dict[str, str]:
        """Save datasets to files"""
        
        if output_dir is None:
            output_dir = self.config['data_gen']['output_directory']
        
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = {}
        file_formats = self.config['data_gen']['file_formats']
        use_compression = self.config['data_gen']['compression']
        
        for name, df in datasets.items():
            # Clean DataFrame before saving
            df_clean = self._clean_dataframe_for_export(df)
            
            for file_format in file_formats:
                if file_format == 'csv':
                    filename = f"{name}.csv"
                    filepath = os.path.join(output_dir, filename)
                    df_clean.to_csv(filepath, index=False, compression='gzip' if use_compression else None)
                
                elif file_format == 'parquet':
                    filename = f"{name}.parquet"
                    filepath = os.path.join(output_dir, filename)
                    df_clean.to_parquet(filepath, compression='snappy' if use_compression else None)
                
                saved_files[f"{name}_{file_format}"] = filepath
                info(f"Saved {name} dataset to {filepath}", "synthetic_ev_generator")
        
        # Save metadata
        metadata = {
            'generation_date': datetime.now().isoformat(),
            'config': self.config,
            'dataset_summary': {
                name: {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
                }
                for name, df in datasets.items()
            }
        }
        
        metadata_file = os.path.join(output_dir, 'metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        saved_files['metadata'] = metadata_file
        
        return saved_files

    def _clean_dataframe_for_export(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean DataFrame to avoid export issues"""
        df_clean = df.copy()
        
        # Convert all object columns that might have mixed types to string
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                # Check if column has mixed types or contains lists/dicts
                sample_values = df_clean[col].dropna().head(10)
                
                if len(sample_values) > 0:
                    # Check for mixed types or complex objects
                    first_type = type(sample_values.iloc[0])
                    has_mixed_types = any(type(val) != first_type for val in sample_values)
                    has_complex_objects = any(isinstance(val, (list, dict)) for val in sample_values)
                    
                    if has_mixed_types or has_complex_objects:
                        # Convert to string representation
                        df_clean[col] = df_clean[col].astype(str)
                        debug(f"Converted column '{col}' to string due to mixed/complex types", "synthetic_ev_generator")
        
        # Specifically handle known problematic columns
        problematic_columns = ['station_id', 'connector_types', 'gps_trace']
        
        for col in problematic_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str)
        
        return df_clean




# Usage example and testing functions
def main():
    """Main function to generate synthetic EV fleet data"""
    
    # Configuration override for testing
    config_override = {
        'fleet': {'fleet_size': 10},  # Smaller fleet for testing
        'simulation': {'simulation_days': 7}  # One week of data
    }
    
    # Initialize generator
    generator = SyntheticEVGenerator(config_override)
    
    # Log infrastructure statuss
    generator.log_infrastructure_status()

    # Check network status
    network_info = generator.network_db.get_network_info()
    print(f"\n📊 Network Status: {network_info}")

    # Generate complete dataset
    datasets = generator.generate_complete_dataset(num_days=7)
    
    # Save datasets
    saved_files = generator.save_datasets(datasets)
    
    print(f"\nDatasets saved to: {generator.config['data_gen']['output_directory']}")
    print("Files created:")
    for name, filepath in saved_files.items():
        print(f"  {name}: {filepath}")

    try:
        infra_files = generator.infrastructure_manager.export_infrastructure_data(
            generator.config['data_gen']['output_directory']
        )
        print(f"\nInfrastructure data:")
        for name, filepath in infra_files.items():
            print(f"  {name}: {filepath}")
    except Exception as e:
        warning(f"Could not export infrastructure data: {e}", "synthetic_ev_generator")

if __name__ == "__main__":
    main()


