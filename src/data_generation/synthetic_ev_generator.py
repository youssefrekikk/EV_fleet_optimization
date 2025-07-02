"""
Comprehensive synthetic EV fleet data generator
Creates realistic GPS traces, consumption patterns, and charging behavior
"""

import numpy as np
import pandas as pd
import osmnx as ox
import networkx as nx
from datetime import datetime, timedelta
import random
from typing import Dict, List, Tuple, Optional
import json
import os
from geopy.distance import geodesic
import logging

# Import our configurations
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config.ev_config import *

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Key locations in Bay Area for route generation
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

class SyntheticEVGenerator:
    def __init__(self, config_override: Optional[Dict] = None):
        """Initialize the synthetic EV data generator"""
        self.config = self._merge_config(config_override)
        self.road_network = None
        self.charging_stations = []
        self.weather_data = []
        self.fleet_vehicles = []
        
        # Initialize random seed for reproducibility
        np.random.seed(42)
        random.seed(42)
        
        logger.info("Synthetic EV Generator initialized")
    
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
    
    def load_road_network(self) -> nx.MultiDiGraph:
        """Load road network for the Bay Area"""
        logger.info("Loading Bay Area road network...")
        
        try:
            # Try multiple approaches to load the network
            
            # Approach 1: Try with place name first (most reliable)
            try:
                logger.info("Attempting to load network by place name...")
                self.road_network = ox.graph_from_place(
                    "San Francisco Bay Area, California, USA",
                    network_type='drive',
                    simplify=True
                )
                logger.info("Successfully loaded network by place name")
            except Exception as e1:
                logger.warning(f"Place name approach failed: {e1}")
                
                # Approach 2: Try with expanded bounding box
                try:
                    logger.info("Attempting to load network with expanded bounds...")
                    # Use more generous bounds
                    expanded_bounds = {
                        'north': 38.5,
                        'south': 36.8,
                        'east': -120.5,
                        'west': -123.5
                    }
                    
                    self.road_network = ox.graph_from_bbox(
                        north=expanded_bounds['north'],
                        south=expanded_bounds['south'], 
                        east=expanded_bounds['east'],
                        west=expanded_bounds['west'],
                        network_type='drive',
                        simplify=True
                    )
                    logger.info("Successfully loaded network with expanded bounds")
                except Exception as e2:
                    logger.warning(f"Expanded bounds approach failed: {e2}")
                    
                    # Approach 3: Try with specific city
                    try:
                        logger.info("Attempting to load network for San Francisco city...")
                        self.road_network = ox.graph_from_place(
                            "San Francisco, California, USA",
                            network_type='drive',
                            simplify=True
                        )
                        logger.info("Successfully loaded San Francisco network")
                    except Exception as e3:
                        logger.warning(f"San Francisco approach failed: {e3}")
                        raise Exception("All network loading approaches failed")
            
            # Add edge speeds and travel times if network was loaded successfully
            if self.road_network is not None:
                try:
                    self.road_network = ox.add_edge_speeds(self.road_network)
                    self.road_network = ox.add_edge_travel_times(self.road_network)
                except Exception as e:
                    logger.warning(f"Failed to add speeds/times: {e}")
                
                logger.info(f"Loaded road network with {len(self.road_network.nodes)} nodes and {len(self.road_network.edges)} edges")
                return self.road_network
            else:
                raise Exception("Network is None")
                
        except Exception as e:
            logger.error(f"Failed to load road network: {e}")
            # Fallback: create a simplified mock network for testing
            logger.info("Creating simplified mock network for testing...")
            self.road_network = self._create_mock_network()
            return self.road_network




    def _create_mock_network(self) -> nx.MultiDiGraph:
        """Create a simplified mock road network for testing when OSM fails"""
        
        # Create a simple grid network for the Bay Area
        bounds = self.config['geography']
        
        # Create nodes in a grid pattern
        G = nx.MultiDiGraph()
        
        # Grid parameters
        lat_steps = 20
        lon_steps = 20
        
        lats = np.linspace(bounds['south'], bounds['north'], lat_steps)
        lons = np.linspace(bounds['west'], bounds['east'], lon_steps)
        
        node_id = 0
        node_positions = {}
        
        # Create nodes
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                G.add_node(node_id, y=lat, x=lon)
                node_positions[(i, j)] = node_id
                node_id += 1
        
        # Create edges (connect adjacent nodes)
        for i in range(lat_steps):
            for j in range(lon_steps):
                current_node = node_positions[(i, j)]
                
                # Connect to right neighbor
                if j < lon_steps - 1:
                    right_node = node_positions[(i, j + 1)]
                    distance = geodesic(
                        (lats[i], lons[j]), 
                        (lats[i], lons[j + 1])
                    ).meters
                    
                    G.add_edge(current_node, right_node, 0, 
                              length=distance, 
                              speed_kph=50, 
                              travel_time=distance / (50 * 1000 / 3600))
                    G.add_edge(right_node, current_node, 0, 
                              length=distance, 
                              speed_kph=50, 
                              travel_time=distance / (50 * 1000 / 3600))
                
                # Connect to bottom neighbor
                if i < lat_steps - 1:
                    bottom_node = node_positions[(i + 1, j)]
                    distance = geodesic(
                        (lats[i], lons[j]), 
                        (lats[i + 1], lons[j])
                    ).meters
                    
                    G.add_edge(current_node, bottom_node, 0, 
                              length=distance, 
                              speed_kph=50, 
                              travel_time=distance / (50 * 1000 / 3600))
                    G.add_edge(bottom_node, current_node, 0, 
                              length=distance, 
                              speed_kph=50, 
                              travel_time=distance / (50 * 1000 / 3600))
        
        logger.info(f"Created mock network with {len(G.nodes)} nodes and {len(G.edges)} edges")
        return G

    


    def generate_fleet_vehicles(self) -> List[Dict]:
        """Generate fleet of vehicles with realistic characteristics"""
        logger.info("Generating fleet vehicles...")
        
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
            # Select driving style (behavioral pattern) - separate from profile
            driving_style = self._select_driving_style()
            # Determine home charging access
            has_home_charging = (
                home_charging_enabled and 
                np.random.random() < home_charging_availability
            )
            # Generate vehicle-specific characteristics
            vehicle = {
                'vehicle_id': f'EV_{vehicle_id:03d}',
                'model': model_name,
                'battery_capacity': model_specs['battery_capacity'],
                'efficiency': model_specs['efficiency'] * np.random.normal(1.0, 0.05),  # 5% variation
                'max_charging_speed': model_specs['max_charging_speed'],
                'driver_profile': driver_profile,
                'driving_style': driving_style,
                'has_home_charging': has_home_charging,
                'home_location': self._generate_home_location(),
                'current_battery_soc': np.random.uniform(0.3, 0.9),  # Start with random charge
                'odometer': np.random.uniform(0, 50000),  # km
                'last_service': datetime.now() - timedelta(days=np.random.randint(0, 365))
            }
            
            vehicles.append(vehicle)
        
        self.fleet_vehicles = vehicles
        logger.info(f"Generated {len(vehicles)} fleet vehicles")
        logger.info(f"Home charging enabled: {home_charging_enabled}")
        logger.info(f"Vehicles with home charging: {sum(1 for v in vehicles if v['has_home_charging'])}")
        return vehicles
    
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
    
    def _select_driving_style(self) -> str:
        """Select driving style based on proportions"""
        styles = list(DRIVING_STYLES.keys())
        weights = [DRIVING_STYLES[style]['proportion'] for style in styles]
        return np.random.choice(styles, p=weights)


    def _generate_home_location(self) -> Tuple[float, float]:
        """Generate realistic home location"""
        # Bias towards residential areas
        residential_areas = [
            'daly_city', 'hayward', 'fremont', 'mountain_view', 
            'sunnyvale', 'santa_clara'
        ]
        
        if np.random.random() < 0.7:  # 70% in residential areas
            base_location = MAJOR_LOCATIONS[np.random.choice(residential_areas)]
        else:
            base_location = random.choice(list(MAJOR_LOCATIONS.values()))
        
        # Add random offset (within ~5km)
        lat_offset = np.random.normal(0, 0.02)
        lon_offset = np.random.normal(0, 0.02)
        
        return (base_location[0] + lat_offset, base_location[1] + lon_offset)
    
    def _generate_work_location(self) -> Tuple[float, float]:
        """Generate realistic work location"""
        # Bias towards business districts
        business_areas = [
            'downtown_sf', 'silicon_valley', 'palo_alto', 'san_jose'
        ]
        
        if np.random.random() < 0.8:  # 80% in business areas
            base_location = MAJOR_LOCATIONS[np.random.choice(business_areas)]
        else:
            base_location = random.choice(list(MAJOR_LOCATIONS.values()))
        
        # Add random offset
        lat_offset = np.random.normal(0, 0.01)
        lon_offset = np.random.normal(0, 0.01)
        
        return (base_location[0] + lat_offset, base_location[1] + lon_offset)

    def generate_daily_routes(self, vehicle: Dict, date: datetime) -> List[Dict]:
        """Generate realistic daily routes for a vehicle"""
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
        
        # Generate individual trips
        current_location = vehicle['home_location']
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
            
            # Generate route
            route = self._generate_route(
                current_location, destination, vehicle, date, trip_id
            )
            
            if route:
                routes.append(route)
                current_location = destination
                remaining_km -= trip_distance
        
        return routes
    
    def _generate_trip_destination(self, origin: Tuple[float, float], 
                                 target_distance: float, driver_profile: str, 
                                 is_weekend: bool) -> Tuple[float, float]:
        """Generate realistic trip destination"""
        
        if driver_profile == 'commuter' and not is_weekend:
            # Commuters go to work and back
            if np.random.random() < 0.6:
                return self._get_nearby_business_location(origin, target_distance)
        
        elif driver_profile == 'delivery':
            # Delivery vehicles go to commercial/residential areas
            return self._get_delivery_destination(origin, target_distance)
        
        elif driver_profile == 'rideshare':
            # Rideshare goes to popular destinations
            return self._get_popular_destination(origin, target_distance)
        
        # Default: random destination within target distance
        return self._get_random_destination_within_distance(origin, target_distance)
    
    def _get_nearby_business_location(self, origin: Tuple[float, float], 
                                    target_distance: float) -> Tuple[float, float]:
        """Get business location within target distance"""
        business_areas = ['downtown_sf', 'silicon_valley', 'palo_alto', 'san_jose']
        
        # Find business areas within reasonable distance
        suitable_areas = []
        for area_name in business_areas:
            area_location = MAJOR_LOCATIONS[area_name]
            distance = geodesic(origin, area_location).kilometers
            if abs(distance - target_distance) < target_distance * 0.3:  # Within 30% of target
                suitable_areas.append(area_location)
        
        if suitable_areas:
            base_location = random.choice(suitable_areas)
        else:
            base_location = random.choice([MAJOR_LOCATIONS[area] for area in business_areas])
        
        # Add small random offset
        lat_offset = np.random.normal(0, 0.005)
        lon_offset = np.random.normal(0, 0.005)
        
        return (base_location[0] + lat_offset, base_location[1] + lon_offset)
    
    def _get_delivery_destination(self, origin: Tuple[float, float], 
                                target_distance: float) -> Tuple[float, float]:
        """Get delivery destination (commercial/residential mix)"""
        # 60% residential, 40% commercial
        if np.random.random() < 0.6:
            # Residential delivery
            residential_areas = ['daly_city', 'hayward', 'fremont', 'mountain_view']
            base_location = MAJOR_LOCATIONS[np.random.choice(residential_areas)]
        else:
            # Commercial delivery
            commercial_areas = ['downtown_sf', 'oakland', 'san_jose']
            base_location = MAJOR_LOCATIONS[np.random.choice(commercial_areas)]
        
        # Larger offset for delivery (covering more area)
        lat_offset = np.random.normal(0, 0.01)
        lon_offset = np.random.normal(0, 0.01)
        
        return (base_location[0] + lat_offset, base_location[1] + lon_offset)
    
    def _get_popular_destination(self, origin: Tuple[float, float], 
                               target_distance: float) -> Tuple[float, float]:
        """Get popular rideshare destination"""
        popular_areas = ['downtown_sf', 'silicon_valley', 'oakland', 'berkeley']
        base_location = MAJOR_LOCATIONS[np.random.choice(popular_areas)]
        
        # Medium offset for rideshare
        lat_offset = np.random.normal(0, 0.008)
        lon_offset = np.random.normal(0, 0.008)
        
        return (base_location[0] + lat_offset, base_location[1] + lon_offset)
    
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
        
        # Ensure destination is within geographic bounds
        bounds = self.config['geography']
        destination = (
            np.clip(destination[0], bounds['south'], bounds['north']),
            np.clip(destination[1], bounds['west'], bounds['east'])
        )
        
        return destination
    
    def _find_nearest_node(self, lat: float, lon: float) -> int:
        """Find nearest node in the network (with fallback for mock network)"""
        try:
            # Try OSMnx method first
            return ox.nearest_nodes(self.road_network, lon, lat)
        except:
            # Fallback: find closest node manually
            min_distance = float('inf')
            nearest_node = None
            
            for node_id, node_data in self.road_network.nodes(data=True):
                node_lat = node_data['y']
                node_lon = node_data['x']
                distance = geodesic((lat, lon), (node_lat, node_lon)).meters
                
                if distance < min_distance:
                    min_distance = distance
                    nearest_node = node_id
            
            return nearest_node
    
    def _generate_route(self, origin: Tuple[float, float], destination: Tuple[float, float],
                       vehicle: Dict, date: datetime, trip_id: int) -> Dict:
        """Generate realistic route using OSM road network"""
        
        # Ensure road network is loaded
        if self.road_network is None:
            logger.warning("Road network not loaded. Loading now...")
            self.load_road_network()
        
        # Double check that we have a network
        if self.road_network is None:
            logger.error("Failed to load any road network")
            return None
        
        try:
            # Find nearest nodes in road network
            origin_node = self._find_nearest_node(origin[0], origin[1])
            dest_node = self._find_nearest_node(destination[0], destination[1])
            
            if origin_node is None or dest_node is None:
                logger.error("Could not find nearest nodes")
                return None
            
            # Calculate shortest path
            try:
                path = nx.shortest_path(
                    self.road_network, origin_node, dest_node, weight='travel_time'
                )
            except nx.NetworkXNoPath:
                logger.warning(f"No path found between {origin} and {destination}")
                # Try without weight
                try:
                    path = nx.shortest_path(self.road_network, origin_node, dest_node)
                except nx.NetworkXNoPath:
                    logger.error("No path found even without weight")
                    return None
            
            # Extract route details
            route_coords = []
            route_speeds = []
            route_elevations = []
            total_distance = 0
            total_time = 0
            
            for i in range(len(path) - 1):
                node1, node2 = path[i], path[i + 1]
                
                # Get node coordinates
                node1_data = self.road_network.nodes[node1]
                node2_data = self.road_network.nodes[node2]
                
                coord1 = (node1_data['y'], node1_data['x'])
                coord2 = (node2_data['y'], node2_data['x'])
                
                route_coords.extend([coord1, coord2])
                
                # Get edge data (handle both OSM and mock networks)
                try:
                    edge_data = self.road_network.edges[node1, node2, 0]
                except KeyError:
                    # Fallback for mock network
                    try:
                        edge_data = list(self.road_network[node1][node2].values())[0]
                    except (KeyError, IndexError):
                        # Create default edge data
                        edge_data = {
                            'length': geodesic(coord1, coord2).meters,
                            'speed_kph': 50,
                            'travel_time': geodesic(coord1, coord2).meters / (50 * 1000 / 3600)
                        }
                
                # Speed and distance
                speed_kmh = edge_data.get('speed_kph', 50)  # Default 50 km/h
                distance_m = edge_data.get('length', geodesic(coord1, coord2).meters)   # Calculate if not available
                
                route_speeds.append(speed_kmh)
                total_distance += distance_m / 1000  # Convert to km
                total_time += edge_data.get('travel_time', distance_m / (speed_kmh * 1000 / 3600))
                
                # Elevation (simplified - could use elevation API)
                elevation = np.random.normal(50, 20)  # Simplified elevation model
                route_elevations.append(elevation)
            
            # Generate realistic GPS trace with time stamps
            gps_trace = self._generate_gps_trace(
                route_coords, route_speeds, route_elevations, date, trip_id
            )
            
            # Calculate energy consumption
            consumption_data = self._calculate_energy_consumption(
                gps_trace, vehicle, date
            )
            
            route_data = {
                'vehicle_id': vehicle['vehicle_id'],
                'trip_id': f"{vehicle['vehicle_id']}_{date.strftime('%Y%m%d')}_{trip_id:02d}",
                'date': date.strftime('%Y-%m-%d'),
                'origin': origin,
                'destination': destination,
                'total_distance_km': total_distance,
                'total_time_minutes': total_time / 60,
                'gps_trace': gps_trace,
                'consumption_data': consumption_data,
                'driver_profile': vehicle['driver_profile']
            }
            
            return route_data
            
        except Exception as e:
            logger.error(f"Error generating route: {e}")
            import traceback
            traceback.print_exc()
            return None
    





    def _generate_gps_trace(self, route_coords: List[Tuple[float, float]], 
                           route_speeds: List[float], route_elevations: List[float],
                           date: datetime, trip_id: int) -> List[Dict]:
        """Generate realistic GPS trace with timestamps"""
        
        gps_trace = []
        current_time = date
        
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
        """Calculate realistic energy consumption based on physics"""
        
        if not gps_trace or len(gps_trace) < 2:
            return {
                'total_consumption_kwh': 0, 
                'total_distance_km': 0,
                'efficiency_kwh_per_100km': 0,
                'temperature_celsius': 20,
                'temperature_efficiency_factor': 1.0,
                'consumption_breakdown': {
                    'rolling_resistance': 0,
                    'aerodynamic_drag': 0,
                    'elevation_change': 0,
                    'acceleration': 0,
                    'hvac': 0,
                    'auxiliary': 0,
                    'regenerative_braking': 0
                },
                'weather_conditions': {
                    'temperature': 20,
                    'is_raining': False,
                    'wind_speed_kmh': 15,
                    'humidity': 0.65,
                    'season': 'spring'
                }
            }
        
        # Get vehicle specifications
        model_specs = EV_MODELS[vehicle['model']]
        base_efficiency = model_specs['efficiency']  # kWh/100km
        vehicle_mass = model_specs['weight']  # kg
        drag_coeff = model_specs['drag_coefficient']
        frontal_area = model_specs['frontal_area']  # m²
        
        # Physics constants
        air_density = PHYSICS_CONSTANTS['air_density']
        rolling_resistance = PHYSICS_CONSTANTS['rolling_resistance']
        gravity = PHYSICS_CONSTANTS['gravity']
        motor_efficiency = PHYSICS_CONSTANTS['motor_efficiency']
        regen_efficiency = PHYSICS_CONSTANTS['regen_efficiency']
        
        total_consumption = 0
        total_distance = 0
        consumption_breakdown = {
            'rolling_resistance': 0,
            'aerodynamic_drag': 0,
            'elevation_change': 0,
            'acceleration': 0,
            'hvac': 0,
            'auxiliary': 0,
            'regenerative_braking': 0
        }
        
        # Get weather conditions for the day
        weather = self._get_weather_conditions(date)
        temperature = weather['temperature']
        
        # Temperature efficiency factor
        temp_efficiency = self._get_temperature_efficiency(temperature)
        
        for i in range(len(gps_trace) - 1):
            current_point = gps_trace[i]
            next_point = gps_trace[i + 1]
            
            # Calculate segment distance and time
            distance_m = geodesic(
                (current_point['latitude'], current_point['longitude']),
                (next_point['latitude'], next_point['longitude'])
            ).meters
            
            if distance_m == 0:
                continue
            
            # Parse timestamps
            time1 = datetime.fromisoformat(current_point['timestamp'])
            time2 = datetime.fromisoformat(next_point['timestamp'])
            time_diff_s = (time2 - time1).total_seconds()
            
            if time_diff_s <= 0:
                continue
            
            # Calculate speeds and acceleration
            speed_ms = current_point['speed_kmh'] / 3.6  # Convert to m/s
            next_speed_ms = next_point['speed_kmh'] / 3.6
            acceleration = (next_speed_ms - speed_ms) / time_diff_s
            
            # Elevation change
            elevation_change = next_point['elevation_m'] - current_point['elevation_m']
            
            # Energy calculations (in Joules, then convert to kWh)
            
            # 1. Rolling resistance
            rolling_energy = rolling_resistance * vehicle_mass * gravity * distance_m
            
            # 2. Aerodynamic drag
            drag_energy = 0.5 * air_density * drag_coeff * frontal_area * (speed_ms ** 2) * distance_m
            
            # 3. Elevation change (potential energy)
            elevation_energy = vehicle_mass * gravity * elevation_change
            
            # 4. Acceleration energy (kinetic energy change)
            kinetic_energy_change = 0.5 * vehicle_mass * (next_speed_ms ** 2 - speed_ms ** 2)
            
            # 5. HVAC energy (based on temperature)
            hvac_power = self._calculate_hvac_power(temperature)
            hvac_energy = hvac_power * 1000 * time_diff_s  # Convert kW to W, then to Joules
            
            # 6. Auxiliary systems (lights, electronics, etc.)
            aux_energy = PHYSICS_CONSTANTS['auxiliary_power'] * 1000 * time_diff_s
            
            # Total energy required (before efficiency losses)
            segment_energy = rolling_energy + drag_energy + hvac_energy + aux_energy
            
            # Handle elevation and acceleration separately (can be negative)
            if elevation_change > 0:  # Going uphill
                segment_energy += elevation_energy
                consumption_breakdown['elevation_change'] += elevation_energy / 3.6e6  # Convert to kWh
            else:  # Going downhill - regenerative braking
                regen_energy = abs(elevation_energy) * regen_efficiency
                segment_energy -= regen_energy
                consumption_breakdown['regenerative_braking'] -= regen_energy / 3.6e6
            
            if acceleration > 0:  # Accelerating
                segment_energy += kinetic_energy_change
                consumption_breakdown['acceleration'] += kinetic_energy_change / 3.6e6
            else:  # Decelerating - regenerative braking
                regen_energy = abs(kinetic_energy_change) * regen_efficiency
                segment_energy -= regen_energy
                consumption_breakdown['regenerative_braking'] -= regen_energy / 3.6e6
            
            # Apply motor efficiency and temperature effects
            segment_energy_kwh = (segment_energy / 3.6e6) / motor_efficiency / temp_efficiency
            
            # Ensure minimum consumption (can't be negative overall)
            segment_energy_kwh = max(0, segment_energy_kwh)
            
            total_consumption += segment_energy_kwh
            total_distance += distance_m / 1000  # Convert to km
            
            # Update consumption breakdown
            consumption_breakdown['rolling_resistance'] += rolling_energy / 3.6e6
            consumption_breakdown['aerodynamic_drag'] += drag_energy / 3.6e6
            consumption_breakdown['hvac'] += hvac_energy / 3.6e6
            consumption_breakdown['auxiliary'] += aux_energy / 3.6e6
        
        # Calculate efficiency
        efficiency_kwh_per_100km = (total_consumption / total_distance * 100) if total_distance > 0 else 0
        
        return {
            'total_consumption_kwh': round(total_consumption, 3),
            'total_distance_km': round(total_distance, 2),
            'efficiency_kwh_per_100km': round(efficiency_kwh_per_100km, 2),
            'temperature_celsius': temperature,
            'temperature_efficiency_factor': temp_efficiency,
            'consumption_breakdown': {k: round(v, 4) for k, v in consumption_breakdown.items()},
            'weather_conditions': weather
        }
    
    def _get_temperature_efficiency(self, temperature: float) -> float:
        """Get battery efficiency factor based on temperature"""
        # Interpolate between known temperature points
        temp_points = sorted(TEMPERATURE_EFFICIENCY.keys())
        
        if temperature <= temp_points[0]:
            return TEMPERATURE_EFFICIENCY[temp_points[0]]
        elif temperature >= temp_points[-1]:
            return TEMPERATURE_EFFICIENCY[temp_points[-1]]
        
        # Linear interpolation
        for i in range(len(temp_points) - 1):
            if temp_points[i] <= temperature <= temp_points[i + 1]:
                t1, t2 = temp_points[i], temp_points[i + 1]
                eff1, eff2 = TEMPERATURE_EFFICIENCY[t1], TEMPERATURE_EFFICIENCY[t2]
                
                # Linear interpolation
                factor = (temperature - t1) / (t2 - t1)
                return eff1 + factor * (eff2 - eff1)
        
        return 1.0  # Default efficiency
    
    def _calculate_hvac_power(self, temperature: float) -> float:
        """Calculate HVAC power consumption based on temperature"""
        optimal_temp = 22  # Celsius
        temp_diff = abs(temperature - optimal_temp)
        
        base_power = PHYSICS_CONSTANTS['hvac_base_power']
        
        if temp_diff <= 2:
            return base_power * 0.3  # Minimal HVAC usage
        elif temp_diff <= 5:
            return base_power * 0.6
        elif temp_diff <= 10:
            return base_power * 1.0
        else:
            return base_power * 1.5  # Maximum HVAC usage
    
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
        
        return {
            'temperature': round(daily_temp, 1),
            'is_raining': is_raining,
            'wind_speed_kmh': max(0, round(wind_speed, 1)),
            'humidity': np.clip(humidity, 0.2, 0.95),
            'season': self._get_season(date)
        }
    
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
    
    def generate_charging_sessions(self, vehicle: Dict, routes: List[Dict], date: datetime) -> List[Dict]:
        """Generate realistic charging sessions for a vehicle"""
        charging_sessions = []
        driver_profile = DRIVER_PROFILES[vehicle['driver_profile']]
        driving_style = DRIVING_STYLES[vehicle['driving_style']]
        
        # Track battery state throughout the day
        current_soc = vehicle['current_battery_soc']
        battery_capacity = vehicle['battery_capacity']
        
        # Use driving style for charging threshold instead of driver profile
        charging_threshold = driving_style['charging_threshold']
        
        # Check if vehicle needs charging at start of day (only if has home charging)
        if (vehicle['has_home_charging'] and 
            current_soc < charging_threshold and 
            np.random.random() < 0.8):  # 80% chance to charge at home if available
            
            # Home charging session
            home_session = self._generate_home_charging_session(
                vehicle, date, current_soc
            )
            charging_sessions.append(home_session)
            current_soc = home_session['end_soc']
        
        # Process each route and check for charging needs
        for route in routes:
            # Consume energy for this route
            consumption_kwh = route['consumption_data']['total_consumption_kwh']
            energy_consumed_percent = consumption_kwh / battery_capacity
            current_soc -= energy_consumed_percent
            
            # Check if charging is needed
            needs_charging = current_soc < charging_threshold
            
            # Opportunistic charging (even if not needed) - less likely if has home charging
            opportunistic_prob = 0.1 if vehicle['has_home_charging'] else 0.3
            opportunistic_charging = (
                current_soc < 0.7 and  # Less than 70%
                np.random.random() < opportunistic_prob and
                vehicle['driver_profile'] == 'rideshare'  # More likely for rideshare
            )
            
            if needs_charging or opportunistic_charging:
                # Find nearby charging station
                charging_session = self._generate_public_charging_session(
                    vehicle, route['destination'], date, current_soc, needs_charging
                )
                
                if charging_session:
                    charging_sessions.append(charging_session)
                    current_soc = charging_session['end_soc']
        
        # End of day home charging (only if has home charging)
        if (vehicle['has_home_charging'] and 
            current_soc < 0.8 and 
            np.random.random() < 0.9):  # High probability to charge at home overnight
            
            end_day_session = self._generate_home_charging_session(
                vehicle, date + timedelta(hours=20), current_soc
            )
            charging_sessions.append(end_day_session)
        
        return charging_sessions


    
    def _generate_home_charging_session(self, vehicle: Dict, start_time: datetime, 
                                      start_soc: float) -> Dict:
        """Generate home charging session"""
        
        # Home charging is typically overnight, slow charging
        charging_power = self.config['charging']['home_charging_power']  # 7.4 kW
        battery_capacity = vehicle['battery_capacity']
        
        # Charge to 80-90% (typical home charging behavior)
        target_soc = np.random.uniform(0.8, 0.9)
        energy_needed = (target_soc - start_soc) * battery_capacity
        
        # Calculate charging time (with charging curve - slower as battery fills)
        charging_time_hours = self._calculate_charging_time(
            energy_needed, charging_power, start_soc, target_soc
        )
        
        # Cost calculation
        electricity_cost = self.config['charging']['base_electricity_cost']
        total_cost = energy_needed * electricity_cost
        
        return {
            'session_id': f"{vehicle['vehicle_id']}_{start_time.strftime('%Y%m%d_%H%M')}_home",
            'vehicle_id': vehicle['vehicle_id'],
            'charging_type': 'home',
            'location': f"({float(vehicle['home_location'][0]):.6f}, {float(vehicle['home_location'][1]):.6f})",
            'start_time': start_time.isoformat(),
            'end_time': (start_time + timedelta(hours=charging_time_hours)).isoformat(),
            'start_soc': round(start_soc, 3),
            'end_soc': round(target_soc, 3),
            'energy_delivered_kwh': round(energy_needed, 2),
            'charging_power_kw': charging_power,
            'duration_hours': round(charging_time_hours, 2),
            'cost_usd': round(total_cost, 2),
            'cost_per_kwh': electricity_cost
        }
    
    def _generate_public_charging_session(self, vehicle: Dict, location: Tuple[float, float],
                                        date: datetime, start_soc: float, 
                                        is_emergency: bool) -> Optional[Dict]:
        """Generate public charging session"""
        
        # Find nearby charging stations (using our OpenChargeMap data)
        nearby_stations = self._find_nearby_charging_stations(location, radius_km=10)
        
        if not nearby_stations:
            return None
        
        # Select charging station based on preferences
        selected_station = self._select_charging_station(
            nearby_stations, vehicle, is_emergency
        )
        
        # Determine charging parameters
        if is_emergency:
            # Emergency charging - charge to 80%
            target_soc = 0.8
            charging_power = min(
                selected_station.get('max_power_kw', 50),
                vehicle['max_charging_speed']
            )
        else:
            # Opportunistic charging - partial charge
            target_soc = np.random.uniform(0.6, 0.8)
            charging_power = min(
                selected_station.get('max_power_kw', 22),
                vehicle['max_charging_speed']
            ) * 0.8  # Don't always use max power
        
        battery_capacity = vehicle['battery_capacity']
        energy_needed = (target_soc - start_soc) * battery_capacity
        
        # Calculate charging time
        charging_time_hours = self._calculate_charging_time(
            energy_needed, charging_power, start_soc, target_soc
        )
        
        # Cost calculation (public charging is more expensive)
        base_cost = selected_station.get('cost_usd_per_kwh', 0.35)
        
        # Peak hour pricing
        hour = date.hour
        is_peak_hour = any(start <= hour <= end for start, end in self.config['charging']['peak_hours'])
        if is_peak_hour:
            cost_per_kwh = base_cost * self.config['charging']['peak_pricing_multiplier']
        else:
            cost_per_kwh = base_cost
        
        total_cost = energy_needed * cost_per_kwh
        
        return {
            'session_id': f"{vehicle['vehicle_id']}_{date.strftime('%Y%m%d_%H%M')}_public",
            'vehicle_id': vehicle['vehicle_id'],
            'charging_type': 'public',
            'station_id': selected_station.get('ocm_id', 'unknown'),
            'station_operator': selected_station.get('operator', 'Unknown'),
            'location': f"({float(selected_station.get('latitude', 0)):.6f}, {float(selected_station.get('longitude', 0)):.6f})",
            'start_time': date.isoformat(),
            'end_time': (date + timedelta(hours=charging_time_hours)).isoformat(),
            'start_soc': round(start_soc, 3),
            'end_soc': round(target_soc, 3),
            'energy_delivered_kwh': round(energy_needed, 2),
            'charging_power_kw': charging_power,
            'duration_hours': round(charging_time_hours, 2),
            'cost_usd': round(total_cost, 2),
            'cost_per_kwh': round(cost_per_kwh, 3),
            'is_emergency_charging': is_emergency,
            'connector_type': selected_station.get('connector_types', ['Unknown'])[0]
        }
    
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
    
    def _find_nearby_charging_stations(self, location: Tuple[float, float], 
                                     radius_km: int = 10) -> List[Dict]:
        """Find nearby charging stations (mock implementation - would use real data)"""
        # This would integrate with our OpenChargeMap API
        # For now, generate some realistic mock stations
        
        num_stations = np.random.poisson(3)  # Average 3 stations nearby
        stations = []
        
        for i in range(num_stations):
            # Generate station within radius
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.uniform(0.5, radius_km)
            
            # Convert to lat/lon offset
            lat_offset = (distance / 111.0) * np.cos(angle)
            lon_offset = (distance / 111.0) * np.sin(angle)
            
            station_location = (
                location[0] + lat_offset,
                location[1] + lon_offset
            )
            
            # Generate realistic station characteristics
            station_types = ['AC Level 2', 'DC Fast Charger', 'Tesla Supercharger']
            station_type = np.random.choice(station_types, p=[0.5, 0.4, 0.1])
            
            if station_type == 'AC Level 2':
                max_power = np.random.choice([7.4, 11, 22], p=[0.6, 0.3, 0.1])
                cost_per_kwh = np.random.uniform(0.25, 0.35)
            elif station_type == 'DC Fast Charger':
                max_power = np.random.choice([50, 75, 100, 150], p=[0.3, 0.3, 0.2, 0.2])
                cost_per_kwh = np.random.uniform(0.35, 0.50)
            else:  # Tesla Supercharger
                max_power = np.random.choice([150, 250], p=[0.4, 0.6])
                cost_per_kwh = np.random.uniform(0.30, 0.45)
            
            station = {
                'ocm_id': f'mock_{i}_{int(location[0]*1000)}_{int(location[1]*1000)}',
                'latitude': station_location[0],
                'longitude': station_location[1],
                'operator': np.random.choice(['ChargePoint', 'EVgo', 'Electrify America', 'Tesla']),
                'max_power_kw': max_power,
                'cost_usd_per_kwh': cost_per_kwh,
                'connector_types': [station_type],
                'availability': np.random.choice(['Available', 'Busy'], p=[0.7, 0.3]),
                'distance_km': distance
            }
            
            stations.append(station)
        
        return sorted(stations, key=lambda x: x['distance_km'])
    
    def _select_charging_station(self, stations: List[Dict], vehicle: Dict, 
                               is_emergency: bool) -> Dict:
        """Select best charging station based on preferences"""
        
        if not stations:
            return stations[0] if stations else {}
        
        # Filter available stations
        available_stations = [s for s in stations if s['availability'] == 'Available']
        if not available_stations:
            available_stations = stations  # Use any station if none available
        
        if is_emergency:
            # Emergency: prioritize fast charging
            fast_stations = [s for s in available_stations if s['max_power_kw'] >= 50]
            if fast_stations:
                return min(fast_stations, key=lambda x: x['distance_km'])
        
        # Normal selection: balance distance, cost, and charging speed
        def station_score(station):
            distance_score = 1 / (1 + station['distance_km'])  # Closer is better
            cost_score = 1 / (1 + station['cost_usd_per_kwh'])  # Cheaper is better
            power_score = station['max_power_kw'] / 250  # Higher power is better
            
            return distance_score * 0.4 + cost_score * 0.3 + power_score * 0.3
        
        return max(available_stations, key=station_score)
    


    def generate_complete_dataset(self, num_days: int = 30) -> Dict[str, pd.DataFrame]:
        """Generate complete synthetic EV fleet dataset"""
        
        logger.info(f"Generating complete dataset for {num_days} days...")
        
        # Initialize data structures
        all_routes = []
        all_charging_sessions = []
        all_vehicle_states = []
        all_weather_data = []
        
        # Load road network ONCE at the beginning
        logger.info("Loading road network...")
        self.load_road_network()
        
        # Generate fleet vehicles
        if not self.fleet_vehicles:
            self.generate_fleet_vehicles()
        
        # Generate data for each day
        start_date = datetime.strptime(self.config['fleet']['start_date'], '%Y-%m-%d')
        
        for day in range(num_days):
            current_date = start_date + timedelta(days=day)
            logger.info(f"Generating data for day {day + 1}/{num_days}: {current_date.strftime('%Y-%m-%d')}")
            
            # Generate weather for the day
            daily_weather = self._get_weather_conditions(current_date)
            daily_weather['date'] = current_date.strftime('%Y-%m-%d')
            all_weather_data.append(daily_weather)
            
            # Generate data for each vehicle
            for vehicle in self.fleet_vehicles:
                # Generate daily routes
                daily_routes = self.generate_daily_routes(vehicle, current_date)
                
                # Filter out None routes
                daily_routes = [route for route in daily_routes if route is not None]
                all_routes.extend(daily_routes)
                
                # Generate charging sessions
                charging_sessions = self.generate_charging_sessions(
                    vehicle, daily_routes, current_date
                )
                all_charging_sessions.extend(charging_sessions)
                
                # Track vehicle state
                total_distance = sum(route['total_distance_km'] for route in daily_routes)
                total_consumption = sum(
                    route['consumption_data']['total_consumption_kwh'] 
                    for route in daily_routes
                )
                
                vehicle_state = {
                    'vehicle_id': vehicle['vehicle_id'],
                    'date': current_date.strftime('%Y-%m-%d'),
                    'total_distance_km': round(total_distance, 2),
                    'total_consumption_kwh': round(total_consumption, 3),
                    'efficiency_kwh_per_100km': round(
                        (total_consumption / total_distance * 100) if total_distance > 0 else 0, 2
                    ),
                    'num_trips': len(daily_routes),
                    'num_charging_sessions': len(charging_sessions),
                    'driver_profile': vehicle['driver_profile'],
                    'vehicle_model': vehicle['model']
                }
                all_vehicle_states.append(vehicle_state)
        
        # Convert to DataFrames
        datasets = {
            'routes': self._create_routes_dataframe(all_routes),
            'charging_sessions': pd.DataFrame(all_charging_sessions),
            'vehicle_states': pd.DataFrame(all_vehicle_states),
            'weather': pd.DataFrame(all_weather_data),
            'fleet_info': pd.DataFrame(self.fleet_vehicles)
        }
        
        logger.info("Dataset generation complete!")
        self._print_dataset_summary(datasets)
        
        return datasets





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
            for file_format in file_formats:
                if file_format == 'csv':
                    filename = f"{name}.csv"
                    filepath = os.path.join(output_dir, filename)
                    df.to_csv(filepath, index=False, compression='gzip' if use_compression else None)
                
                elif file_format == 'parquet':
                    filename = f"{name}.parquet"
                    filepath = os.path.join(output_dir, filename)
                    df.to_parquet(filepath, compression='snappy' if use_compression else None)
                
                saved_files[f"{name}_{file_format}"] = filepath
                logger.info(f"Saved {name} dataset to {filepath}")
        
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
    
    # Generate complete dataset
    datasets = generator.generate_complete_dataset(num_days=7)
    
    # Save datasets
    saved_files = generator.save_datasets(datasets)
    
    print(f"\nDatasets saved to: {generator.config['data_gen']['output_directory']}")
    print("Files created:")
    for name, filepath in saved_files.items():
        print(f"  {name}: {filepath}")

if __name__ == "__main__":
    main()


