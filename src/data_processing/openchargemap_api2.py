import requests
import pandas as pd
import pickle
import time
import json
import numpy as np
import os
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dotenv import load_dotenv
from geopy.distance import geodesic
import sys

# Import our charging infrastructure config
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config.charging_infrastructure_config import (
    BAY_AREA_BOUNDS, BAY_AREA_REGIONS, INFRASTRUCTURE_SCENARIOS,
    MOCK_STATION_CONFIG, CAPACITY_MODELING, HOME_CHARGING_CONFIG,
    API_CONFIG, DATA_PATHS
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

class ChargingInfrastructureManager:
    """
    Manages comprehensive Bay Area charging infrastructure:
    1. Real stations (one-time build from OpenChargeMap)
    2. Mock stations for critical gaps (persistent across simulations)
    3. Home stations (generated per simulation)
    4. Capacity and availability modeling
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('OPENCHARGEMAP_API_KEY')
        
        # Use the single scenario configuration
        self.scenario_config = INFRASTRUCTURE_SCENARIOS
        self.scenario_name = self.scenario_config["name"]
        self.scenario = self.scenario_config["name"]
        # Set up data directory
        self.data_dir = DATA_PATHS['base_dir']
        os.makedirs(self.data_dir, exist_ok=True)
        
        # File paths (remove scenario suffix since we have one scenario)
        self.real_stations_csv = os.path.join(self.data_dir, DATA_PATHS['real_stations_csv'])
        self.real_stations_pkl = os.path.join(self.data_dir, DATA_PATHS['real_stations_pkl'])
        self.mock_stations_csv = os.path.join(self.data_dir, DATA_PATHS['mock_stations_csv'])
        self.mock_stations_pkl = os.path.join(self.data_dir, DATA_PATHS['mock_stations_pkl'])
        self.coverage_analysis_file = os.path.join(self.data_dir, DATA_PATHS['coverage_analysis'])
        
        # API setup (unchanged)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'EV-Fleet-Optimizer/1.0',
            'Accept': 'application/json'
        })
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = API_CONFIG['openchargemap']['rate_limit_seconds']
        
        # Station databases (loaded on demand)
        self._real_stations_df = None
        self._mock_stations_df = None
        self._combined_stations_df = None
        
        logger.info(f"ChargingInfrastructureManager initialized for: {self.scenario_name}")
    
    def _rate_limit(self):
        """Implement rate limiting for API requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()
    
    def find_nearby_stations_api(self, latitude: float, longitude: float, 
                                distance_km: int = 15, max_results: int = 100) -> List[Dict]:
        """Find charging stations via OpenChargeMap API"""
        if not self.api_key:
            logger.warning("No OpenChargeMap API key available")
            return []
        
        self._rate_limit()
        
        params = {
            'output': 'json',
            'latitude': latitude,
            'longitude': longitude,
            'distance': distance_km,
            'maxresults': max_results,
            'compact': 'false',
            'verbose': 'false',
            'countrycode': 'US',
            'key': self.api_key
        }
        
        try:
            url = f"{API_CONFIG['openchargemap']['base_url']}/poi/"
            response = self.session.get(
                url, params=params, 
                timeout=API_CONFIG['openchargemap']['timeout_seconds']
            )
            
            if response.status_code == 200:
                stations = response.json()
                logger.debug(f"Found {len(stations)} stations near ({latitude:.3f}, {longitude:.3f})")
                return stations
            else:
                logger.error(f"API Error {response.status_code}: {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"API request failed: {e}")
            return []
    
    def extract_station_data(self, raw_station: Dict) -> Optional[Dict]:
        """Extract and standardize station data from OpenChargeMap response"""
        try:
            address_info = raw_station.get('AddressInfo', {})
            operator_info = raw_station.get('OperatorInfo', {})
            usage_type = raw_station.get('UsageType', {})
            status_type = raw_station.get('StatusType', {})
            connections = raw_station.get('Connections', [])
            
            # Skip if no location data
            lat = address_info.get('Latitude')
            lon = address_info.get('Longitude')
            if not lat or not lon:
                return None
            
            # Skip if outside Bay Area bounds
            if not self._is_in_bay_area(lat, lon):
                return None
            
            # Process charging capabilities
            charging_info = self._process_connections(connections)
            
            # Assign realistic capacity based on station type
            capacity_ports = self._assign_station_capacity(charging_info['max_power_kw'])
            
            station_data = {
                # Identifiers
                'station_id': f"ocm_{raw_station.get('ID')}",
                'ocm_id': raw_station.get('ID'),
                'data_source': 'OpenChargeMap',
                
                # Location
                'latitude': lat,
                'longitude': lon,
                'address': address_info.get('AddressLine1', ''),
                'city': address_info.get('Town', ''),
                'state': address_info.get('StateOrProvince', ''),
                'country': address_info.get('Country', {}).get('Title', ''),
                
                # Operator and access
                'operator': operator_info.get('Title', 'Unknown'),
                'access_type': usage_type.get('Title', 'Public'),
                'is_operational': status_type.get('IsOperational', True),
                'status': status_type.get('Title', 'Operational'),
                
                # Charging capabilities
                'max_power_kw': charging_info['max_power_kw'],
                'min_power_kw': charging_info['min_power_kw'],
                'connector_types': charging_info['connector_types'],
                'charging_levels': charging_info['charging_levels'],
                'has_fast_charging': charging_info['has_fast_charging'],
                
                # Capacity modeling
                'capacity_ports': capacity_ports,
                'current_available_ports': capacity_ports,  # Initially all available
                
                # Cost estimation
                'estimated_cost_per_kwh': self._estimate_cost_from_power(charging_info['max_power_kw']),
                
                # Metadata
                'date_created': raw_station.get('DateCreated'),
                'date_last_verified': raw_station.get('DateLastVerified'),
                'created_at': datetime.now().isoformat(),
                'scenario': self.scenario
            }
            
            return station_data
            
        except Exception as e:
            logger.error(f"Error extracting station data: {e}")
            return None
    
    def _is_in_bay_area(self, lat: float, lon: float) -> bool:
        """Check if coordinates are within Bay Area bounds"""
        return (BAY_AREA_BOUNDS['south'] <= lat <= BAY_AREA_BOUNDS['north'] and
                BAY_AREA_BOUNDS['west'] <= lon <= BAY_AREA_BOUNDS['east'])
    
    def _process_connections(self, connections: List[Dict]) -> Dict:
        """Process connection data to extract charging capabilities"""
        if not connections:
            return {
                'max_power_kw': 22,
                'min_power_kw': 7,
                'connector_types': ['Type 2'],
                'charging_levels': ['Level 2'],
                'has_fast_charging': False
            }
        
        power_ratings = []
        connector_types = set()
        charging_levels = set()
        
        for conn in connections:
            # Power rating
            power_kw = conn.get('PowerKW', 0)
            if power_kw and power_kw > 0:
                power_ratings.append(power_kw)
            else:
                # Infer from level
                level_title = conn.get('Level', {}).get('Title', '')
                if 'Level 3' in level_title or 'DC' in level_title or 'Fast' in level_title:
                    power_ratings.append(50)
                elif 'Level 2' in level_title:
                    power_ratings.append(22)
                else:
                    power_ratings.append(7)
            
            # Connector types
            conn_type = conn.get('ConnectionType', {}).get('Title', 'Type 2')
            connector_types.add(conn_type)
            
            # Charging levels
            level = conn.get('Level', {}).get('Title', 'Level 2')
            charging_levels.add(level)
        
        max_power = max(power_ratings) if power_ratings else 22
        min_power = min(power_ratings) if power_ratings else 7
        
        return {
            'max_power_kw': max_power,
            'min_power_kw': min_power,
            'connector_types': list(connector_types),
            'charging_levels': list(charging_levels),
            'has_fast_charging': max_power >= 50
        }
    
    def _assign_station_capacity(self, max_power_kw: float) -> int:
        """Assign realistic port capacity based on station power and type"""
        capacity_dist = CAPACITY_MODELING['capacity_distribution']
        
        if max_power_kw >= 150:  # Ultra-fast charging hub
            return np.random.choice([8, 12], p=[0.7, 0.3])
        elif max_power_kw >= 50:  # Fast charging station
            return np.random.choice([4, 6, 8], p=[0.5, 0.3, 0.2])
        else:  # Level 2 station
            return np.random.choice([2, 4], p=[0.6, 0.4])
    
    def _estimate_cost_from_power(self, power_kw: float) -> float:
        """Estimate charging cost based on power level (Bay Area 2024)"""
        if power_kw >= 150:
            return np.random.uniform(0.45, 0.60)
        elif power_kw >= 50:
            return np.random.uniform(0.35, 0.50)
        elif power_kw >= 22:
            return np.random.uniform(0.30, 0.40)
        else:
            return np.random.uniform(0.25, 0.35)
    
    def build_real_stations_database(self, force_refresh: bool = False) -> Dict:
        """
        Build comprehensive real stations database (ONE-TIME OPERATION)
        This creates the persistent database of all real Bay Area charging stations
        """
        logger.info("🏗️ Building comprehensive Bay Area real stations database...")
        
        # Check if database already exists
        if os.path.exists(self.real_stations_csv) and not force_refresh:
            existing_df = pd.read_csv(self.real_stations_csv)
            logger.info(f"Real stations database already exists: {len(existing_df)} stations")
            return {
                'total_stations': len(existing_df),
                'source': 'existing_cache',
                'coverage_area_km2': self._calculate_coverage_area(existing_df)
            }
        
        if not self.api_key:
            logger.error("Cannot build real stations database: No OpenChargeMap API key")
            return {'error': 'No API key available'}
        
        all_stations = []
        seen_ids = set()
        
        logger.info(f"Scanning {len(BAY_AREA_REGIONS)} Bay Area regions...")
        
        for region_name, region_info in BAY_AREA_REGIONS.items():
            logger.info(f"🌉 Scanning region: {region_info['name']}")
            
            # Create grid points within region for comprehensive coverage
            grid_points = self._create_regional_grid(
                region_info['center_lat'], 
                region_info['center_lon'], 
                region_info['radius_km']
            )
            
            for point_idx, (lat, lon) in enumerate(grid_points):
                logger.info(f"   📍 Grid point {point_idx + 1}/{len(grid_points)}: ({lat:.3f}, {lon:.3f})")
                
                # Get stations from API
                raw_stations = self.find_nearby_stations_api(lat, lon, distance_km=15)
                
                # Process each station
                for raw_station in raw_stations:
                    ocm_id = raw_station.get('ID')
                    if ocm_id in seen_ids:
                        continue
                    
                    seen_ids.add(ocm_id)
                    station_data = self.extract_station_data(raw_station)
                    
                    if station_data:
                        all_stations.append(station_data)
        
        # Save to files
        if all_stations:
            stations_df = pd.DataFrame(all_stations)
            
            # Save CSV and pickle
            stations_df.to_csv(self.real_stations_csv, index=False)
            stations_df.to_pickle(self.real_stations_pkl)
            
            logger.info(f"✅ Real stations database built: {len(stations_df)} stations saved")
            
            # Generate coverage analysis
            coverage_stats = self._analyze_coverage(stations_df)
            with open(self.coverage_analysis_file, 'w') as f:
                json.dump(coverage_stats, f, indent=2)
            
            return {
                'total_stations': len(stations_df),
                'source': 'api_fresh',
                'coverage_stats': coverage_stats
            }
        else:
            logger.error("No stations found - database build failed")
            return {'error': 'No stations found'}
    
    def _create_regional_grid(self, center_lat: float, center_lon: float, radius_km: float) -> List[Tuple[float, float]]:
        """Create grid points within a circular region for comprehensive coverage"""
        grid_points = []
        
        # Calculate grid spacing (aim for ~10km between points)
        grid_spacing_km = 10
        grid_spacing_deg = grid_spacing_km / 111.0  # Rough conversion
        
        # Create square grid, then filter to circle
        steps = int(2 * radius_km / grid_spacing_km) + 1
        
        for i in range(steps):
            for j in range(steps):
                # Calculate grid point
                lat_offset = (i - steps//2) * grid_spacing_deg
                lon_offset = (j - steps//2) * grid_spacing_deg
                
                point_lat = center_lat + lat_offset
                point_lon = center_lon + lon_offset
                
                # Check if point is within radius and Bay Area bounds
                distance = geodesic((center_lat, center_lon), (point_lat, point_lon)).kilometers
                
                if (distance <= radius_km and 
                    self._is_in_bay_area(point_lat, point_lon)):
                    grid_points.append((point_lat, point_lon))
        
        return grid_points
    
    def _calculate_coverage_area(self, stations_df: pd.DataFrame) -> float:
        """Calculate approximate coverage area of stations"""
        if len(stations_df) == 0:
            return 0
        
        # Simple bounding box calculation
        lat_range = stations_df['latitude'].max() - stations_df['latitude'].min()
        lon_range = stations_df['longitude'].max() - stations_df['longitude'].min()
        
        # Convert to km² (rough approximation)
        area_km2 = lat_range * lon_range * 111 * 111 * np.cos(np.radians(stations_df['latitude'].mean()))
        return area_km2
    
    def _analyze_coverage(self, stations_df: pd.DataFrame) -> Dict:
        """Analyze station coverage and identify gaps"""
        logger.info("📊 Analyzing station coverage...")
        
        coverage_stats = {
            'total_stations': len(stations_df),
            'coverage_area_km2': self._calculate_coverage_area(stations_df),
            'station_density_per_km2': 0,
            'power_distribution': {},
            'operator_distribution': {},
            'coverage_gaps': []
        }
        
        if len(stations_df) == 0:
            return coverage_stats
        
        # Calculate density
        coverage_stats['station_density_per_km2'] = len(stations_df) / coverage_stats['coverage_area_km2']
        
        # Power distribution
        power_bins = [0, 22, 50, 150, 1000]
        power_labels = ['Level 1-2 (≤22kW)', 'Medium DC (22-50kW)', 'Fast DC (50-150kW)', 'Ultra-Fast (>150kW)']
        power_counts = pd.cut(stations_df['max_power_kw'], bins=power_bins, labels=power_labels).value_counts()
        coverage_stats['power_distribution'] = power_counts.to_dict()
        
        # Operator distribution
        operator_counts = stations_df['operator'].value_counts().head(10)
        coverage_stats['operator_distribution'] = operator_counts.to_dict()
        
        # Identify coverage gaps
        coverage_stats['coverage_gaps'] = self._identify_coverage_gaps(stations_df)
        
        return coverage_stats
    
    def _identify_coverage_gaps(self, stations_df: pd.DataFrame) -> List[Dict]:
        """Identify areas with poor charging station coverage"""
        gaps = []
        
        # Create analysis grid
        grid_size_km = MOCK_STATION_CONFIG['gap_analysis_grid_size_km']
        gap_threshold_km = MOCK_STATION_CONFIG['critical_gap_threshold_km']
        
        # Analyze coverage across Bay Area
        lat_min, lat_max = BAY_AREA_BOUNDS['south'], BAY_AREA_BOUNDS['north']
        lon_min, lon_max = BAY_AREA_BOUNDS['west'], BAY_AREA_BOUNDS['east']
        
        grid_spacing_deg = grid_size_km / 111.0
        
        lat_points = np.arange(lat_min, lat_max, grid_spacing_deg)
        lon_points = np.arange(lon_min, lon_max, grid_spacing_deg)
        
        for lat in lat_points:
            for lon in lon_points:
                if not self._is_in_bay_area(lat, lon):
                    continue
                
                # Find nearest station
                distances = stations_df.apply(
                    lambda row: geodesic((lat, lon), (row['latitude'], row['longitude'])).kilometers,
                    axis=1
                )
                
                nearest_distance = distances.min() if len(distances) > 0 else float('inf')
                
                # If gap is critical, record it
                if nearest_distance > gap_threshold_km:
                    gaps.append({
                        'latitude': lat,
                        'longitude': lon,
                        'nearest_station_km': nearest_distance,
                        'gap_severity': 'critical' if nearest_distance > gap_threshold_km * 1.5 else 'moderate'
                    })
        
        logger.info(f"Found {len(gaps)} coverage gaps")
        return gaps
    
    def build_mock_stations_for_gaps(self, force_refresh: bool = False) -> Dict:
        """
        Build mock stations for critical coverage gaps (PERSISTENT ACROSS SIMULATIONS)
        Only creates minimal stations for major gaps, not comprehensive coverage
        """
        logger.info("🎭 Building mock stations for critical coverage gaps...")
        
        # Check if mock stations already exist
        if os.path.exists(self.mock_stations_csv) and not force_refresh:
            existing_df = pd.read_csv(self.mock_stations_csv)
            logger.info(f"Mock stations already exist: {len(existing_df)} stations")
            return {
                'total_mock_stations': len(existing_df),
                'source': 'existing_cache'
            }
        
        # Load real stations for gap analysis
        real_stations_df = self.load_real_stations()
        if real_stations_df is None or len(real_stations_df) == 0:
            logger.error("Cannot build mock stations: No real stations database found")
            return {'error': 'No real stations database'}
        
        # Analyze coverage gaps
        coverage_analysis = self._analyze_coverage(real_stations_df)
        gaps = coverage_analysis['coverage_gaps']
        
        if not gaps:
            logger.info("No coverage gaps found - no mock stations needed")
            return {'total_mock_stations': 0, 'source': 'no_gaps_found'}
        
        # Generate mock stations for critical gaps only
        mock_stations = []
        max_mock_per_gap = self.scenario_config['max_mock_stations_per_gap']
        
        critical_gaps = [gap for gap in gaps if gap['gap_severity'] == 'critical']
        logger.info(f"Found {len(critical_gaps)} critical gaps requiring mock stations")
        
        for gap_idx, gap in enumerate(critical_gaps):
            # Determine how many mock stations to place for this gap
            num_stations = min(
                np.random.randint(1, max_mock_per_gap + 1),
                max_mock_per_gap
            )
            
            for station_idx in range(num_stations):
                mock_station = self._generate_mock_station_for_gap(gap, station_idx)
                if mock_station:
                    mock_stations.append(mock_station)
        
        # Save mock stations
        if mock_stations:
            mock_df = pd.DataFrame(mock_stations)
            mock_df.to_csv(self.mock_stations_csv, index=False)
            mock_df.to_pickle(self.mock_stations_pkl)
            
            logger.info(f"✅ Mock stations created: {len(mock_stations)} stations for {len(critical_gaps)} critical gaps")
            
            return {
                'total_mock_stations': len(mock_stations),
                'critical_gaps_filled': len(critical_gaps),
                'source': 'generated_fresh'
            }
        else:
            logger.warning("No mock stations generated")
            return {'total_mock_stations': 0, 'source': 'generation_failed'}
    


    def _generate_mock_station_for_gap(self, gap: Dict, station_idx: int) -> Dict:
        """Generate a single mock station to fill a coverage gap - WITH LAND VALIDATION"""
        try:
            # Try to find a land location near the gap
            station_lat, station_lon = self._find_land_location_near_gap(gap)
            
            # Select station type based on gap severity and location
            station_types = MOCK_STATION_CONFIG['mock_station_types']
            
            # Weight selection based on gap severity
            if gap['gap_severity'] == 'critical':
                # Prefer highway rest stops for critical gaps
                type_weights = [0.6, 0.25, 0.15]  # highway, shopping, urban
            else:
                # More balanced for moderate gaps
                type_weights = [0.3, 0.4, 0.3]
            
            station_type_key = np.random.choice(
                list(station_types.keys()), 
                p=type_weights
            )
            station_type = station_types[station_type_key]
            
            # Generate station characteristics
            power_kw = np.random.uniform(*station_type['power_kw_range'])
            cost_per_kwh = np.random.uniform(*station_type['cost_per_kwh_range'])
            operator = np.random.choice(MOCK_STATION_CONFIG['realistic_operators'])
            
            mock_station = {
                'station_id': f"mock_{int(gap['latitude']*1000)}_{int(abs(gap['longitude'])*1000)}_{station_idx}",
                'data_source': 'MockGenerated',
                'latitude': station_lat,
                'longitude': station_lon,
                'operator': operator,
                'max_power_kw': power_kw,
                'min_power_kw': min(power_kw, 7.4),
                'connector_types': station_type['connector_types'],
                'charging_levels': self._infer_charging_levels(power_kw),
                'has_fast_charging': power_kw >= 50,
                'capacity_ports': station_type['capacity_ports'],
                'current_available_ports': station_type['capacity_ports'],
                'estimated_cost_per_kwh': cost_per_kwh,
                'access_type': 'Public',
                'is_operational': True,
                'status': 'Operational',
                'station_type': 'mock',
                'gap_severity': gap['gap_severity'],
                'scenario': self.scenario,
                'created_at': datetime.now().isoformat()
            }
            
            return mock_station
            
        except Exception as e:
            logger.error(f"Error generating mock station: {e}")
            return None

    def _find_land_location_near_gap(self, gap: Dict) -> Tuple[float, float]:
        """Find a land location near the coverage gap"""
        
        gap_lat = gap['latitude']
        gap_lon = gap['longitude']
        
        # Try locations in expanding circles around the gap
        search_radiuses = [0.01, 0.02, 0.05, 0.1]  # Degrees (roughly 1km, 2km, 5km, 10km)
        
        for radius in search_radiuses:
            for attempt in range(20):  # 20 attempts per radius
                # Generate random offset within radius
                angle = np.random.uniform(0, 2 * np.pi)
                distance = np.random.uniform(0, radius)
                
                lat_offset = distance * np.cos(angle)
                lon_offset = distance * np.sin(angle)
                
                candidate_lat = gap_lat + lat_offset
                candidate_lon = gap_lon + lon_offset
                
                # Check if still in Bay Area bounds
                if not self._is_in_bay_area(candidate_lat, candidate_lon):
                    continue
                
                # Check if on land
                if self._is_location_on_land(candidate_lat, candidate_lon):
                    return (candidate_lat, candidate_lon)
        
        # Fallback: use gap location but move it to nearest known land area
        return self._move_to_nearest_land(gap_lat, gap_lon)

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
        
        return True

    def _move_to_nearest_land(self, lat: float, lon: float) -> Tuple[float, float]:
        """Move a water location to the nearest land area"""
        
        # Known safe land locations in Bay Area
        safe_locations = [
            (37.7749, -122.4194),  # San Francisco
            (37.3382, -122.0922),  # San Jose
            (37.8044, -122.2712),  # Oakland
            (37.4419, -122.1430),  # Palo Alto
            (37.5630, -122.3255),  # San Mateo
            (37.6688, -122.0808),  # Hayward
            (37.8715, -122.2730),  # Berkeley
        ]
        
        # Find closest safe location
        min_distance = float('inf')
        closest_location = safe_locations[0]
        
        for safe_lat, safe_lon in safe_locations:
            distance = ((lat - safe_lat) ** 2 + (lon - safe_lon) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                closest_location = (safe_lat, safe_lon)
        
        return closest_location





    def _infer_charging_levels(self, power_kw: float) -> List[str]:
        """Infer charging levels from power rating"""
        if power_kw >= 150:
            return ['Level 3 DC Fast', 'Ultra-Fast']
        elif power_kw >= 50:
            return ['Level 3 DC Fast']
        elif power_kw >= 22:
            return ['Level 2 AC']
        else:
            return ['Level 1 AC', 'Level 2 AC']
    
    def load_real_stations(self) -> Optional[pd.DataFrame]:
        """Load real stations database"""
        if self._real_stations_df is not None:
            return self._real_stations_df
        
        if os.path.exists(self.real_stations_pkl):
            self._real_stations_df = pd.read_pickle(self.real_stations_pkl)
            logger.info(f"Loaded {len(self._real_stations_df)} real stations from cache")
            return self._real_stations_df
        elif os.path.exists(self.real_stations_csv):
            self._real_stations_df = pd.read_csv(self.real_stations_csv)
            logger.info(f"Loaded {len(self._real_stations_df)} real stations from CSV")
            return self._real_stations_df
        else:
            logger.warning("No real stations database found")
            return None
    
    def load_mock_stations(self) -> Optional[pd.DataFrame]:
        """Load mock stations database"""
        if self._mock_stations_df is not None:
            return self._mock_stations_df
        
        if os.path.exists(self.mock_stations_pkl):
            self._mock_stations_df = pd.read_pickle(self.mock_stations_pkl)
            logger.info(f"Loaded {len(self._mock_stations_df)} mock stations from cache")
            return self._mock_stations_df
        elif os.path.exists(self.mock_stations_csv):
            self._mock_stations_df = pd.read_csv(self.mock_stations_csv)
            logger.info(f"Loaded {len(self._mock_stations_df)} mock stations from CSV")
            return self._mock_stations_df
        else:
            logger.info("No mock stations database found")
            return None
    
    def get_combined_infrastructure(self) -> pd.DataFrame:
        """
        Get combined infrastructure (real + mock) for current scenario
        This is what the simulation uses to find charging stations
        """
        if self._combined_stations_df is not None:
            return self._combined_stations_df
        
        stations_list = []
        
        # Add real stations if enabled
        if self.scenario_config['real_stations_enabled']:
            real_stations = self.load_real_stations()
            if real_stations is not None:
                stations_list.append(real_stations)
        
        # Add mock stations if enabled
        if self.scenario_config['mock_stations_enabled']:
            mock_stations = self.load_mock_stations()
            if mock_stations is not None:
                stations_list.append(mock_stations)
        
        if stations_list:
            self._combined_stations_df = pd.concat(stations_list, ignore_index=True)
            
            # Apply scenario-specific modifications
            self._combined_stations_df = self._apply_scenario_modifications(self._combined_stations_df)
            
            logger.info(f"Combined infrastructure: {len(self._combined_stations_df)} stations "
                       f"({self.scenario} scenario)")
        else:
            self._combined_stations_df = pd.DataFrame()
            logger.warning("No stations available for combined infrastructure")
        
        return self._combined_stations_df
    
    def _apply_scenario_modifications(self, stations_df: pd.DataFrame) -> pd.DataFrame:
        """Apply scenario-specific modifications to station availability and capacity"""
        modified_df = stations_df.copy()
        
        # Apply reliability factor (some stations randomly unavailable)
        reliability = self.scenario_config.get('station_reliability', 0.95)
        if reliability < 1.0:
            # Randomly mark some stations as unavailable
            unavailable_mask = np.random.random(len(modified_df)) > reliability
            modified_df.loc[unavailable_mask, 'is_operational'] = False
            modified_df.loc[unavailable_mask, 'current_available_ports'] = 0
            
            logger.info(f"Applied {reliability*100:.1f}% reliability: "
                       f"{unavailable_mask.sum()} stations marked unavailable")
        
        # Apply capacity modifications
        capacity_per_station = self.scenario_config.get('capacity_per_station', 4)
        if 'capacity_ports' in modified_df.columns:
            # Adjust capacity based on scenario
            capacity_factor = capacity_per_station / modified_df['capacity_ports'].mean()
            modified_df['capacity_ports'] = (modified_df['capacity_ports'] * capacity_factor).round().astype(int)
            modified_df['current_available_ports'] = modified_df['capacity_ports'].copy()
        
        return modified_df
    


    def find_nearby_stations(self, latitude: float, longitude: float, 
                        radius_km: int = 10, max_results: int = 15,
                        include_home: bool = False) -> List[Dict]:
        """
        Find nearby charging stations - SIMPLIFIED for optimization focus
        
        Args:
            latitude: Search center latitude
            longitude: Search center longitude
            radius_km: Search radius in kilometers
            max_results: Maximum number of results
            include_home: Whether to include home stations in results
            
        Returns:
            List of nearby stations with basic info
        """
        
        try:
            # Get combined infrastructure
            combined_stations = self.get_combined_infrastructure()
            
            if len(combined_stations) == 0:
                logger.warning("No stations available in combined infrastructure")
                return []
            
            # Filter by station type if needed
            if not include_home:
                combined_stations = combined_stations[combined_stations['station_type'] != 'home']
            
            # Calculate distances
            from geopy.distance import geodesic
            search_location = (latitude, longitude)
            
            distances = []
            for _, station in combined_stations.iterrows():
                station_location = (station['latitude'], station['longitude'])
                distance_km = geodesic(search_location, station_location).kilometers
                distances.append(distance_km)
            
            combined_stations = combined_stations.copy()
            combined_stations['distance_km'] = distances
            
            # Filter by radius
            nearby_stations = combined_stations[combined_stations['distance_km'] <= radius_km]
            
            # Sort by distance
            nearby_stations = nearby_stations.sort_values('distance_km')
            
            # Limit results
            nearby_stations = nearby_stations.head(max_results)
            
            # Convert to list of dictionaries with simplified fields
            result_stations = []
            for _, station in nearby_stations.iterrows():
                station_dict = {
                    'station_id': station['station_id'],
                    'latitude': station['latitude'],
                    'longitude': station['longitude'],
                    'distance_km': station['distance_km'],
                    'max_power_kw': station.get('max_power_kw', 22),
                    'operator': station.get('operator', 'Unknown'),
                    'estimated_cost_per_kwh': station.get('estimated_cost_per_kwh', 0.30),
                    'connector_types': station.get('connector_types', ['Type 1 (J1772)']),
                    'is_operational': True,  # Simplified - assume all stations are operational
                    'station_type': station.get('station_type', 'public')
                }
                result_stations.append(station_dict)
            
            logger.debug(f"Found {len(result_stations)} stations within {radius_km}km of ({latitude:.4f}, {longitude:.4f})")
            
            return result_stations
            
        except Exception as e:
            logger.error(f"Error finding nearby stations: {e}")
            return []

    def get_infrastructure_statistics(self) -> Dict:
        """Get simplified infrastructure statistics"""
        
        try:
            combined_stations = self.get_combined_infrastructure()
            
            if len(combined_stations) == 0:
                return {
                    'total_stations': 0,
                    'real_stations': 0,
                    'mock_stations': 0,
                    'home_stations': 0
                }
            
            # Count by station type
            station_counts = combined_stations['station_type'].value_counts()
            
            return {
                'total_stations': len(combined_stations),
                'real_stations': station_counts.get('real', 0),
                'mock_stations': station_counts.get('mock', 0),
                'home_stations': station_counts.get('home', 0)
            }
            
        except Exception as e:
            logger.error(f"Error getting infrastructure statistics: {e}")
            return {
                'total_stations': 0,
                'real_stations': 0,
                'mock_stations': 0,
                'home_stations': 0
            }

    def generate_home_stations_for_fleet(self, fleet_vehicles: List[Dict]) -> List[Dict]:
        """
        Generate home charging stations for fleet vehicles (PER SIMULATION)
        This is called at the start of each simulation
        """
        logger.info("🏠 Generating home charging stations for fleet...")
        
        home_stations = []
        
        for vehicle in fleet_vehicles:
            # Check if vehicle has home charging access
            if not vehicle.get('has_home_charging', False):
                continue
            
            # Get driver profile to determine installation probability
            driver_profile = vehicle.get('driver_profile', 'casual')
            installation_rates = HOME_CHARGING_CONFIG['installation_rate_by_profile']
            installation_prob = installation_rates.get(driver_profile, 0.75)
            
            # Random check for actual installation
            if np.random.random() > installation_prob:
                continue
            
            # Generate home station characteristics
            home_location = vehicle['home_location']
            
            
            home_lat, home_lon = home_location[0], home_location[1]
            
            # Validate home location is on land
            if not self._is_location_on_land(home_lat, home_lon):
                logger.warning(f"Home location for {vehicle['vehicle_id']} is in water, moving to land")
                home_lat, home_lon = self._move_to_nearest_land(home_lat, home_lon)


            # Select power level based on distribution
            power_dist = HOME_CHARGING_CONFIG['power_distribution']
            power_options = list(power_dist.keys())
            power_probs = [power_dist[opt]['probability'] for opt in power_options]
            selected_power_key = np.random.choice(power_options, p=power_probs)
            power_kw = power_dist[selected_power_key]['power_kw']
            
            home_station = {
                'station_id': f"home_{vehicle['vehicle_id']}",
                'vehicle_id': vehicle['vehicle_id'],
                'data_source': 'HomeGenerated',
                'latitude': home_lat,
                'longitude': home_lon,
                'operator': 'Home',
                'max_power_kw': power_kw,
                'min_power_kw': power_kw,
                'connector_types': ['J1772', 'Tesla'],
                'charging_levels': ['Level 2 AC'] if power_kw > 3 else ['Level 1 AC'],
                'has_fast_charging': False,
                'capacity_ports': 1,  # Home stations typically have 1 port
                'current_available_ports': 1,
                'estimated_cost_per_kwh': HOME_CHARGING_CONFIG['cost_per_kwh'],
                'access_type': 'Private',
                'is_operational': True,
                'status': 'Operational',
                'station_type': 'home',
                'availability': HOME_CHARGING_CONFIG['availability'],
                'created_at': datetime.now().isoformat()
            }
            
            home_stations.append(home_station)
        
        logger.info(f"Generated {len(home_stations)} home charging stations")
        return home_stations
    

    def update_station_availability(self, current_time: datetime, 
                               active_charging_sessions: List[Dict] = None) -> None:
        """
        Update station availability based on realistic usage patterns and active sessions
        
        Args:
            current_time: Current simulation time
            active_charging_sessions: List of currently active charging sessions
        """
        if not CAPACITY_MODELING['enable_capacity_constraints']:
            return
        
        combined_stations = self.get_combined_infrastructure()
        if len(combined_stations) == 0:
            return
        
        # Get time-based utilization factors
        hour = current_time.hour
        is_weekend = current_time.weekday() >= 5
        is_holiday = self._is_holiday(current_time.date())
        
        availability_patterns = CAPACITY_MODELING['availability_patterns']
        
        # Determine base utilization based on time patterns
        base_utilization = self._calculate_base_utilization(hour, is_weekend, is_holiday)
        
        # Track stations currently occupied by active sessions
        occupied_ports_by_station = {}
        if active_charging_sessions:
            occupied_ports_by_station = self._count_occupied_ports(active_charging_sessions, current_time)
        
        # Update each station's available ports
        for idx, station in combined_stations.iterrows():
            station_id = station['station_id']
            total_ports = station['capacity_ports']
            
            # Start with ports occupied by active sessions
            occupied_ports = occupied_ports_by_station.get(station_id, 0)
            
            # Add random utilization for non-tracked usage (other EV drivers)
            random_utilization = self._calculate_random_utilization(
                station, base_utilization, current_time
            )
            
            # Calculate total busy ports
            random_busy_ports = int((total_ports - occupied_ports) * random_utilization)
            total_busy_ports = min(total_ports, occupied_ports + random_busy_ports)
            
            # Calculate available ports
            available_ports = max(0, total_ports - total_busy_ports)
            
            # Apply station-specific factors
            available_ports = self._apply_station_specific_factors(
                station, available_ports, current_time
            )
            
            # Update the dataframe
            combined_stations.at[idx, 'current_available_ports'] = available_ports
            combined_stations.at[idx, 'last_updated'] = current_time.isoformat()
        
        # Update cached dataframe
        self._combined_stations_df = combined_stations
        
        logger.debug(f"Updated availability for {len(combined_stations)} stations at {current_time}")

    def _calculate_base_utilization(self, hour: int, is_weekend: bool, is_holiday: bool) -> float:
        """Calculate base utilization rate based on time patterns"""
        
        availability_patterns = CAPACITY_MODELING['availability_patterns']
        
        # Define realistic hourly utilization patterns
        weekday_pattern = {
            0: 0.1, 1: 0.05, 2: 0.05, 3: 0.05, 4: 0.1,   # Late night/early morning
            5: 0.2, 6: 0.4, 7: 0.6, 8: 0.7, 9: 0.5,      # Morning rush
            10: 0.4, 11: 0.5, 12: 0.6, 13: 0.5, 14: 0.4, # Midday
            15: 0.5, 16: 0.6, 17: 0.8, 18: 0.7, 19: 0.6, # Evening rush
            20: 0.5, 21: 0.4, 22: 0.3, 23: 0.2            # Evening
        }
        
        weekend_pattern = {
            0: 0.1, 1: 0.05, 2: 0.05, 3: 0.05, 4: 0.05,  # Late night
            5: 0.1, 6: 0.1, 7: 0.2, 8: 0.3, 9: 0.4,      # Morning (later start)
            10: 0.5, 11: 0.6, 12: 0.7, 13: 0.6, 14: 0.5, # Midday (higher usage)
            15: 0.6, 16: 0.7, 17: 0.6, 18: 0.5, 19: 0.5, # Afternoon
            20: 0.4, 21: 0.3, 22: 0.3, 23: 0.2            # Evening
        }
        
        # Select pattern based on day type
        if is_weekend or is_holiday:
            base_utilization = weekend_pattern.get(hour, 0.3)
            base_utilization *= availability_patterns.get('weekend_factor', 0.8)
        else:
            base_utilization = weekday_pattern.get(hour, 0.3)
        
        # Apply holiday factor if applicable
        if is_holiday:
            base_utilization *= 0.6  # Lower utilization on holidays
        
        return base_utilization

    def _calculate_random_utilization(self, station: pd.Series, base_utilization: float, 
                                    current_time: datetime) -> float:
        """Calculate random utilization with station-specific factors"""
        
        # Station type affects utilization patterns
        station_type = station.get('station_type', 'public')
        power_kw = station.get('max_power_kw', 22)
        operator = station.get('operator', 'Unknown')
        
        # Power level affects utilization (fast chargers more popular during day)
        if power_kw >= 150:  # Ultra-fast
            power_factor = 1.3 if 8 <= current_time.hour <= 20 else 0.8
        elif power_kw >= 50:  # Fast DC
            power_factor = 1.2 if 8 <= current_time.hour <= 20 else 0.9
        else:  # Level 2
            power_factor = 1.0
        
        # Operator reliability affects utilization
        reliable_operators = ['Tesla', 'Electrify America', 'ChargePoint']
        operator_factor = 1.1 if operator in reliable_operators else 1.0
        
        # Location type affects patterns (if we have this data)
        location_factor = 1.0
        if 'highway' in station.get('address', '').lower():
            location_factor = 1.2  # Highway stations busier
        elif 'mall' in station.get('address', '').lower() or 'shopping' in station.get('address', '').lower():
            location_factor = 1.1 if 10 <= current_time.hour <= 21 else 0.7
        
        # Apply all factors
        adjusted_utilization = base_utilization * power_factor * operator_factor * location_factor
        
        # Add randomness (±30% variation)
        random_factor = np.random.uniform(0.7, 1.3)
        final_utilization = adjusted_utilization * random_factor
        
        # Ensure bounds
        return np.clip(final_utilization, 0.0, 1.0)

    def _count_occupied_ports(self, active_sessions: List[Dict], current_time: datetime) -> Dict[str, int]:
        """Count ports occupied by active charging sessions"""
        
        occupied_ports = {}
        
        for session in active_sessions:
            # Check if session is currently active
            start_time = datetime.fromisoformat(session['start_time'])
            end_time = datetime.fromisoformat(session['end_time'])
            
            if start_time <= current_time <= end_time:
                station_id = session.get('station_id', '')
                if station_id:
                    occupied_ports[station_id] = occupied_ports.get(station_id, 0) + 1
        
        return occupied_ports

    def _apply_station_specific_factors(self, station: pd.Series, available_ports: int, 
                                    current_time: datetime) -> int:
        """Apply station-specific availability factors"""
        
        # Maintenance windows (some stations have regular maintenance)
        if self._is_maintenance_window(station, current_time):
            # Reduce capacity during maintenance
            available_ports = max(0, available_ports - 1)
        
        # Weather impact (if raining, slightly higher usage)
        if hasattr(self, '_current_weather') and self._current_weather.get('is_raining', False):
            # People prefer covered charging stations when raining
            if 'covered' in station.get('address', '').lower():
                available_ports = max(0, available_ports - 1)  # More popular
        
        # Network congestion (some operators have network issues)
        unreliable_operators = ['Blink', 'SemaConnect']  # Known for reliability issues
        if (station.get('operator', '') in unreliable_operators and 
            np.random.random() < 0.05):  # 5% chance of network issues
            available_ports = 0  # Station temporarily unavailable
        
        return available_ports

    def _is_maintenance_window(self, station: pd.Series, current_time: datetime) -> bool:
        """Check if station is in maintenance window"""
        
        # Some stations have regular maintenance windows
        # Fast chargers: Tuesday 2-4 AM
        # Level 2: Sunday 1-3 AM
        
        power_kw = station.get('max_power_kw', 22)
        
        if power_kw >= 50:  # Fast chargers
            return (current_time.weekday() == 1 and  # Tuesday
                    2 <= current_time.hour < 4 and
                    np.random.random() < 0.1)  # 10% chance
        else:  # Level 2
            return (current_time.weekday() == 6 and  # Sunday
                    1 <= current_time.hour < 3 and
                    np.random.random() < 0.05)  # 5% chance

    def _is_holiday(self, date) -> bool:
        """Check if date is a holiday (simplified)"""
        
        # Major US holidays that affect charging patterns
        holidays_2024 = [
            (1, 1),   # New Year's Day
            (7, 4),   # Independence Day
            (11, 28), # Thanksgiving
            (12, 25), # Christmas
        ]
        
        return (date.month, date.day) in holidays_2024

    def set_current_weather(self, weather_conditions: Dict) -> None:
        """Set current weather conditions for availability calculations"""
        self._current_weather = weather_conditions


    def simulate_station_outages(self, current_date: datetime) -> None:
        """
        Simulate random station outages and maintenance
        """
        if not CAPACITY_MODELING['enable_random_outages']:
            return
        
        combined_stations = self.get_combined_infrastructure()
        if len(combined_stations) == 0:
            return
        
        outage_config = CAPACITY_MODELING['outage_modeling']
        
        # Daily outage probability per station
        daily_outage_prob = outage_config['daily_outage_probability']
        maintenance_prob = outage_config['maintenance_outage_probability']
        
        for idx, station in combined_stations.iterrows():
            # Skip if already out of service
            if not station['is_operational']:
                continue
            
            # Check for new outages
            if np.random.random() < daily_outage_prob:
                # Regular outage
                combined_stations.at[idx, 'is_operational'] = False
                combined_stations.at[idx, 'current_available_ports'] = 0
                combined_stations.at[idx, 'outage_type'] = 'technical'
                
                # Set restoration time
                outage_hours = np.random.exponential(outage_config['average_outage_duration_hours'])
                restoration_time = current_date + timedelta(hours=outage_hours)
                combined_stations.at[idx, 'restoration_time'] = restoration_time.isoformat()
                
            elif np.random.random() < maintenance_prob:
                # Maintenance outage (longer)
                combined_stations.at[idx, 'is_operational'] = False
                combined_stations.at[idx, 'current_available_ports'] = 0
                combined_stations.at[idx, 'outage_type'] = 'maintenance'
                
                # Set restoration time
                maintenance_hours = outage_config['maintenance_duration_hours']
                restoration_time = current_date + timedelta(hours=maintenance_hours)
                combined_stations.at[idx, 'restoration_time'] = restoration_time.isoformat()
        
        # Update cached dataframe
        self._combined_stations_df = combined_stations
    
    def restore_stations_after_outages(self, current_time: datetime) -> None:
        """
        Restore stations that have completed their outage period
        """
        combined_stations = self.get_combined_infrastructure()
        if len(combined_stations) == 0:
            return
        
        # Find stations that should be restored
        for idx, station in combined_stations.iterrows():
            if (not station['is_operational'] and 
                'restoration_time' in station and 
                pd.notna(station['restoration_time'])):
                
                restoration_time = datetime.fromisoformat(station['restoration_time'])
                
                if current_time >= restoration_time:
                    # Restore station
                    combined_stations.at[idx, 'is_operational'] = True
                    combined_stations.at[idx, 'current_available_ports'] = station['capacity_ports']
                    combined_stations.at[idx, 'outage_type'] = None
                    combined_stations.at[idx, 'restoration_time'] = None
        
        # Update cached dataframe
        self._combined_stations_df = combined_stations
    

    def export_infrastructure_data(self, output_dir: str = None) -> Dict[str, str]:
        """Export all infrastructure data for analysis"""
        if output_dir is None:
            output_dir = self.data_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        exported_files = {}
        
        # Export combined infrastructure
        combined_stations = self.get_combined_infrastructure()
        if len(combined_stations) > 0:
            combined_csv = os.path.join(output_dir, f'combined_infrastructure_{self.scenario}.csv')
            combined_stations.to_csv(combined_csv, index=False)
            exported_files['combined_infrastructure'] = combined_csv
        
        # Export statistics
        stats = self.get_infrastructure_statistics()
        stats_file = os.path.join(output_dir, f'infrastructure_stats_{self.scenario}.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        exported_files['statistics'] = stats_file
        
        logger.info(f"Infrastructure data exported to {output_dir}")
        return exported_files

    def get_home_station_for_vehicle(self, vehicle_id: str) -> Optional[Dict]:
        """Get home charging station info for a specific vehicle"""
        
        try:
            home_stations = self.get_home_stations()
            
            # Find station for this vehicle
            vehicle_stations = home_stations[home_stations['vehicle_id'] == vehicle_id]
            
            if len(vehicle_stations) > 0:
                station = vehicle_stations.iloc[0]
                return {
                    'station_id': station['station_id'],
                    'latitude': station['latitude'],
                    'longitude': station['longitude'],
                    'max_power_kw': station['max_power_kw'],
                    'capacity_ports': station['capacity_ports']
                }
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error getting home station for {vehicle_id}: {e}")
            return None

    def find_nearby_stations(self, latitude: float, longitude: float, 
                            radius_km: int = 10, max_results: int = 15,
                            include_home: bool = False) -> List[Dict]:
        """
        Find nearby charging stations with current availability
        
        Args:
            latitude: Search center latitude
            longitude: Search center longitude
            radius_km: Search radius in kilometers
            max_results: Maximum number of results
            include_home: Whether to include home stations in results
            
        Returns:
            List of nearby stations with availability info
        """
        
        try:
            # Get combined infrastructure
            combined_stations = self.get_combined_infrastructure()
            
            if len(combined_stations) == 0:
                logger.warning("No stations available in combined infrastructure")
                return []
            
            # Filter by station type if needed
            if not include_home:
                combined_stations = combined_stations[combined_stations['station_type'] != 'home']
            
            # Calculate distances
            from geopy.distance import geodesic
            search_location = (latitude, longitude)
            
            distances = []
            for _, station in combined_stations.iterrows():
                station_location = (station['latitude'], station['longitude'])
                distance_km = geodesic(search_location, station_location).kilometers
                distances.append(distance_km)
            
            combined_stations = combined_stations.copy()
            combined_stations['distance_km'] = distances
            
            # Filter by radius
            nearby_stations = combined_stations[combined_stations['distance_km'] <= radius_km]
            
            # Sort by distance
            nearby_stations = nearby_stations.sort_values('distance_km')
            
            # Limit results
            nearby_stations = nearby_stations.head(max_results)
            
            # Convert to list of dictionaries
            result_stations = []
            for _, station in nearby_stations.iterrows():
                station_dict = station.to_dict()
                
                # Ensure all required fields are present
                station_dict.setdefault('current_available_ports', station_dict.get('capacity_ports', 1))
                station_dict.setdefault('is_operational', True)
                station_dict.setdefault('estimated_cost_per_kwh', 0.30)
                
                result_stations.append(station_dict)
            
            logger.debug(f"Found {len(result_stations)} stations within {radius_km}km of ({latitude:.4f}, {longitude:.4f})")
            
            return result_stations
            
        except Exception as e:
            logger.error(f"Error finding nearby stations: {e}")
            return []

    def get_infrastructure_statistics(self) -> Dict:
        """Get current infrastructure statistics for monitoring"""
        
        try:
            combined_stations = self.get_combined_infrastructure()
            
            if len(combined_stations) == 0:
                return {
                    'total_stations': 0,
                    'real_stations': 0,
                    'mock_stations': 0,
                    'home_stations': 0,
                    'available_ports': 0,
                    'total_capacity': 0,
                    'utilization_rate': 0.0
                }
            
            # Count by station type
            station_counts = combined_stations['station_type'].value_counts()
            
            # Calculate capacity metrics
            total_capacity = combined_stations['capacity_ports'].sum()
            available_ports = combined_stations['current_available_ports'].sum()
            utilization_rate = 1.0 - (available_ports / total_capacity) if total_capacity > 0 else 0.0
            
            return {
                'total_stations': len(combined_stations),
                'real_stations': station_counts.get('real', 0),
                'mock_stations': station_counts.get('mock', 0),
                'home_stations': station_counts.get('home', 0),
                'available_ports': int(available_ports),
                'total_capacity': int(total_capacity),
                'utilization_rate': utilization_rate
            }
            
        except Exception as e:
            logger.error(f"Error getting infrastructure statistics: {e}")
            return {
                'total_stations': 0,
                'real_stations': 0,
                'mock_stations': 0,
                'home_stations': 0,
                'available_ports': 0,
                'total_capacity': 0,
                'utilization_rate': 0.0
            }

    def export_infrastructure_data(self, output_dir: str) -> Dict[str, str]:
        """Export infrastructure data to files"""
        
        os.makedirs(output_dir, exist_ok=True)
        exported_files = {}
        
        try:
            # Export combined infrastructure
            combined_stations = self.get_combined_infrastructure()
            if len(combined_stations) > 0:
                combined_file = os.path.join(output_dir, 'charging_infrastructure.csv')
                combined_stations.to_csv(combined_file, index=False)
                exported_files['combined_infrastructure'] = combined_file
            
            # Export real stations if available
            real_stations = self.get_real_stations()
            if len(real_stations) > 0:
                real_file = os.path.join(output_dir, 'real_charging_stations.csv')
                real_stations.to_csv(real_file, index=False)
                exported_files['real_stations'] = real_file
            
            # Export mock stations if available
            mock_stations = self.get_mock_stations()
            if len(mock_stations) > 0:
                mock_file = os.path.join(output_dir, 'mock_charging_stations.csv')
                mock_stations.to_csv(mock_file, index=False)
                exported_files['mock_stations'] = mock_file
            
            # Export home stations if available
            home_stations = self.get_home_stations()
            if len(home_stations) > 0:
                home_file = os.path.join(output_dir, 'home_charging_stations.csv')
                home_stations.to_csv(home_file, index=False)
                exported_files['home_stations'] = home_file
            
            # Export infrastructure statistics
            stats = self.get_infrastructure_statistics()
            stats_file = os.path.join(output_dir, 'infrastructure_statistics.json')
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            exported_files['statistics'] = stats_file
            
            logger.info(f"Exported {len(exported_files)} infrastructure files to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error exporting infrastructure data: {e}")
        
        return exported_files










# Convenience functions for easy usage
def build_bay_area_infrastructure(api_key: str = None, scenario: str = "current_reality", 
                                force_refresh: bool = False) -> ChargingInfrastructureManager:
    """
    Build complete Bay Area charging infrastructure (one-time setup)
    
    Args:
        api_key: OpenChargeMap API key
        scenario: Infrastructure scenario to use
        force_refresh: Force rebuild of databases
    
    Returns:
        ChargingInfrastructureManager instance
    """
    manager = ChargingInfrastructureManager(api_key=api_key, scenario=scenario)
    
    # Build real stations database
    logger.info("Building real stations database...")
    real_stats = manager.build_real_stations_database(force_refresh=force_refresh)
    logger.info(f"Real stations: {real_stats}")
    
    # Build mock stations for gaps
    logger.info("Building mock stations for coverage gaps...")
    mock_stats = manager.build_mock_stations_for_gaps(force_refresh=force_refresh)
    logger.info(f"Mock stations: {mock_stats}")
    
    # Get final statistics
    final_stats = manager.get_infrastructure_statistics()
    logger.info(f"Final infrastructure: {final_stats.get('total_stations', 0)} total stations")
    
    return manager


def get_charging_infrastructure(scenario: str = "current_reality") -> ChargingInfrastructureManager:
    """
    Get existing charging infrastructure (for use in simulations)
    
    Args:
        scenario: Infrastructure scenario to use
    
    Returns:
        ChargingInfrastructureManager instance
    """
    return ChargingInfrastructureManager(scenario=scenario)





# Example usage and testing
if __name__ == "__main__":
    # Build infrastructure (one-time)
    api_key = os.getenv('OPENCHARGEMAP_API_KEY')
    
    if api_key:
        print("🏗️ Building Bay Area charging infrastructure...")
        manager = build_bay_area_infrastructure(
            api_key=api_key, 
            force_refresh=False
        )
        
        # Test finding stations
        print("\n🔍 Testing station search...")
        test_locations = [
            (37.7749, -122.4194),  # San Francisco
            (37.3382, -122.0922),  # San Jose
            (37.8044, -122.2712),  # Oakland
        ]
        
        for lat, lon in test_locations:
            stations = manager.find_nearby_stations(lat, lon, radius_km=10)
            print(f"Found {len(stations)} stations near ({lat:.3f}, {lon:.3f})")
        
        # Export data
        print("\n📊 Exporting infrastructure data...")
        exported = manager.export_infrastructure_data()
        print(f"Exported files: {list(exported.keys())}")
        
    else:
        print("❌ No OpenChargeMap API key found")
        print("Set OPENCHARGEMAP_API_KEY environment variable to build real infrastructure")
        
        # Test with mock-only scenario
        print("\n🎭 Testing mock-only infrastructure...")
        manager = ChargingInfrastructureManager()
        
        # Build mock stations for gaps (this will work without API key)
        print("Building mock stations...")
        mock_stats = manager.build_mock_stations_for_gaps(force_refresh=False)
        print(f"Mock stations result: {mock_stats}")
        
        # Get stats
        stats = manager.get_infrastructure_statistics()
        print(f"Infrastructure stats: {stats}")


