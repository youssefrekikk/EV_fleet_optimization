import requests
import pandas as pd
import time
import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import logging
import os
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()
class OpenChargeMapAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.openchargemap.io/v3"
        self.session = requests.Session()
        
        # Rate limiting to be respectful to the API
        self.last_request_time = 0
        self.min_request_interval = 0.5  # 500ms between requests
        
        # Set headers
        self.session.headers.update({
            'User-Agent': 'EV-Fleet-Simulator/1.0',
            'Accept': 'application/json'
        })
    
    def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()
    
    def find_nearby_stations(self, latitude: float, longitude: float, 
                           distance_km: int = 10, max_results: int = 50,
                           country_code: str = None) -> List[Dict]:
        """
        Find charging stations near a location
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate  
            distance_km: Search radius in kilometers
            max_results: Maximum number of results
            country_code: Optional country code filter (e.g., 'US')
        
        Returns:
            List of charging station dictionaries
        """
        self._rate_limit()
        
        params = {
            'output': 'json',
            'latitude': latitude,
            'longitude': longitude,
            'distance': distance_km,
            'maxresults': max_results,
            'compact': 'false',  # Get detailed information
            'verbose': 'false',
            'key': self.api_key
        }
        
        if country_code:
            params['countrycode'] = country_code
            
        try:
            url = f"{self.base_url}/poi/"
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                stations = response.json()
                logger.info(f"Found {len(stations)} stations near ({latitude}, {longitude})")
                return stations
            else:
                logger.error(f"API Error {response.status_code}: {response.text}")
                return []
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return []
    
    def get_station_by_id(self, station_id: int, compact: bool = False) -> Optional[Dict]:
        """
        Get detailed information for a specific charging station
        
        Args:
            station_id: Unique station identifier
            compact: Whether to return compact information
            
        Returns:
            Station dictionary or None if not found
        """
        self._rate_limit()
        
        params = {
            'output': 'json',
            'id': station_id,
            'compact': 'true' if compact else 'false',
            'key': self.api_key
        }
        
        try:
            url = f"{self.base_url}/poi/"
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                stations = response.json()
                return stations[0] if stations else None
            else:
                logger.error(f"API Error {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return None
    
    def get_operators(self) -> List[Dict]:
        """
        Get list of charging station operators
        
        Returns:
            List of operator dictionaries
        """
        self._rate_limit()
        
        try:
            url = f"{self.base_url}/operators/"
            params = {'output': 'json',
                      'key': self.api_key}
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                operators = response.json()
                logger.info(f"Retrieved {len(operators)} operators")
                return operators
            else:
                logger.error(f"API Error {response.status_code}: {response.text}")
                return []
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return []
    
    def extract_station_data(self, raw_station: Dict) -> Dict:
        """
        Extract and standardize relevant data from raw OpenChargeMap station data
        
        Args:
            raw_station: Raw station data from API
            
        Returns:
            Standardized station dictionary
        """
        try:
            # Basic information
            address_info = raw_station.get('AddressInfo', {})
            operator_info = raw_station.get('OperatorInfo', {})
            usage_type = raw_station.get('UsageType', {})
            status_type = raw_station.get('StatusType', {})
            
            # Extract connections (charging points)
            connections = raw_station.get('Connections', [])
            
            # Process connections to get charging capabilities
            charging_info = self._process_connections(connections)
            
            station_data = {
                # Identifiers
                'ocm_id': raw_station.get('ID'),
                'uuid': raw_station.get('UUID'),
                
                # Location
                'latitude': address_info.get('Latitude'),
                'longitude': address_info.get('Longitude'),
                'address': address_info.get('AddressLine1', ''),
                'city': address_info.get('Town', ''),
                'state': address_info.get('StateOrProvince', ''),
                'country': address_info.get('Country', {}).get('Title', ''),
                'postcode': address_info.get('Postcode', ''),
                
                # Operator and access
                'operator': operator_info.get('Title', 'Unknown'),
                'operator_website': operator_info.get('WebsiteURL', ''),
                'access_type': usage_type.get('Title', 'Unknown'),
                'is_public': usage_type.get('IsPublic', True),
                
                # Status
                'status': status_type.get('Title', 'Unknown'),
                'is_operational': status_type.get('IsOperational', True),
                
                # Charging capabilities
                'num_points': raw_station.get('NumberOfPoints', len(connections)),
                'max_power_kw': charging_info['max_power_kw'],
                'min_power_kw': charging_info['min_power_kw'],
                'connector_types': charging_info['connector_types'],
                'charging_levels': charging_info['charging_levels'],
                'has_fast_charging': charging_info['has_fast_charging'],
                
                # Additional info
                'cost_description': raw_station.get('UsageCost', ''),
                'general_comments': raw_station.get('GeneralComments', ''),
                'date_created': raw_station.get('DateCreated'),
                'date_last_verified': raw_station.get('DateLastVerified'),
                
                # Metadata
                'data_provider': raw_station.get('DataProvider', {}).get('Title', 'OpenChargeMap'),
                'submission_status': raw_station.get('SubmissionStatus', {}).get('Title', 'Unknown')
            }
            
            return station_data
            
        except Exception as e:
            logger.error(f"Error extracting station data: {e}")
            return {}
    
    def _process_connections(self, connections: List[Dict]) -> Dict:
        """Process connection data to extract charging capabilities"""
        if not connections:
            return {
                'max_power_kw': 0,
                'min_power_kw': 0,
                'connector_types': [],
                'charging_levels': [],
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
            
            # Connector type
            connection_type = conn.get('ConnectionType', {})
            if connection_type:
                connector_title = connection_type.get('Title', '')
                if connector_title:
                    connector_types.add(connector_title)
            
            # Charging level
            level = conn.get('Level', {})
            if level:
                level_title = level.get('Title', '')
                if level_title:
                    charging_levels.add(level_title)
        
        max_power = max(power_ratings) if power_ratings else 0
        min_power = min(power_ratings) if power_ratings else 0
        has_fast_charging = max_power >= 50  # Consider 50kW+ as fast charging
        
        return {
            'max_power_kw': max_power,
            'min_power_kw': min_power,
            'connector_types': list(connector_types),
            'charging_levels': list(charging_levels),
            'has_fast_charging': has_fast_charging
        }
    
    def get_stations_for_region(self, region_bounds: Dict, 
                               grid_size: int = 5, country_code: str = 'US') -> pd.DataFrame:
        """
        Get charging stations for a geographic region using a grid sampling approach
        
        Args:
            region_bounds: {'north': lat, 'south': lat, 'east': lon, 'west': lon}
            grid_size: Number of grid points per dimension (grid_size x grid_size)
            country_code: Country code filter
            
        Returns:
            DataFrame with standardized station data
        """
        logger.info(f"Fetching stations for region with {grid_size}x{grid_size} grid")
        
        all_stations = []
        seen_ids = set()
        
        # Create grid of sample points
        lats = np.linspace(region_bounds['south'], region_bounds['north'], grid_size)
        lons = np.linspace(region_bounds['west'], region_bounds['east'], grid_size)
        
        total_points = len(lats) * len(lons)
        current_point = 0
        
        for lat in lats:
            for lon in lons:
                current_point += 1
                logger.info(f"Fetching point {current_point}/{total_points}: ({lat:.3f}, {lon:.3f})")
                
                # Get stations near this point
                raw_stations = self.find_nearby_stations(
                    lat, lon, 
                    distance_km=20,  # Larger radius to ensure coverage
                    max_results=100,
                    country_code=country_code
                )
                
                # Process each station
                for raw_station in raw_stations:
                    station_id = raw_station.get('ID')
                    
                    # Skip if we've already processed this station
                    if station_id in seen_ids:
                        continue
                    
                    seen_ids.add(station_id)
                    
                    # Extract standardized data
                    station_data = self.extract_station_data(raw_station)
                    if station_data and station_data.get('latitude') and station_data.get('longitude'):
                        all_stations.append(station_data)
        
        logger.info(f"Collected {len(all_stations)} unique stations")
        
        # Convert to DataFrame
        if all_stations:
            df = pd.DataFrame(all_stations)
            
            # Additional cleaning
            df = df.dropna(subset=['latitude', 'longitude'])
            df = df[df['is_operational'] == True]  # Only operational stations
            
            logger.info(f"Final dataset: {len(df)} operational stations")
            return df
        else:
            logger.warning("No stations found")
            return pd.DataFrame()

# Usage example and testing functions
def test_api_integration(api_key: str):
    """Test the OpenChargeMap API integration"""
    
    api = OpenChargeMapAPI(api_key)
    
    # Test 1: Find stations near San Francisco
    print("Test 1: Finding stations near San Francisco...")
    sf_stations = api.find_nearby_stations(37.7749, -122.4194, distance_km=5, max_results=10)
    print(f"Found {len(sf_stations)} stations")
    
    if sf_stations:
        # Test 2: Extract data from first station
        print("\nTest 2: Extracting data from first station...")
        first_station = api.extract_station_data(sf_stations[0])
        print("Extracted data:")
        for key, value in first_station.items():
            print(f"  {key}: {value}")
        
        # Test 3: Get detailed info by ID
        station_id = sf_stations[0].get('ID')
        if station_id:
            print(f"\nTest 3: Getting detailed info for station {station_id}...")
            detailed_station = api.get_station_by_id(station_id)
            if detailed_station:
                print("Successfully retrieved detailed station info")
                print(detailed_station)
    
    # Test 4: Get operators
    print("\nTest 4: Getting operators...")
    operators = api.get_operators()
    print(f"Found {len(operators)} operators")
    if operators:
        print("First 5 operators:")
        for op in operators[:5]:
            print(f"  - {op.get('Title', 'Unknown')}")

def get_bay_area_stations(api_key: str) -> pd.DataFrame:
    """Get charging stations for the Bay Area"""
    
    # Bay Area bounds (approximate)
    bay_area_bounds = {
        'north': 38.0,
        'south': 37.2,
        'east': -121.5,
        'west': -122.8
    }
    
    api = OpenChargeMapAPI(api_key)
    return api.get_stations_for_region(bay_area_bounds, grid_size=4, country_code='US')


def get_api_key() -> str:
    """Get API key from environment variable"""
    api_key = os.getenv('OPENCHARGEMAP_API_KEY')
    if not api_key:
        raise ValueError(
            "OpenChargeMap API key not found. Please set the OPENCHARGEMAP_API_KEY environment variable.\n"
            "You can get a free API key from: https://openchargemap.org/site/develop/api  \n"
            "in powershell $env:OPENCHARGEMAP_API_KEY='XXXXXX'"

        )
    return api_key

if __name__ == "__main__":
    # You'll need to set your API key here
    API_KEY = get_api_key()
    
    # Run tests
    test_api_integration(API_KEY)
    
    # Get Bay Area stations
    # bay_area_df = get_bay_area_stations(API_KEY)
    # bay_area_df.to_csv('data/processed/bay_area_charging_stations.csv', index=False)
