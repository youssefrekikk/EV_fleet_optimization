import pickle
import gzip
from pathlib import Path

import osmnx as ox
import networkx as nx
from datetime import datetime, timedelta
import random
from typing import Dict, List, Tuple, Optional

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

class NetworkDatabase:
    """Manages persistent storage and loading of road networks"""
    
    def __init__(self, db_path: str = "data/networks/bay_area_network.pkl.gz"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.network = None
        self.metadata = {}
    
    def network_exists(self) -> bool:
        """Check if network file exists"""
        return self.db_path.exists()
    
    def save_network(self, network: nx.MultiDiGraph, metadata: Dict = None):
        """Save network to compressed pickle file"""
        logger.info(f"Saving network to {self.db_path}")
        
        data = {
            'network': network,
            'metadata': metadata or {},
            'created_at': datetime.now().isoformat(),
            'nodes_count': len(network.nodes),
            'edges_count': len(network.edges)
        }
        
        with gzip.open(self.db_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        logger.info(f"Network saved: {len(network.nodes)} nodes, {len(network.edges)} edges")
    
    def load_network(self) -> nx.MultiDiGraph:
        """Load network from pickle file"""
        if not self.network_exists():
            raise FileNotFoundError(f"Network file not found: {self.db_path}")
        
        logger.info(f"Loading network from {self.db_path}")
        
        with gzip.open(self.db_path, 'rb') as f:
            data = pickle.load(f)
        
        self.network = data['network']
        self.metadata = data.get('metadata', {})
        
        logger.info(f"Network loaded: {len(self.network.nodes)} nodes, {len(self.network.edges)} edges")
        logger.info(f"Created: {data.get('created_at', 'Unknown')}")
        
        return self.network
    
    def validate_network_connectivity(self, test_locations: List[Tuple[float, float]]) -> float:
        """Test network connectivity between locations"""
        if not self.network:
            return 0.0
        
        successful_routes = 0
        total_tests = 0
        
        for i, origin in enumerate(test_locations):
            for j, dest in enumerate(test_locations):
                if i != j:
                    total_tests += 1
                    try:
                        origin_node = self._find_nearest_node(origin[0], origin[1])
                        dest_node = self._find_nearest_node(dest[0], dest[1])
                        
                        if origin_node and dest_node:
                            nx.shortest_path(self.network, origin_node, dest_node)
                            successful_routes += 1
                    except (nx.NetworkXNoPath, Exception):
                        pass
        
        return successful_routes / total_tests if total_tests > 0 else 0
    
    def _find_nearest_node(self, lat: float, lon: float) -> int:
        """Find nearest node in network"""
        min_distance = float('inf')
        nearest_node = None
        
        for node_id, node_data in self.network.nodes(data=True):
            node_lat = node_data['y']
            node_lon = node_data['x']
            distance = geodesic((lat, lon), (node_lat, node_lon)).meters
            
            if distance < min_distance:
                min_distance = distance
                nearest_node = node_id
        
        return nearest_node
