import pickle
import gzip
from pathlib import Path
from config.ev_config import *
import osmnx as ox
import networkx as nx
from datetime import datetime, timedelta
import random
from typing import Dict, List, Tuple, Optional
import numpy as np
import os
from geopy.distance import geodesic
from config.logging_config import *
from src.utils.logger import info, warning, error, debug

# Import our configurations
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config.ev_config import *

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
        info(f"Saving network to {self.db_path}", 'road_network_db')
        
        data = {
            'network': network,
            'metadata': metadata or {},
            'created_at': datetime.now().isoformat(),
            'nodes_count': len(network.nodes),
            'edges_count': len(network.edges)
        }
        
        with gzip.open(self.db_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        info(f"Network saved: {len(network.nodes)} nodes, {len(network.edges)} edges", 'road_network_db')
    
    def load_network(self) -> nx.MultiDiGraph:
        """Load network from pickle file"""
        if not self.network_exists():
            raise FileNotFoundError(f"Network file not found: {self.db_path}")
        
        info(f"Loading network from {self.db_path}", 'road_network_db')
        
        with gzip.open(self.db_path, 'rb') as f:
            data = pickle.load(f)
        
        self.network = data['network']
        self.metadata = data.get('metadata', {})
        
        info(f"Network loaded: {len(self.network.nodes)} nodes, {len(self.network.edges)} edges", 'road_network_db')
        info(f"Created: {data.get('created_at', 'Unknown')}", 'road_network_db')
        
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
    
    def load_or_create_network(self) -> nx.MultiDiGraph:
        """Load road network from database or create new one"""
        
        # Try to load existing network from database
        if self.network_exists():
            try:
                info("📁 Loading existing network from database...", 'road_network_db')
                road_network = self.load_network()
                
                # Validate connectivity with Bay Area locations
                test_locations = [
                    (37.7749, -122.4194),  # San Francisco
                    (37.8044, -122.2712),  # Oakland
                    (37.3382, -122.0922),  # San Jose
                    (37.5485, -122.9886),  # Fremont
                    (37.4419, -122.1430),  # Palo Alto
                ]
                
                connectivity = self.validate_network_connectivity(test_locations)
                info(f"📊 Loaded network connectivity: {connectivity:.2f}", 'road_network_db')
                
                if connectivity > 0.4:  # Accept 40%+ connectivity
                    info("✅ Using existing network from database", 'road_network_db')
                    return road_network
                else:
                    warning("⚠️ Loaded network has poor connectivity, rebuilding...", 'road_network_db')
                    
            except Exception as e:
                warning(f"❌ Failed to load existing network: {e}", 'road_network_db')
        
        # Create new network if none exists or existing one is poor
        info("🔨 Creating new road network...", 'road_network_db')
        road_network = self._create_and_save_network()
        return road_network

    def _create_and_save_network(self) -> nx.MultiDiGraph:
        """Create new network and save to database with better Bay Area coverage"""
        
        network = None
        
        # Strategy 1: Larger Bay Area bounding box (most reliable)
        try:
            info("🌐 Trying comprehensive Bay Area bounding box...", 'road_network_db')
            # Expanded Bay Area bounds to include all major cities
            # [north, south, east, west] - covers SF to San Jose, Oakland to coast
            bbox_bounds = (38.1, 37.1, -121.3, -123.0)  # Much larger area
            
            network = ox.graph_from_bbox(
                bbox=bbox_bounds,
                network_type='drive',
                simplify=True,
                retain_all=True,  # Keep all components initially
                truncate_by_edge=True  # Better boundary handling
            )
            
            if network and len(network.nodes) > 2000:
                info(f"✅ Large Bay Area bbox successful: {len(network.nodes)} nodes", 'road_network_db')
                
                # Test connectivity immediately
                test_locations = [
                    (37.7749, -122.4194),  # San Francisco
                    (37.8044, -122.2712),  # Oakland  
                    (37.3382, -122.0922),  # San Jose
                    (37.5485, -122.9886),  # Fremont
                ]
                
                connectivity = self._quick_connectivity_test(network, test_locations)
                info(f"📊 Bbox network connectivity: {connectivity:.2f}", 'road_network_db')
                
                if connectivity > 0.5:  # Good connectivity
                    network = self._add_network_attributes(network)
                    return self._enhance_network_connectivity(network)
                else:
                    warning("⚠️ Bbox network has poor connectivity", 'road_network_db')
            else:
                warning("⚠️ Bbox network too small", 'road_network_db')
                network = None
                
        except Exception as e:
            warning(f"❌ Large bbox failed: {e}", 'road_network_db')
            network = None
        
        # Strategy 2: Multiple overlapping city networks (merge approach)
        if network is None:
            network = self._create_merged_city_networks()
        
        # Strategy 3: State-level network with filtering
        if network is None:
            network = self._create_filtered_state_network()
        
        # Strategy 4: Enhanced mock network as fallback
        if network is None:
            info("🎭 Creating enhanced mock network with full Bay Area coverage...", 'road_network_db')
            network = self._create_comprehensive_mock_network()
        
        # Always enhance connectivity
        if network:
            network = self._enhance_network_connectivity(network)
            
            # Save to database
            metadata = {
                'source': 'osm' if hasattr(network, 'graph') and 'crs' in network.graph else 'mock',
                'bay_area_bounds': GEOGRAPHIC_BOUNDS,
                'connectivity_enhanced': True,
                'creation_method': 'comprehensive_bay_area',
                'nodes_count': len(network.nodes),
                'edges_count': len(network.edges)
            }
            
            info("💾 Saving network to database...", 'road_network_db')
            self.network_db.save_network(network, metadata)
            
            return network
        else:
            raise Exception("Failed to create any network - all strategies failed")

    def _create_merged_city_networks(self) -> Optional[nx.MultiDiGraph]:
        """Create network by merging multiple city networks"""
        try:
            info("🏙️ Trying merged city networks approach...", 'road_network_db')
            
            # Major Bay Area cities with larger radius
            cities = [
                ("San Francisco, California, USA", (37.7749, -122.4194), 12000),
                ("Oakland, California, USA", (37.8044, -122.2712), 10000),
                ("San Jose, California, USA", (37.3382, -122.0922), 15000),
                ("Fremont, California, USA", (37.5485, -122.9886), 8000),
                ("Palo Alto, California, USA", (37.4419, -122.1430), 8000),
            ]
            
            merged_network = None
            
            for city_name, center_point, radius in cities:
                try:
                    info(f"📍 Loading network for {city_name} (radius: {radius}m)", 'road_network_db')
                    
                    # Use point-based approach for reliability
                    city_network = ox.graph_from_point(
                        center_point,
                        dist=radius,
                        network_type='drive',
                        simplify=True
                    )
                    
                    if merged_network is None:
                        merged_network = city_network
                        info(f"  🏗️ Base network: {len(city_network.nodes)} nodes", 'road_network_db')
                    else:
                        # Merge networks
                        merged_network = nx.compose_all([merged_network, city_network])
                        info(f"  ➕ Merged network: {len(merged_network.nodes)} nodes", 'road_network_db')
                    
                except Exception as e:
                    warning(f"  ❌ Failed to load {city_name}: {e}", 'road_network_db')
                    continue
            
            if merged_network and len(merged_network.nodes) > 5000:
                info(f"✅ Merged network successful: {len(merged_network.nodes)} nodes", 'road_network_db')
                merged_network = self._add_network_attributes(merged_network)
                return merged_network
            else:
                warning("⚠️ Merged network insufficient", 'road_network_db')
                return None
                
        except Exception as e:
            warning(f"❌ Merged city networks failed: {e}", 'road_network_db')
            return None

    def _create_filtered_state_network(self) -> Optional[nx.MultiDiGraph]:
        """Create network from California state data, filtered to Bay Area"""
        try:
            info("🏛️ Trying filtered state network approach...", 'road_network_db')
            
            # Get California network and filter to Bay Area
            bbox_bounds = (38.2, 36.8, -121.0, -123.2)  # Very large Bay Area
            
            network = ox.graph_from_bbox(
                bbox=bbox_bounds,
                network_type='drive',
                simplify=True,
                retain_all=True
            )
            
            if network and len(network.nodes) > 3000:
                info(f"✅ State network successful: {len(network.nodes)} nodes", 'road_network_db')
                network = self._add_network_attributes(network)
                return network
            else:
                warning("⚠️ State network insufficient", 'road_network_db')
                return None
                
        except Exception as e:
            warning(f"❌ State network failed: {e}", 'road_network_db')
            return None

    def _create_comprehensive_mock_network(self) -> nx.MultiDiGraph:
        """Create comprehensive mock network covering full Bay Area"""
        info("Creating comprehensive mock network with full Bay Area coverage...", 'road_network_db')
        
        G = nx.MultiDiGraph()
        
        # Expanded major locations covering entire Bay Area
        major_hubs = {
            # San Francisco
            'sf_downtown': (37.7749, -122.4194),
            'sf_sunset': (37.7449, -122.4794),
            'sf_mission': (37.7599, -122.4148),
            
            # East Bay
            'oakland_downtown': (37.8044, -122.2712),
            'berkeley': (37.8715, -122.2730),
            'fremont': (37.5485, -122.9886),
            'hayward': (37.6688, -122.0808),
            'richmond': (37.9358, -122.3477),
            
            # South Bay
            'san_jose_downtown': (37.3382, -122.0922),
            'palo_alto': (37.4419, -122.1430),
            'mountain_view': (37.3861, -122.0839),
            'sunnyvale': (37.3688, -122.0363),
            'santa_clara': (37.3541, -122.0322),
            'cupertino': (37.3230, -122.0322),
            
            # Peninsula
            'daly_city': (37.6879, -122.4702),
            'san_mateo': (37.5630, -122.3255),
            'redwood_city': (37.4852, -122.2364),
            'menlo_park': (37.4530, -122.1817),
            
            # North Bay (basic coverage)
            'san_rafael': (37.9735, -122.5311),
            'novato': (38.1074, -122.5697),
        }
        
        # Add hub nodes
        node_id = 0
        hub_nodes = {}
        
        for hub_name, (lat, lon) in major_hubs.items():
            G.add_node(node_id, y=lat, x=lon, hub=hub_name)
            hub_nodes[hub_name] = node_id
            node_id += 1
        
        # Define major highway connections (realistic Bay Area highways)
        major_highways = [
            # US-101 (Peninsula spine)
            ('sf_downtown', 'daly_city', 70, 12.7),
            ('daly_city', 'san_mateo', 70, 15.2),
            ('san_mateo', 'redwood_city', 70, 8.9),
            ('redwood_city', 'palo_alto', 70, 12.1),
            ('palo_alto', 'mountain_view', 65, 8.1),
            ('mountain_view', 'sunnyvale', 65, 6.4),
            ('sunnyvale', 'san_jose_downtown', 65, 12.8),
            
            # I-880 (East Bay)
            ('oakland_downtown', 'hayward', 75, 12.3),
            ('hayward', 'fremont', 75, 8.9),
            ('fremont', 'san_jose_downtown', 75, 25.4),
            
            # I-80 (Bay Bridge to East Bay)
            ('sf_downtown', 'oakland_downtown', 70, 8.5),  # Bay Bridge
            ('oakland_downtown', 'berkeley', 65, 6.8),
            ('berkeley', 'richmond', 70, 12.1),
            
            # Cross-bay connections
            ('sf_downtown', 'san_mateo', 60, 25.2),  # San Mateo Bridge area
            ('hayward', 'san_mateo', 65, 18.7),     # San Mateo Bridge
            
            # Local major roads
            ('palo_alto', 'menlo_park', 50, 5.2),
            ('mountain_view', 'cupertino', 55, 8.9),
            ('sunnyvale', 'santa_clara', 50, 4.8),
            
            # North Bay connections
            ('sf_downtown', 'san_rafael', 65, 32.2),  # Golden Gate Bridge
            ('san_rafael', 'novato', 65, 15.6),
        ]
        
        # Add highway connections
        for hub1, hub2, speed_kmh, distance_km in major_highways:
            if hub1 in hub_nodes and hub2 in hub_nodes:
                node1 = hub_nodes[hub1]
                node2 = hub_nodes[hub2]
                
                distance_m = distance_km * 1000
                travel_time = distance_m / (speed_kmh * 1000 / 3600)
                
                # Add bidirectional edges
                G.add_edge(node1, node2, 0, 
                        length=distance_m, 
                        speed_kph=speed_kmh, 
                        travel_time=travel_time,
                        highway='primary')
                G.add_edge(node2, node1, 0, 
                        length=distance_m, 
                        speed_kph=speed_kmh, 
                        travel_time=travel_time,
                        highway='primary')
        
        # Add dense local grids around each hub (larger grids)
        for hub_name, (hub_lat, hub_lon) in major_hubs.items():
            hub_node = hub_nodes[hub_name]
            
            # Create larger local grid (7x7) around each hub
            grid_size = 7
            grid_spacing = 0.008  # ~800m spacing
            
            local_nodes = []
            
            for i in range(grid_size):
                for j in range(grid_size):
                    if i == 3 and j == 3:  # Skip center (that's the hub)
                        local_nodes.append(hub_node)
                        continue
                    
                    lat = hub_lat + (i - 3) * grid_spacing
                    lon = hub_lon + (j - 3) * grid_spacing
                    
                    G.add_node(node_id, y=lat, x=lon, hub_area=hub_name)
                    local_nodes.append(node_id)
                    node_id += 1
            
            # Connect local grid with more connections
            for i in range(grid_size):
                for j in range(grid_size):
                    current_idx = i * grid_size + j
                    current_node = local_nodes[current_idx]
                    
                    # Connect to right neighbor
                    if j < grid_size - 1:
                        right_node = local_nodes[current_idx + 1]
                        self._add_local_edge(G, current_node, right_node)
                    
                    # Connect to bottom neighbor
                    if i < grid_size - 1:
                        bottom_node = local_nodes[current_idx + grid_size]
                        self._add_local_edge(G, current_node, bottom_node)
                    
                    # Add diagonal connections for better connectivity
                    if i < grid_size - 1 and j < grid_size - 1:
                        diag_node = local_nodes[current_idx + grid_size + 1]
                        self._add_local_edge(G, current_node, diag_node, speed_kmh=35)
    
            # Add inter-hub connections for areas that should be connected
            additional_connections = [
                # Connect nearby hubs that should have direct routes
                ('sf_mission', 'sf_downtown', 40, 3.2),
                ('oakland_downtown', 'sf_mission', 50, 12.1),
                ('berkeley', 'sf_downtown', 55, 18.7),
                ('san_jose_downtown', 'santa_clara', 45, 4.8),
                ('mountain_view', 'palo_alto', 50, 8.1),
                ('cupertino', 'sunnyvale', 45, 6.4),
                ('hayward', 'san_mateo', 60, 18.7),  # San Mateo Bridge
                ('fremont', 'palo_alto', 55, 22.3),
            ]
            
            for hub1, hub2, speed_kmh, distance_km in additional_connections:
                if hub1 in hub_nodes and hub2 in hub_nodes:
                    node1 = hub_nodes[hub1]
                    node2 = hub_nodes[hub2]
                    
                    distance_m = distance_km * 1000
                    travel_time = distance_m / (speed_kmh * 1000 / 3600)
                    
                    G.add_edge(node1, node2, 0, 
                            length=distance_m, 
                            speed_kph=speed_kmh, 
                            travel_time=travel_time,
                            highway='secondary')
                    G.add_edge(node2, node1, 0, 
                            length=distance_m, 
                            speed_kph=speed_kmh, 
                            travel_time=travel_time,
                            highway='secondary')
            
            info(f"Created comprehensive mock network with {len(G.nodes)} nodes and {len(G.edges)} edges", 'road_network_db')
            return G

    def _add_local_edge(self, G, node1, node2, speed_kmh=40):
        """Add local street edge between two nodes with configurable speed"""
        node1_data = G.nodes[node1]
        node2_data = G.nodes[node2]
        
        coord1 = (node1_data['y'], node1_data['x'])
        coord2 = (node2_data['y'], node2_data['x'])
        
        distance = geodesic(coord1, coord2).meters
        travel_time = distance / (speed_kmh * 1000 / 3600)
        
        # Add bidirectional edges
        G.add_edge(node1, node2, 0, 
                length=distance, 
                speed_kph=speed_kmh, 
                travel_time=travel_time,
                highway='residential')
        G.add_edge(node2, node1, 0, 
                length=distance, 
                speed_kph=speed_kmh, 
                travel_time=travel_time,
                highway='residential')

    def _enhance_network_connectivity(self, network: nx.MultiDiGraph) -> nx.MultiDiGraph:
        """Enhanced connectivity improvement with Bay Area specific bridges"""
        
        # Find disconnected components
        undirected = network.to_undirected()
        components = list(nx.connected_components(undirected))
        
        if len(components) > 1:
            info(f"🔗 Found {len(components)} disconnected components, adding bridges...", 'road_network_db')
            
            # Connect largest component to others
            largest_component = max(components, key=len)
            bridges_added = 0
            
            for component in components:
                if component != largest_component and len(component) > 5:  # Only connect significant components
                    # Find closest nodes between components
                    min_distance = float('inf')
                    best_connection = None
                    
                    # Sample nodes for performance
                    sample_size = min(20, len(component), len(largest_component))
                    component_sample = list(component)[:sample_size]
                    largest_sample = list(largest_component)[:sample_size]
                    
                    for node1 in largest_sample:
                        for node2 in component_sample:
                            try:
                                coord1 = (network.nodes[node1]['y'], network.nodes[node1]['x'])
                                coord2 = (network.nodes[node2]['y'], network.nodes[node2]['x'])
                                distance = geodesic(coord1, coord2).meters
                                
                                if distance < min_distance:
                                    min_distance = distance
                                    best_connection = (node1, node2)
                            except:
                                continue
                    
                    # Add synthetic bridge if reasonable distance
                    if best_connection and min_distance < 30000:  # Max 30km bridge
                        node1, node2 = best_connection
                        
                        # Determine bridge type and speed based on distance
                        if min_distance < 5000:  # Local connection
                            speed_kmh = 50
                            highway_type = 'synthetic_local'
                        elif min_distance < 15000:  # Regional connection
                            speed_kmh = 65
                            highway_type = 'synthetic_regional'
                        else:  # Long-distance bridge
                            speed_kmh = 80
                            highway_type = 'synthetic_bridge'
                        
                        travel_time = min_distance / (speed_kmh * 1000 / 3600)
                        
                        # Add bidirectional edges
                        network.add_edge(node1, node2, 0,
                                    length=min_distance,
                                    speed_kph=speed_kmh,
                                    travel_time=travel_time,
                                    highway=highway_type)
                        network.add_edge(node2, node1, 0,
                                    length=min_distance,
                                    speed_kph=speed_kmh,
                                    travel_time=travel_time,
                                    highway=highway_type)
                        
                        bridges_added += 1
                        info(f"🌉 Added {highway_type}: {min_distance/1000:.1f}km", 'road_network_db')
            
            info(f"✅ Added {bridges_added} synthetic connections", 'road_network_db')
        else:
            info("✅ Network is already fully connected", 'road_network_db')
        
        # Add strategic Bay Area connections if this is a mock network
        if not hasattr(network, 'graph') or 'crs' not in network.graph:
            network = self._add_strategic_bay_area_connections(network)
        
        return network

    def _add_strategic_bay_area_connections(self, network: nx.MultiDiGraph) -> nx.MultiDiGraph:
        """Add strategic connections for key Bay Area routes"""
        
        info("🌉 Adding strategic Bay Area connections...", 'road_network_db')
        
        # Key Bay Area connection points that should always be connected
        strategic_connections = [
            # Major bridge/tunnel equivalents
            ((37.7749, -122.4194), (37.8044, -122.2712), 70, "Bay Bridge equivalent"),
            ((37.7749, -122.4194), (37.9735, -122.5311), 65, "Golden Gate Bridge equivalent"),
            ((37.6688, -122.0808), (37.5630, -122.3255), 65, "San Mateo Bridge equivalent"),
            
            # Major highway connections
            ((37.3382, -122.0922), (37.5485, -122.9886), 75, "US-880 connection"),
            ((37.7749, -122.4194), (37.3382, -122.0922), 70, "US-101 connection"),
            ((37.8044, -122.2712), (37.3382, -122.0922), 75, "I-880 full connection"),
        ]
        
        connections_added = 0
        
        for (lat1, lon1), (lat2, lon2), speed_kmh, description in strategic_connections:
            try:
                # Find nearest nodes to these strategic points
                node1 = self._find_nearest_node(lat1, lon1)
                node2 = self._find_nearest_node(lat2, lon2)
                
                if node1 and node2 and node1 != node2:
                    # Check if already connected
                    try:
                        nx.shortest_path(network, node1, node2)
                        debug(f"✅ {description} already connected", 'road_network_db')
                        continue
                    except nx.NetworkXNoPath:
                        # Add connection
                        coord1 = (network.nodes[node1]['y'], network.nodes[node1]['x'])
                        coord2 = (network.nodes[node2]['y'], network.nodes[node2]['x'])
                        distance = geodesic(coord1, coord2).meters
                        travel_time = distance / (speed_kmh * 1000 / 3600)
                        
                        network.add_edge(node1, node2, 0,
                                    length=distance,
                                    speed_kph=speed_kmh,
                                    travel_time=travel_time,
                                    highway='strategic_connection',
                                    description=description)
                        network.add_edge(node2, node1, 0,
                                    length=distance,
                                    speed_kph=speed_kmh,
                                    travel_time=travel_time,
                                    highway='strategic_connection',
                                    description=description)
                        
                        connections_added += 1
                        info(f"🔗 Added {description}: {distance/1000:.1f}km", 'road_network_db')
            
            except Exception as e:
                debug(f"Could not add {description}: {e}", 'road_network_db')
                continue
        
        info(f"✅ Added {connections_added} strategic Bay Area connections", 'road_network_db')
        return network



    def _quick_connectivity_test(self, network: nx.MultiDiGraph, test_locations: List[Tuple[float, float]]) -> float:
        """Enhanced connectivity test with better error handling"""
        successful_routes = 0
        total_tests = 0
        
        # Test connectivity between all pairs
        for i, origin in enumerate(test_locations):
            for j, dest in enumerate(test_locations):
                if i != j:
                    total_tests += 1
                    try:
                        origin_node = self._find_nearest_node(origin[0], origin[1])
                        dest_node = self._find_nearest_node(dest[0], dest[1])
                        
                        if origin_node and dest_node and origin_node != dest_node:
                            try:
                                path = nx.shortest_path(network, origin_node, dest_node)
                                if len(path) > 1:  # Valid path
                                    successful_routes += 1
                                    debug(f"✅ Route {i}→{j}: {len(path)} nodes", 'road_network_db')
                                else:
                                    debug(f"⚠️ Route {i}→{j}: trivial path", 'road_network_db')
                            except nx.NetworkXNoPath:
                                debug(f"❌ Route {i}→{j}: no path", 'road_network_db')
                        else:
                            debug(f"❌ Route {i}→{j}: invalid nodes", 'road_network_db')
                    except Exception as e:
                        debug(f"❌ Route {i}→{j}: error {e}", 'road_network_db')
        
        connectivity_score = successful_routes / total_tests if total_tests > 0 else 0
        info(f"📊 Connectivity: {successful_routes}/{total_tests} routes ({connectivity_score:.2%})", 'road_network_db')
        return connectivity_score

    def validate_network_coverage(self) -> Dict:
        """Validate that network covers all major Bay Area locations"""
        if not self.network:
            return {"status": "No network loaded"}
        
        # Test locations across the Bay Area
        test_locations = {
            'San Francisco': (37.7749, -122.4194),
            'Oakland': (37.8044, -122.2712),
            'San Jose': (37.3382, -122.0922),
            'Fremont': (37.5485, -122.9886),
            'Palo Alto': (37.4419, -122.1430),
            'Berkeley': (37.8715, -122.2730),
            'Hayward': (37.6688, -122.0808),
            'Mountain View': (37.3861, -122.0839),
            'Daly City': (37.6879, -122.4702),
            'Sunnyvale': (37.3688, -122.0363)
        }
        
        coverage_results = {}
        total_connectivity = 0
        total_tests = 0
        
        for name, location in test_locations.items():
            try:
                nearest_node = self._find_nearest_node(location[0], location[1])
                if nearest_node:
                    node_data = self.network.nodes[nearest_node]
                    distance = geodesic(location, (node_data['y'], node_data['x'])).meters
                    coverage_results[name] = {
                        'covered': True,
                        'distance_to_network_m': distance,
                        'nearest_node': nearest_node
                    }
                else:
                    coverage_results[name] = {
                        'covered': False,
                        'distance_to_network_m': float('inf'),
                        'nearest_node': None
                    }
            except Exception as e:
                coverage_results[name] = {
                    'covered': False,
                    'error': str(e)
                }
        
        # Test inter-city connectivity
        covered_locations = [(name, loc) for name, loc in test_locations.items() 
                            if coverage_results[name].get('covered', False)]
        
        for i, (name1, loc1) in enumerate(covered_locations):
            for j, (name2, loc2) in enumerate(covered_locations):
                if i < j:  # Avoid duplicate tests
                    total_tests += 1
                    try:
                        node1 = self._find_nearest_node(loc1[0], loc1[1])
                        node2 = self._find_nearest_node(loc2[0], loc2[1])
                        
                        if node1 and node2:
                            nx.shortest_path(self.network, node1, node2)
                            total_connectivity += 1
                    except:
                        pass
        
        connectivity_score = total_connectivity / total_tests if total_tests > 0 else 0
        
        return {
            'network_nodes': len(self.network.nodes),
            'network_edges': len(self.network.edges),
            'location_coverage': coverage_results,
            'inter_city_connectivity': connectivity_score,
            'total_connectivity_tests': total_tests,
            'successful_connections': total_connectivity
        }








    def _quick_connectivity_test(self, network: nx.MultiDiGraph, test_locations: List[Tuple[float, float]]) -> float:
        """Quick connectivity test for network validation"""
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
                            nx.shortest_path(network, origin_node, dest_node)
                            successful_routes += 1
                    except:
                        pass
        
        return successful_routes / total_tests if total_tests > 0 else 0

    def _enhance_network_connectivity(self, network: nx.MultiDiGraph) -> nx.MultiDiGraph:
        """Enhance network connectivity by adding synthetic connections where needed"""
        
        # Find disconnected components
        undirected = network.to_undirected()
        components = list(nx.connected_components(undirected))
        
        if len(components) > 1:
            info(f"🔗 Found {len(components)} disconnected components, adding bridges...", 'road_network_db')
            
            # Connect largest component to others
            largest_component = max(components, key=len)
            bridges_added = 0
            
            for component in components:
                if component != largest_component:
                    # Find closest nodes between components
                    min_distance = float('inf')
                    best_connection = None
                    
                    # Sample nodes for performance (don't check all combinations)
                    sample_size = min(50, len(component), len(largest_component))
                    component_sample = list(component)[:sample_size]
                    largest_sample = list(largest_component)[:sample_size]
                    
                    for node1 in largest_sample:
                        for node2 in component_sample:
                            try:
                                coord1 = (network.nodes[node1]['y'], network.nodes[node1]['x'])
                                coord2 = (network.nodes[node2]['y'], network.nodes[node2]['x'])
                                distance = geodesic(coord1, coord2).meters
                                
                                if distance < min_distance:
                                    min_distance = distance
                                    best_connection = (node1, node2)
                            except:
                                continue
                    
                    # Add synthetic bridge
                    if best_connection and min_distance < 50000:  # Max 50km bridge
                        node1, node2 = best_connection
                        
                        # Add bidirectional edges
                        network.add_edge(node1, node2, 0,
                                    length=min_distance,
                                    speed_kph=60,
                                    travel_time=min_distance / (60 * 1000 / 3600),
                                    highway='synthetic_bridge')
                        network.add_edge(node2, node1, 0,
                                    length=min_distance,
                                    speed_kph=60,
                                    travel_time=min_distance / (60 * 1000 / 3600),
                                    highway='synthetic_bridge')
                        
                        bridges_added += 1
                        info(f"🌉 Added synthetic bridge: {min_distance/1000:.1f}km", 'road_network_db')
            
            info(f"✅ Added {bridges_added} synthetic bridges", 'road_network_db')
        else:
            info("✅ Network is already fully connected", 'road_network_db')
        
        return network




    def get_network_info(self) -> Dict:
        """Get information about the current network"""
        if not self.network:
            if self.network_exists():
                self.load_network()
            else:
                return {"status": "No network available"}
        
        # Test connectivity
        test_locations = [
            (37.7749, -122.4194),  # San Francisco
            (37.8044, -122.2712),  # Oakland
            (37.3382, -122.0922),  # San Jose
            (37.5485, -122.9886),  # Fremont
            (37.4419, -122.1430),  # Palo Alto
        ]
        
        connectivity = self.validate_network_connectivity(test_locations)
        
        return {
            "status": "Available",
            "nodes": len(self.network.nodes),
            "edges": len(self.network.edges),
            "connectivity_score": connectivity,
            "metadata": self.metadata,
            "file_path": str(self.db_path),
            "file_exists": self.network_exists()
        }

    









    def _add_network_attributes(self, network):  # Add 'network' parameter
        """Add speed and travel time attributes to network edges"""
        try:
            info("Adding edge speeds and travel times...", 'road_network_db')
            network = ox.add_edge_speeds(network)  # Use the parameter
            network = ox.add_edge_travel_times(network)  # Use the parameter
            info("Successfully added network attributes", 'road_network_db')
            return network  # Return the modified network
        except Exception as e:
            warning(f"Failed to add network attributes: {e}", 'road_network_db')
            # Add basic attributes manually
            return self._add_basic_network_attributes(network)  # Pass network parameter


    def _add_basic_network_attributes(self, network):  # Add network parameter
        """Add basic speed and travel time attributes manually"""
        info("Adding basic network attributes manually...", 'road_network_db')
        
        for u, v, key, data in network.edges(keys=True, data=True):  # Use parameter
            # Get edge length
            length = data.get('length', 100)  # Default 100m if missing
            
            # Estimate speed based on road type
            highway_type = data.get('highway', 'residential')
            
            if isinstance(highway_type, list):
                highway_type = highway_type[0]
            
            # Speed mapping (km/h)
            speed_map = {
                'motorway': 100,
                'trunk': 80,
                'primary': 60,
                'secondary': 50,
                'tertiary': 40,
                'residential': 30,
                'service': 20,
                'unclassified': 40
            }
            
            speed_kmh = speed_map.get(highway_type, 40)  # Default 40 km/h
            
            # Add attributes
            network.edges[u, v, key]['speed_kph'] = speed_kmh
            network.edges[u, v, key]['travel_time'] = length / (speed_kmh * 1000 / 3600)
        
        return network  # Return the modified network





    def _find_nearest_node(self, lat: float, lon: float) -> int:
        """Find nearest node in the network (with fallback for mock network)"""
        try:
            # Try OSMnx method first
            return ox.nearest_nodes(self.network, lon, lat)
        except:
            # Fallback: find closest node manually
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




