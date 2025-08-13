from __future__ import annotations

from typing import Dict, Any

from src.data_generation.road_network_db import NetworkDatabase


def build_or_load_network(db_path: str = "data/networks/bay_area_network.pkl.gz") -> Dict[str, Any]:
    nd = NetworkDatabase(db_path=db_path)
    G = nd.load_or_create_network()
    info = nd.get_network_info()
    return {"graph_loaded": True, "num_nodes": info.get("num_nodes"), "num_edges": info.get("num_edges"), "db_path": db_path}

