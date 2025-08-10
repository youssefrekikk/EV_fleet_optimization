# NetworkDatabase – Bay Area Graph Loader/Builder

File: `src/data_generation/road_network_db.py`

Manages loading, building, enhancing, validating, and caching a road network for the Bay Area as a `networkx.MultiDiGraph` with attributes needed for routing.

## Capabilities

- Load cached graph: `data/networks/bay_area_network.pkl.gz`
- Build graph via strategies:
  1) Large bbox over Bay Area using OSMnx
  2) Merge multiple city-centered graphs (SF, Oakland, San Jose, Fremont, Palo Alto)
  3) State bbox filtered to Bay Area
  4) Comprehensive mock network with major hubs, highways, grids
- Enhance connectivity: add synthetic bridges between components and strategic links (Bay Bridge, Golden Gate, San Mateo, Dumbarton, etc.)
- Add attributes: speed and travel time per edge (via OSMnx or manual mapping)
- KDTree index for fast nearest-node queries
- Connectivity and coverage validation helpers

Design choices:
- Prefer bbox build first (broad, simple), then merged cities (resilient), then state-level filter, then mock (last resort)
- Always run connectivity enhancement to avoid stranded components common in OSM
- Manual speed maps when OSM attributes are incomplete; conservative defaults

## Key methods

- `network_exists()` → bool
- `save_network(graph, metadata)` / `load_network()`
- `load_or_create_network()` → `nx.MultiDiGraph`
- `_create_and_save_network()` → build via strategies and persist
- `_add_network_attributes(network)` → speed/time attributes
- `_enhance_network_connectivity(network)` → synthetic links between components
- `_add_strategic_bay_area_connections(network)` → force-connect critical corridors
- `_build_kdtree()` → KDTree for nearest node queries
- `_find_nearest_node(lat, lon)` → node id (KDTree → OSMnx → manual fallback)
- `validate_network_connectivity(test_locations)` → ratio
- `validate_network_coverage()` → coverage and inter-city connectivity report
- `get_network_info()` → metadata, node/edge counts, connectivity

## Edge attributes

- `length` (m), `speed_kph`, `travel_time` (s), `highway` category

Manual speed map (km/h) examples:
`motorway` 105; `trunk` 90; `primary` 70; `secondary` 55; `tertiary` 45; `residential` 32; synthetic links (50–80) depending on type.

## Storage format

Gzipped pickle with keys: `network`, `metadata`, `created_at`, `nodes_count`, `edges_count`.

## Example

```python
from src.data_generation.road_network_db import NetworkDatabase

db = NetworkDatabase()
G = db.load_or_create_network()
info = db.get_network_info()
print(info)
```


