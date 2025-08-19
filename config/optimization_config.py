"""
Centralized configuration for route optimization.

Only optimization-related knobs live here to keep concerns separate from
vehicle specs, physics constants, and fleet generation configs.
"""

OPTIMIZATION_CONFIG = {
    # Routing algorithm: 'dijkstra', 'astar' (NetworkX), or 'custom_astar' (our optimized version)
    # - 'dijkstra': Simple, reliable, explores all nodes
    # - 'astar': NetworkX A* with ML weight function (original implementation)
    # - 'custom_astar': Our optimized A* with batched predictions (recommended for performance)
    "route_optimization_algorithm": "custom_astar",

    # Time penalty converted to kWh-equivalent (kWh per hour)
    # Example: 3.0 means 1 hour of time is penalized like 3 kWh of energy
    "gamma_time_weight": 0.02,

    # Price sensitivity for charging in SOC-aware routing (kWh penalty per USD)
    # Example: 2.0 means $1 of price is treated as 2 kWh in the objective
    "price_weight_kwh_per_usd": 0.0,

    # Resource planning
    "prediction_horizon_hours": 24,
    "reoptimization_frequency_hours": 4,

    # Safety buffers and constraints
    "battery_buffer_percentage": 0.15,  # Keep 15% buffer above min SOC when planning
    "max_detour_for_charging_km": 5,

    # Fleet evaluation defaults (used by optimize_fleet.py when CLI not provided)
    # Limit how many days to process (None means all)
    "fleet_eval_max_days": 3,
    # Sample fraction of trips per day to speed up runs (one means all trips)
    "fleet_eval_trip_sample_frac": 0.2,

    # SOC routing objective and weights (defaults)
    # objective: 'energy' | 'cost' | 'time' | 'weighted'
    "soc_objective": "cost",
    # Value of time in USD per hour (used for 'cost' and 'weighted')
    "alpha_usd_per_hour": 20.0,
    # Convert kWh to USD for weighted objective (typical average price)
    "beta_kwh_to_usd": 0.0,

    # Planning horizon / future-knowledge knobs
    # 'myopic' (no look-ahead), 'next_trip' (reserve for next trip), 'rolling_horizon' (reserve for K trips / T hours)
    "planning_mode": "rolling_horizon",
    # Reserve as SOC fraction if no future info or as base floor
    "reserve_soc": 0.15,
    # Reserve as kWh (overrides reserve_soc if > 0 when future need known)
    "reserve_kwh": 0.0,
    # Look-ahead depth
    "horizon_trips": 2,
    # Or time horizon in hours (optional)
    "horizon_hours": 0.0,
    
    # Performance optimization settings
    "lru_cache_size": 10000,  # Size of LRU cache for edge weights
    "batch_size": 50,  # Number of edges to predict in each batch
    "enable_batched_predictions": True,  # Enable batched ML predictions
}

__all__ = ["OPTIMIZATION_CONFIG"]


