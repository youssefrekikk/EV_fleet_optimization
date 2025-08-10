"""
Centralized configuration for route optimization.

Only optimization-related knobs live here to keep concerns separate from
vehicle specs, physics constants, and fleet generation configs.
"""

OPTIMIZATION_CONFIG = {
    # Routing algorithm: 'dijkstra' (default) or 'astar'
    "route_optimization_algorithm": "astar",

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
    # Sample fraction of trips per day to speed up runs (None means all trips)
    "fleet_eval_trip_sample_frac": 0.4,
}

__all__ = ["OPTIMIZATION_CONFIG"]


