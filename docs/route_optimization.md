# Route Optimization – Energy-aware paths and SOC planning

Files:
- Edge energy + routing: `src/models/route_optimization/optimization.py`
- Fleet/day orchestration: `src/models/route_optimization/optimize_fleet.py`

## SegmentEnergyRouter (edge-weighted routing)

- Weight function: `EnergyWeightFunction` wraps `SegmentEnergyPredictor` to compute kWh per edge given vehicle/driver/weather context and departure time. Caches per-context edge weights.
- Algorithms:
  - Dijkstra: `shortest_path_by_energy`
  - A*: `a_star_path_by_energy` with admissible heuristic (rolling-resistance lower bound × straight-line distance)
- Physics validation: `validate_path_with_physics()` reconstructs a coarse GPS trace along nodes and computes kWh with `AdvancedEVEnergyModel` as a consistency check.

Inputs:
- Graph `nx.MultiDiGraph` with edge attrs: `length` (m), `speed_kph`, `travel_time` (s)
- Context: `vehicle_context`, `driver_context`, `weather_context`, `departure_time`

Design choices:
- Use ML edge kWh instead of physics at routing-time for speed; validate a posteriori with physics when needed
- A* heuristic: lower-bound kWh/km derived from rolling resistance with average mass and gravity – admissible and fast
- Cache edge weights per routing context (model, style, season, temperature bin, hour) to avoid repeated ML calls

## SOCResourceRouter (battery-aware routing)

State expansion over SOC buckets with actions:
- Drive edges consuming kWh from ML edge weights
- Charge at nodes with available stations (max power, price), adding SOC with time/cost penalties

Parameters:
- `nominal_battery_capacity_kwh`, `initial_soc`, `min_soc`, `soc_step`
- Trade-off `gamma_time_weight` and optional `price_weight_kwh_per_usd`
- `charging_stations`: optional list with `(node_id or lat/lon)`, `max_power_kw`, `cost_per_kwh`

Output:
- `{ feasible, path, actions, metrics: { energy_kwh, travel_time_s, num_charges } }`

Notes:
- Effective capacity depends on ambient temperature via `(T/Tref)^alpha`
- Charging actions consider station power, overhead (min service time), and optional price weighting
- State space limited by `soc_step`; feasibility requires terminal SOC ≥ `min_soc`

## Charging LP (single-path schedule)

`optimize_charging_schedule_lp(...)`: Linear program that, given edge energies/times and station options along a fixed path, chooses how much to charge at each stop to minimize time (or cost). Returns per-stop charge kWh and aggregate times.

## Fleet optimization per day

`optimize_fleet_day(...)` in `optimize_fleet.py`:
- Loads/creates graph via `NetworkDatabase`
- Builds per-vehicle context from `fleet_info` and `weather`
- For each trip in `routes.csv` on a date: energy-optimal routing (with shared ML cache), metrics accumulation, and physics validation
- Fallback to direct-route physics evaluation if routing fails
- Optional SOC planning run and comparison

Aggregates results to `data/analysis/optimized/fleet_optimization_results.csv` with summary JSON.

Fallback behavior:
- If routing fails (e.g., disconnected components), compute a direct great-circle path with detour factor and evaluate with physics model; record `fallback_used`

Performance considerations:
- Shared ML weight cache across all trips on the same day
- Nearest-node queries cached per coordinate (rounded) and accelerated by `NetworkDatabase` KDTree


