# SyntheticEVGenerator – Data Orchestration

File: `src/data_generation/synthetic_ev_generator.py`

Coordinates fleet creation, trip generation, routing, GPS trace synthesis, physics-based energy computation, charging simulation, and exports.

## High-level flow

1) Initialize configuration (fleet, charging, weather, bounds)
2) Generate fleet with vehicle specs, driver profiles/styles/personalities, home charging, home locations
3) For each day and vehicle:
   - Choose start hour and number of trips based on profile
   - Sample destinations with home weighting and corridor heuristics; occasional long trips
   - Route using `NetworkDatabase` or fallback
   - Synthesize speeds, elevations, timestamps across the path
   - Compute segment/trip energy via `AdvancedEVEnergyModel`
   - Simulate charging: home vs public, thresholds, power limits, prices
4) Export CSVs (and Parquet if enabled) with stable schemas

## Important internal helpers

- `_generate_home_location()` and land checks
- `_generate_trip_destination_with_home_weighting()`; `_check_for_long_trip()`
- `_generate_route_with_timing()` and fallback route builders
- `_generate_gps_trace_with_timing()` and elevation synthesis
- `_calculate_energy_consumption()` delegating to `AdvancedEVEnergyModel`
- `generate_charging_sessions()` with station selection and pricing
- `save_datasets()` with cleaning and schema normalization

## Technical choices

- Destination sampling blends: random within distance band, home bias by time-of-day/profile, and corridor/highway hints to mimic common flows
- Speed synthesis uses edge category → nominal speed with noise; elevations via simple regional model if missing
- Long-trip trigger introduces realistic rare long-distance behavior; parameterized by profile and time budget
- Charging thresholds incorporate driving style and home access; public selection considers power, detour, price, and personality
- Data cleaning ensures minimal NaNs, numeric coercions, and consistent dtypes across exports

## Outputs

See `docs/synthetic_data.md` for dataset contracts. When `return_segments=True` in energy calculation, `segments.csv` is produced with per-segment breakdowns.

## Logging

- Module logs under `synthetic_ev_generator`
- Detailed logs available for: `route_generation`, `gps_trace`, `charging_sessions`, `infrastructure`


