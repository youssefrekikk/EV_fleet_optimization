# EV Consumption Modeling – Split Architecture

## Two-layer design

- **Segment-level predictor** (`src/models/consumption_prediction/consumption_model_v2.py`)
  - Predicts kWh per edge/segment using only pre-segment features (no look-ahead)
  - Powers energy-aware routing in `src/models/route_optimization/optimization.py`

- **Post-trip analysis** (offline)
  - Uses realized trip signals for insight and validation without affecting online predictions

## Targets and features (segment model)

- Target: `log1p(energy_per_km)`; inference multiplies by `distance_m/1000` to get kWh
- Inputs available before traversal:
  - Edge: `distance_m`, `speed_kph` (graph attribute), derived hour/season
  - Weather: `temperature`, `wind_speed_kmh`, `is_raining`, `season`
  - Vehicle/driver: `model`, `driver_profile`, `driving_style`, `driver_personality`
- Excluded to prevent leakage: realized `duration_s`, `avg_speed_kmh`, end speeds, per-segment physics breakdowns, trip aggregates

## Encoding

- `driving_style`: ordinal (eco_friendly < normal < aggressive)
- `model`, `driver_profile`, `driver_personality`, `season`: label-encoded
- `hour` numeric; `is_weekend` may be derived but currently dropped from features

## Training pipeline

1) Load `segments.csv` + join `fleet_info.csv` and `weather.csv`
2) `engineer_features()`: distance bins, hour/weekday, weather joins, encodings
3) `prepare_features()`: select numeric features, persist `feature_columns`
4) Train Linear/RF/GBDT/(XGBoost/LightGBM/CatBoost if available); record metrics
5) Optional tuning; save bundle to `segment_energy_model.pkl`

## Online inference and routing

`SegmentEnergyRouter` assembles one-row feature frames per edge, calls predictor to obtain kWh as edge weights, and runs Dijkstra or A* (admissible heuristic uses rolling-resistance lower bound). Physics validation via `AdvancedEVEnergyModel` is available as a consistency check.

## Benefits

- Leakage-free, fast edge energy estimates for routing
- Analysis decoupled from online prediction for richer insights
- Compatible with SOC-aware planning and charging LP
