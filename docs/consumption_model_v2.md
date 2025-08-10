# SegmentEnergyPredictor – Segment-level ML

File: `src/models/consumption_prediction/consumption_model_v2.py`

Predicts per-edge/segment energy (kWh) using features available before the segment is traversed. Used online as the edge weight in energy-aware routing.

## Pipeline

- `load_data()`: load `segments.csv`, join `fleet_info.csv`, `weather.csv`
- `engineer_features(df)`: distance bins, hour/weekday, label encodings, ordinal driving style, weather numerics, derived features
- `prepare_features(df)`: form target `log1p(energy_per_km)` and numeric feature frame; persist `feature_columns`
- `train_models(...)`: train Linear, RF, GradientBoosting; optionally XGBoost/LightGBM/CatBoost if installed; record metrics and importance
- `tune_*`: optional hyperparameter search
- `save_model(path)` / `load_model(path)`: persist/restore bundle including encoders and `feature_columns`
- `predict(features, model_name, distance_m)`: returns kWh for each segment row

## Features (training-time)

- Spatial: `distance_m`, `log_distance_m`, optional `elevation_gain`
- Temporal: `start_time` → `hour`, `weekday`, `is_weekend` (some dropped in feature selection)
- Speeds: `start_speed_kmh`, `end_speed_kmh`, `speed_delta` (often same for edge-level inference)
- Weather: `weather_temp_c`, `weather_wind_kmh`, `weather_is_raining`, `humidity`, `season`
- Vehicle/driver: `model`, `battery_capacity` (often dropped to avoid leakage), `driver_profile`, `driving_style` (ordinal), `driver_personality`
- Derived: `temp_squared`, distance bins (`distance_bin_encoded`)

Dropped to avoid leakage: realized `duration_s`, `avg_speed_kmh`, segment indices within trips, per-segment physics breakdowns, post-segment timestamps.

## Inputs for inference (per edge)

Minimal row assembled by the router:
- `distance_m`, `start_time` (for hour), `start_speed_kmh`/`end_speed_kmh` (often equal to `speed_kph`),
- Context: `model`, `driver_profile`, `driving_style`, `driver_personality`,
- Weather: `weather_temp_c`, `weather_wind_kmh`, `weather_is_raining`, `season`

`engineer_features()` will create consistent numeric columns and encodings. The router then selects the saved `feature_columns` before prediction.

## Notes

- Excludes realized post-segment fields to avoid look-ahead bias
- Tree models preferred for nonlinearities; linear model kept as baseline
- Distance must be provided at inference to rescale energy_per_km to kWh

## Performance and persistence

- Shared in-memory cache of edge weights keyed by `(u,v,key,context)` dramatically reduces repeated calls during day-wide routing
- Saved bundle: models (possibly multiple), scalers, encoders, `feature_columns`, metrics, feature importances


