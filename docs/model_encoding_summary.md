# EV Consumption Model – Encoding Strategy

This summarizes how `SegmentEnergyPredictor` encodes inputs for segment-level energy prediction and how encoders persist for inference.

## Categorical encodings

- Driving style: ordinal
  - eco_friendly → 0; normal → 1; aggressive → 2
- Vehicle/driver/season: label encoding
  - Columns: `model`, `driver_profile`, `driver_personality`, `season`
  - Mappings are fitted during training and saved in the model bundle

## Temporal features

- `hour` derived from `start_time`
- `weekday`/`is_weekend` derivable but currently dropped in feature selection

## Numeric features

- `distance_m`, `log_distance_m`
- Weather numerics: `weather_temp_c`, `weather_wind_kmh`, `humidity` (if present)
- Optional derived: `temp_squared = (weather_temp_c - 15)^2`

## Feature selection and scaling

- `prepare_features()` drops datetimes/objects, keeps numeric encodings, persists `feature_columns`
- Linear models use `StandardScaler`; tree models use raw features

## Persistence

- `save_model()` stores models, scalers, encoders, feature importances, metrics, and `feature_columns`
- `load_model()` restores the full bundle for routing-time inference

## Benefits

- Preserves ordinal meaning; stable categorical mappings
- Low leakage risk by excluding realized post-segment fields
