# --- routing_features.py (DROP-IN REPLACEMENT FUNCTIONS) ---

import pandas as pd
import numpy as np
from typing import Dict, Any
from datetime import datetime

def robust_parse_datetime(series):
    """Robust datetime parsing for routing context."""
    parsed = pd.to_datetime(series, errors='coerce')
    mask_failed = parsed.isna()
    if mask_failed.any():
        # try ISO + microseconds explicitly for stragglers
        parsed2 = pd.to_datetime(series[mask_failed], format='%Y-%m-%dT%H:%M:%S.%f', errors='coerce')
        parsed[mask_failed] = parsed2
    return parsed


def engineer_features_for_routing(data: pd.DataFrame) -> pd.DataFrame:
    """
    Streamlined feature engineering for routing inference.

    IMPORTANT:
    - Keep columns that training expected (e.g., start_lat, start_lon, heading,
      temperature, wind_speed_kmh, humidity, efficiency).
    - Add safe defaults when not provided by upstream batch builder.
    - Do NOT drop training-critical columns here.
    """

    # 0) Basic typing for critical numerics (ignore if absent)
    num_cols = [
        'distance_m', 'start_elevation_m', 'end_elevation_m',
        'start_speed_kmh', 'end_speed_kmh', 'temperature', 'wind_speed_kmh',
        'humidity', 'efficiency'
    ]
    for c in num_cols:
        if c in data.columns:
            data[c] = pd.to_numeric(data[c], errors='coerce')

    # 1) Spatial basics
    if 'distance_m' in data.columns:
        data['log_distance_m'] = np.log1p(data['distance_m'])
    else:
        data['distance_m'] = 0.0
        data['log_distance_m'] = 0.0

    if {'start_elevation_m','end_elevation_m'}.issubset(data.columns):
        data['elevation_gain'] = data['end_elevation_m'] - data['start_elevation_m']
    else:
        data['elevation_gain'] = 0.0

    # 2) Temporal basics (hour, weekday)
    if 'start_time' in data.columns:
        if data['start_time'].dtype == 'object':
            data['start_time'] = robust_parse_datetime(data['start_time'])
        if hasattr(data['start_time'], 'dt'):
            data['hour'] = data['start_time'].dt.hour.astype('Int64').fillna(12).astype(int)
            data['weekday'] = data['start_time'].dt.dayofweek.astype('Int64').fillna(0).astype(int)
        else:
            data['hour'] = 12
            data['weekday'] = 0
    else:
        data['hour'] = 12
        data['weekday'] = 0

    # 3) Weather column harmonization (map routing names -> training names)
    # If training used temperature/wind_speed_kmh/humidity, populate from routing fields if necessary.
    if 'temperature' not in data.columns and 'weather_temp_c' in data.columns:
        data['temperature'] = pd.to_numeric(data['weather_temp_c'], errors='coerce')
    if 'wind_speed_kmh' not in data.columns and 'weather_wind_kmh' in data.columns:
        data['wind_speed_kmh'] = pd.to_numeric(data['weather_wind_kmh'], errors='coerce')

    # humidity can arrive as either 'humidity' or 'weather_humidity'
    if 'humidity' not in data.columns and 'weather_humidity' in data.columns:
        data['humidity'] = pd.to_numeric(data['weather_humidity'], errors='coerce')

    # safe defaults
    data['temperature'] = data.get('temperature', pd.Series([15.0]*len(data))).fillna(15.0)
    data['wind_speed_kmh'] = data.get('wind_speed_kmh', pd.Series([10.0]*len(data))).fillna(10.0)
    data['humidity'] = data.get('humidity', pd.Series([0.6]*len(data))).fillna(0.6)

    # 4) Vehicle/driver categorical encoding - match training approach
    # The training model uses LabelEncoder, so we need to handle string values properly
    # The actual encoding will be done by the predictor's align_features_for_inference method
    for cat in ['model', 'driver_profile', 'driver_personality', 'season']:
        if cat not in data.columns:
            data[cat] = 0
        else:
            # Check if the column is already numeric
            if data[cat].dtype in ['int64', 'int32', 'float64', 'float32']:
                # Already numeric, convert to int
                data[cat] = data[cat].astype(int)
            else:
                # Keep as string - the predictor's align_features_for_inference will handle encoding
                # This ensures consistency with the training model's LabelEncoder approach
                data[cat] = data[cat].astype(str)

    # driving_style: create a numeric proxy if not present
    if 'driving_style_encoded' not in data.columns:
        if 'driving_style' in data.columns:
            # rough mapping consistent with training's OrdinalEncoder order
            mapping = {'eco_friendly': 0, 'normal': 1, 'aggressive': 2}
            data['driving_style_encoded'] = data['driving_style'].map(mapping).fillna(1).astype(int)
        else:
            data['driving_style_encoded'] = 1

    # 5) Efficiency (consumption proxy) expected by training
    if 'efficiency' not in data.columns:
        # neutral 18 (Wh/m?) or kWh/100km confusion across datasets — use a small neutral,
        # the model will handle scale via other features. 18 is common Wh/km baseline -> 0.018 kWh/km
        data['efficiency'] = 18.0

    # 6) Keep training-expected spatial references if provided upstream (do NOT drop these)
    for col in ['start_lat', 'start_lon', 'heading']:
        if col not in data.columns:
            # leave missing; aligner will fill defaults later
            data[col] = np.nan

    # 7) Distance bin (integer) like training
    if 'distance_m' in data.columns:
        bins = [0, 50, 150, 400, 1200, np.inf]
        labels = [0, 1, 2, 3, 4]
        data['distance_bin_encoded'] = pd.cut(
            data['distance_m'].clip(lower=0),
            bins=bins, labels=labels, right=False, include_lowest=True
        ).astype('Int64').fillna(0).astype(int)
    else:
        data['distance_bin_encoded'] = 0

    # 8) Temp curvature feature used in training (if any)
    data['temp_squared'] = (data['temperature'] - 15.0) ** 2

    # 9) Clean NA numerics
    data = data.fillna(0)

    # 10) Critically: DO NOT drop training columns here.
    # (We’ll align the final frame to the model’s feature_columns in the predictor.)

    return data
