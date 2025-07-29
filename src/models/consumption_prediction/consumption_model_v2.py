"""
📊 ANALYSIS MODEL - EV Consumption Analysis  
Uses all available features including post-trip data for pattern analysis
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import logging
from pathlib import Path
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class EVConsumptionAnalyzer:
    """
    Fixed EV Consumption Prediction Model
    - Removes look-ahead bias
    - Improves vehicle model encoding
    - Separates analysis vs prediction features
    """
    
    def __init__(self, data_path: str = None, model_type: str = 'predictive'):
        """
        model_type: 'predictive' (no look-ahead) or 'analysis' (all features)
        """
        self.data_path = data_path
        self.model_type = model_type
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = {}
        self.model_performance = {}
        self.feature_columns = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def load_data(self) -> pd.DataFrame:
        """Load and merge datasets"""
        if self.data_path is None:
            self.data_path = Path(__file__).parent.parent.parent.parent / "data" / "synthetic"
        
        # Load datasets
        routes = pd.read_csv(self.data_path / "routes.csv")
        fleet_info = pd.read_csv(self.data_path / "fleet_info.csv")
        weather = pd.read_csv(self.data_path / "weather.csv")
        
        # Merge data
        vehicle_features = ['vehicle_id', 'model', 'battery_capacity', 'efficiency', 
                           'max_charging_speed', 'driving_style', 
                           'driver_personality', 'has_home_charging']
        
        data = routes.merge(fleet_info[vehicle_features], on='vehicle_id', how='left')
        data = data.merge(weather, on='date', how='left')
        
        # Remove consumption component features (data leakage)
        consumption_components = [
            'consumption_rolling_resistance_kwh', 'consumption_aerodynamic_drag_kwh', 
            'consumption_elevation_change_kwh', 'consumption_acceleration_kwh',
            'consumption_hvac_kwh', 'consumption_auxiliary_kwh',
            'consumption_regenerative_braking_kwh', 'consumption_battery_thermal_loss_kwh'
        ]
        
        existing_components = [col for col in consumption_components if col in data.columns]
        if existing_components:
            data = data.drop(columns=existing_components)
            self.logger.info(f"Removed {len(existing_components)} consumption component features")
        
        self.logger.info(f"Loaded {len(data)} records for {self.model_type} model")
        return data
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features based on model type"""
        
        # Always available features
        data['date'] = pd.to_datetime(data['date'])
        data['hour'] = data['date'].dt.hour
        data['is_weekend'] = (data['date'].dt.dayofweek >= 5).astype(int)
        
        # Calculate straight-line distance (always available from coordinates)
        data['straight_line_distance_km'] = np.sqrt(
            (data['destination_lat'] - data['origin_lat'])**2 + 
            (data['destination_lon'] - data['origin_lon'])**2
        ) * 111.32  # Approximate km per degree
        
        # Weather impact features
        data['temp_squared'] = data['temperature_celsius'] ** 2
        data['temp_deviation'] = abs(data['temperature_celsius'] - 20)
        
        # Vehicle encoding (improved approach)
        data = self._encode_vehicle_models(data)
        
        if self.model_type == 'analysis':
            # POST-trip features (for analysis only)
            self._add_analysis_features(data)
        else:
            # PRE-trip features only (for prediction)
            self._add_predictive_features(data)
        
        return data
    
    def _encode_vehicle_models(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Improved vehicle model encoding using grouped categories + efficiency
        Instead of one-hot encoding all models, group by characteristics
        """
        
        # Define vehicle categories based on actual characteristics
        vehicle_categories = {
            # Tesla models
            'tesla_model_3': {'category': 'premium_sedan', 'brand': 'tesla'},
            'tesla_model_y': {'category': 'premium_suv', 'brand': 'tesla'},
            'tesla_model_s': {'category': 'luxury_sedan', 'brand': 'tesla'},
            
            # Economy models  
            'nissan_leaf': {'category': 'economy_hatch', 'brand': 'nissan'},
            'renault_zoe': {'category': 'economy_hatch', 'brand': 'renault'},
            'chevy_bolt': {'category': 'economy_hatch', 'brand': 'chevrolet'},
            
            # Premium SUVs
            'ford_mustang_mach_e': {'category': 'premium_suv', 'brand': 'ford'},
            'hyundai_ioniq_5': {'category': 'premium_suv', 'brand': 'hyundai'},
            'kia_ev6': {'category': 'premium_suv', 'brand': 'kia'},
            
            # Luxury
            'audi_e_tron_gt': {'category': 'luxury_sedan', 'brand': 'audi'},
            'bmw_ix': {'category': 'luxury_suv', 'brand': 'bmw'},
        }
        
        # Map models to categories
        data['vehicle_category'] = data['model'].map(
            lambda x: vehicle_categories.get(x, {}).get('category', 'other')
        )
        data['vehicle_brand'] = data['model'].map(
            lambda x: vehicle_categories.get(x, {}).get('brand', 'other')
        )
        
        # Fast charging capability (more informative than individual models)
        data['fast_charging_capable'] = (data['max_charging_speed'] > 50).astype(int)
        
        # Battery size category
        data['battery_size_category'] = pd.cut(
            data['battery_capacity'], 
            bins=[0, 50, 70, 90, 150], 
            labels=['small', 'medium', 'large', 'xlarge']
        ).astype(str)
        
        return data
    
    def _add_analysis_features(self, data: pd.DataFrame):
        """Add features only available AFTER trip completion (for analysis)"""
        
        # Actual trip characteristics
        data['avg_speed_kmh'] = np.where(
            data['total_time_minutes'] > 0,
            data['total_distance_km'] / (data['total_time_minutes'] / 60),
            0
        )
        
        # Route efficiency (actual vs straight line)
        data['route_efficiency'] = np.where(
            data['straight_line_distance_km'] > 0,
            data['total_distance_km'] / data['straight_line_distance_km'],
            1.0
        )
        data['route_efficiency'] = np.clip(data['route_efficiency'], 1.0, 5.0)
        
        # Trip type based on actual data
        data['trip_type'] = 'local'
        data.loc[(data['avg_speed_kmh'] > 60) & (data['total_distance_km'] > 30), 'trip_type'] = 'highway'
        data.loc[(data['avg_speed_kmh'] < 25) & (data['total_distance_km'] < 10), 'trip_type'] = 'city'
        
        # Weather impact with actual distance
        data['wind_impact'] = data['weather_wind_speed_kmh'] * data['total_distance_km']
        data['rain_impact'] = data['weather_is_raining'].astype(int) * data['total_distance_km']
    
    def _add_predictive_features(self, data: pd.DataFrame):
        """Add features available BEFORE trip starts (for prediction)"""
        
        # Estimated trip characteristics (would come from routing API in production)
        # For now, use straight-line distance with realistic factors
        data['estimated_route_distance'] = data['straight_line_distance_km'] * np.random.uniform(1.2, 1.8, len(data))
        
        # Time of day impact on traffic
        data['traffic_factor'] = 1.0
        rush_hours = ((data['hour'].between(7, 9)) | (data['hour'].between(17, 19)))
        data.loc[rush_hours, 'traffic_factor'] = 1.3
        data.loc[data['is_weekend'] == 1, 'traffic_factor'] *= 0.8
        
        # Estimated travel time based on distance and traffic
        data['estimated_travel_time'] = (
            data['estimated_route_distance'] / 45 * 60 * data['traffic_factor']  # 45 km/h average
        )
        
        # Weather impact (base level, not distance dependent)
        data['wind_impact_base'] = data['weather_wind_speed_kmh'] * 0.1
        data['rain_impact_base'] = data['weather_is_raining'].astype(int) * 0.2
        
        # Trip distance category (available from route planning)
        data['distance_category'] = pd.cut(
            data['estimated_route_distance'],
            bins=[0, 10, 30, 100, 500],
            labels=['very_short', 'short', 'medium', 'long']
        ).astype(str)
    
    def prepare_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features based on model type"""
        
        target = data['total_consumption_kwh']
        
        if self.model_type == 'analysis':
            feature_cols = self._get_analysis_features()
        else:
            feature_cols = self._get_predictive_features()
        
        # Select available features
        available_features = [col for col in feature_cols if col in data.columns]
        missing_features = [col for col in feature_cols if col not in data.columns]
        
        if missing_features:
            self.logger.warning(f"Missing features: {missing_features}")
        
        features = data[available_features].copy()
        
        # Handle categorical encoding
        features = self._encode_categorical_features(features)
        
        # Handle missing values
        features = features.fillna(features.median())
        
        self.feature_columns = features.columns.tolist()
        return features, target
    
    def _get_analysis_features(self) -> list:
        """Features for post-trip analysis model"""
        return [
            # Trip characteristics (actual)
            'total_distance_km', 'total_time_minutes', 'avg_speed_kmh', 'route_efficiency',
            
            # Time features
            'hour', 'is_weekend',
            
            # Weather features  
            'temperature_celsius', 'temp_squared', 'temp_deviation',
            'weather_wind_speed_kmh', 'weather_humidity', 'wind_impact', 'rain_impact',
            'weather_is_raining',
            
            # Vehicle features (improved encoding)
            'battery_capacity', 'efficiency', 'vehicle_category', 'vehicle_brand',
            'fast_charging_capable', 'battery_size_category',
            
            # Driver features
            'driving_style', 'driver_profile', 'driver_personality',
            
            # Location
            'straight_line_distance_km',
            
            # Trip type (derived from actual data)
            'trip_type',
            
            # Weather conditions
            'weather_season'
        ]
    
    def _get_predictive_features(self) -> list:
        """Features for pre-trip prediction model"""
        return [
            # Trip planning (available from routing)
            'estimated_route_distance', 'estimated_travel_time', 'straight_line_distance_km',
            'distance_category', 'traffic_factor',
            
            # Time features
            'hour', 'is_weekend',
            
            # Weather forecast
            'temperature_celsius', 'temp_squared', 'temp_deviation',
            'weather_wind_speed_kmh', 'weather_humidity', 'wind_impact_base', 'rain_impact_base',
            'weather_is_raining', 'weather_season',
            
            # Vehicle features (improved encoding)
            'battery_capacity', 'efficiency', 'vehicle_category', 'vehicle_brand',
            'fast_charging_capable', 'battery_size_category',
            
            # Driver features
            'driving_style', 'driver_profile', 'driver_personality'
        ]
    
    def _encode_categorical_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Handle categorical encoding with improved strategy"""
        
        # 1. Ordinal encoding for driving_style (preserves order)
        if 'driving_style' in features.columns:
            if 'driving_style' not in self.encoders:
                driving_style_map = {'eco_friendly': 0, 'normal': 1, 'aggressive': 2}
                features['driving_style'] = features['driving_style'].map(driving_style_map)
                self.encoders['driving_style'] = driving_style_map
            else:
                features['driving_style'] = features['driving_style'].map(self.encoders['driving_style'])
        
        # 2. One-hot for categorical features with few categories
        onehot_cols = ['driver_profile', 'driver_personality', 'weather_season', 
                       'vehicle_category', 'vehicle_brand', 'battery_size_category']
        
        if self.model_type == 'analysis':
            onehot_cols.append('trip_type')
        else:
            onehot_cols.append('distance_category')
        
        available_onehot = [col for col in onehot_cols if col in features.columns]
        
        for col in available_onehot:
            if col not in self.encoders:
                dummies = pd.get_dummies(features[col], prefix=col, drop_first=True)
                self.encoders[col] = list(dummies.columns)
                features = pd.concat([features, dummies], axis=1)
                features = features.drop(columns=[col])
            else:
                # Apply existing encoding
                dummy_cols = self.encoders[col]
                current_dummies = pd.get_dummies(features[col], prefix=col, drop_first=True)
                
                for dummy_col in dummy_cols:
                    if dummy_col not in current_dummies.columns:
                        current_dummies[dummy_col] = 0
                
                current_dummies = current_dummies[dummy_cols]
                features = pd.concat([features, current_dummies], axis=1)
                features = features.drop(columns=[col])
        
        return features
    
    # ... (rest of the methods similar to original, but adapted for the two model types)
    
    def train_models(self, X_train, y_train, X_val, y_val):
        """Train models with same approach as original"""
        # Similar implementation to original...
        pass
    
    def save_model(self, model_path: str):
        """Save model with type information"""
        model_data = {
            'models': self.models,
            'scalers': self.scalers, 
            'encoders': self.encoders,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type
        }
        joblib.dump(model_data, model_path)
        self.logger.info(f"{self.model_type.title()} model saved to {model_path}")

def main():
    """Train both analysis and predictive models"""
    
    print("Training ANALYSIS model (post-trip features)...")
    analysis_model = EVConsumptionPredictor_v2(model_type='analysis')
    data = analysis_model.load_data()
    data = analysis_model.engineer_features(data)
    X, y = analysis_model.prepare_features(data)
    # ... train and save
    
    print("\nTraining PREDICTIVE model (pre-trip features only)...")
    predictive_model = EVConsumptionPredictor_v2(model_type='predictive')
    data = predictive_model.load_data()
    data = predictive_model.engineer_features(data)
    X, y = predictive_model.prepare_features(data)
    # ... train and save

if __name__ == "__main__":
    main()
