"""
📊 ANALYSIS MODEL - EV Consumption Analysis
Uses ALL available features including post-trip data for understanding patterns
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
    📊 ANALYSIS MODEL - EV Consumption Analysis
    
    Uses ALL available features including post-trip data to understand
    consumption patterns and identify optimization opportunities.
    
    Features used (ALL DATA):
    - Trip actuals: actual distance, time, speed, route efficiency
    - Location clusters and route analysis
    - Historical patterns and aggregates
    - Complete weather impact analysis
    
    Target: total_consumption_kwh
    Purpose: Understand "what happened" not "what will happen"
    """
    
    def __init__(self, data_path: str = None):
        self.data_path = data_path
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
        """Load and merge datasets - same as predictive model"""
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
        
        self.logger.info(f"Loaded {len(data)} records for ANALYSIS model")
        return data
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create ALL features including post-trip analysis features"""
        
        # Time-based features
        data['date'] = pd.to_datetime(data['date'])
        data['hour'] = data['date'].dt.hour
        data['day_of_week'] = data['date'].dt.dayofweek
        data['month'] = data['date'].dt.month
        data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
        data['is_rush_hour'] = ((data['hour'].between(7, 9)) | (data['hour'].between(17, 19))).astype(int)
        
        # Distance and speed features (POST-TRIP)
        data['avg_speed_kmh'] = np.where(data['total_time_minutes'] > 0,
                                        data['total_distance_km'] / (data['total_time_minutes'] / 60),
                                        0)
        
        # Trip efficiency features (POST-TRIP)
        data['distance_per_minute'] = np.where(data['total_time_minutes'] > 0,
                                              data['total_distance_km'] / data['total_time_minutes'],
                                              0)
        
        # Calculate straight-line distance
        data['straight_line_distance_km'] = np.sqrt(
            (data['destination_lat'] - data['origin_lat'])**2 + 
            (data['destination_lon'] - data['origin_lon'])**2
        ) * 111.32
        
        # Route efficiency (POST-TRIP)
        data['route_efficiency'] = np.where(data['straight_line_distance_km'] > 0,
                                           data['total_distance_km'] / data['straight_line_distance_km'],
                                           1.0)
        data['route_efficiency'] = np.clip(data['route_efficiency'], 1.0, 10.0)
        
        # Trip type classification (POST-TRIP)
        data['trip_type'] = 'local'
        data.loc[(data['avg_speed_kmh'] > 60) & (data['total_distance_km'] > 30), 'trip_type'] = 'highway'
        data.loc[(data['avg_speed_kmh'] < 25) & (data['total_distance_km'] < 10), 'trip_type'] = 'city'
        
        # Weather impact features (POST-TRIP)
        data['temp_squared'] = data['temperature_celsius'] ** 2
        data['temp_deviation'] = abs(data['temperature_celsius'] - 20)
        data['wind_impact'] = data['weather_wind_speed_kmh'] * data['total_distance_km']
        data['rain_impact'] = data['weather_is_raining'].astype(int) * data['total_distance_km']
        
        # Vehicle-specific features
        data['battery_usage_ratio'] = np.where(data['battery_capacity'] > 0,
                                             data['total_consumption_kwh'] / data['battery_capacity'],
                                             0)
        
        # Charging capability impact
        data['fast_charging_capable'] = (data['max_charging_speed'] > 50).astype(int)
        data['home_charging_available'] = data['has_home_charging'].astype(int)
        
        return data
    
    def prepare_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        📊 ANALYSIS MODEL: Use ALL available features for pattern analysis
        """
        
        # Target variable
        target = data['total_consumption_kwh']
        
        # 📊 ALL FEATURES for analysis (including post-trip data)
        feature_cols = [
            # Trip characteristics (POST-TRIP - actual values)
            'total_distance_km', 'total_time_minutes', 'avg_speed_kmh', 'distance_per_minute',
            'route_efficiency', 'straight_line_distance_km',
            
            # Time features (ALL)
            'hour', 'day_of_week', 'month', 'is_weekend', 'is_rush_hour',
            
            # Weather features (COMPLETE impact analysis)
            'temperature_celsius', 'temp_squared', 'temp_deviation', 'weather_wind_speed_kmh', 
            'weather_humidity', 'wind_impact', 'rain_impact', 'weather_is_raining', 'weather_season',
            
            # Vehicle features (COMPLETE)
            'battery_capacity', 'efficiency', 'max_charging_speed', 'fast_charging_capable',
            'home_charging_available', 'battery_usage_ratio',
            
            # Driver features  
            'driver_profile', 'model', 'driving_style', 'driver_personality',
            
            # Trip classification (POST-TRIP)
            'trip_type',
            
            # Temperature efficiency
            'temperature_efficiency_factor'
        ]
        
        # Select available features
        available_features = [col for col in feature_cols if col in data.columns]
        missing_features = [col for col in feature_cols if col not in data.columns]
        
        if missing_features:
            self.logger.warning(f"Missing features: {missing_features}")
        
        features = data[available_features].copy()
        
        # Handle categorical encoding (simple label encoding for analysis)
        categorical_cols = ['driver_profile', 'model', 'driving_style', 'driver_personality', 'weather_season', 'trip_type']
        available_categorical = [col for col in categorical_cols if col in features.columns]
        
        for col in available_categorical:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                features[col] = self.encoders[col].fit_transform(features[col].astype(str))
            else:
                # Handle unseen categories
                values = features[col].astype(str)
                encoded_values = []
                for val in values:
                    if val in self.encoders[col].classes_:
                        encoded_values.append(self.encoders[col].transform([val])[0])
                    else:
                        encoded_values.append(0)
                features[col] = encoded_values
        
        # Handle missing values
        features = features.fillna(features.median())
        
        # Store feature columns
        self.feature_columns = features.columns.tolist()
        
        return features, target
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series):
        """Train models for analysis"""
        
        # Scale features for linear model
        self.scalers['standard'] = StandardScaler()
        X_train_scaled = self.scalers['standard'].fit_transform(X_train)
        X_val_scaled = self.scalers['standard'].transform(X_val)
        
        # Define models
        models_config = {
            'random_forest': RandomForestRegressor(
                n_estimators=100, max_depth=15, min_samples_split=5,
                min_samples_leaf=2, random_state=42, n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
            )
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models_config['xgboost'] = xgb.XGBRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                random_state=42, eval_metric='rmse'
            )
        
        # Train and evaluate models
        for name, model in models_config.items():
            self.logger.info(f"Training {name} for ANALYSIS...")
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            # Calculate metrics
            mae = mean_absolute_error(y_val, y_pred)
            mse = mean_squared_error(y_val, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_val, y_pred)
            
            self.models[name] = model
            self.model_performance[name] = {
                'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2
            }
            
            # Feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = dict(zip(X_train.columns, model.feature_importances_))
            
            self.logger.info(f"{name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
    
    def get_analysis_insights(self) -> Dict:
        """Generate insights from analysis model"""
        insights = {
            'high_consumption_factors': [],
            'efficiency_patterns': {},
            'optimization_opportunities': []
        }
        
        # Get feature importance from best model
        best_model_name = max(self.model_performance.keys(), 
                             key=lambda x: self.model_performance[x]['R2'])
        
        if best_model_name in self.feature_importance:
            importances = self.feature_importance[best_model_name]
            sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
            
            insights['top_consumption_drivers'] = sorted_features[:10]
            insights['low_impact_features'] = sorted_features[-5:]
        
        return insights

def main():
    """Train analysis model"""
    
    # Initialize analyzer  
    analyzer = EVConsumptionAnalyzer()
    
    # Load and prepare data
    print("📊 Training ANALYSIS model (all features including post-trip)...")
    data = analyzer.load_data()
    data = analyzer.engineer_features(data)
    X, y = analyzer.prepare_features(data)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples") 
    print(f"Test set: {len(X_test)} samples")
    
    # Train models
    analyzer.train_models(X_train, y_train, X_val, y_val)
    
    # Get insights
    insights = analyzer.get_analysis_insights()
    print("\n📊 Analysis Insights:")
    print("Top consumption drivers:", insights.get('top_consumption_drivers', [])[:5])
    
    # Save model
    model_path = Path(__file__).parent / "consumption_analyzer.pkl"
    analyzer.save_model = lambda path: joblib.dump({
        'models': analyzer.models,
        'encoders': analyzer.encoders,
        'scalers': analyzer.scalers,
        'feature_columns': analyzer.feature_columns
    }, path)
    analyzer.save_model(model_path)
    print(f"\nAnalysis model saved to {model_path}")

if __name__ == "__main__":
    main()
