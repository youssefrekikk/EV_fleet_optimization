import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import logging
from pathlib import Path
from typing import Tuple, Dict, Any
from sklearn.cluster import KMeans
from geopy.distance import geodesic
import warnings
warnings.filterwarnings('ignore')

class EVConsumptionPredictor:
    """
    Electric Vehicle Consumption Prediction Model
    
    This class implements multiple ML models to predict EV energy consumption
    based on trip characteristics, weather conditions, and vehicle specifications.
    """
    
    def __init__(self, data_path: str = None):
        self.data_path = data_path
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = {}
        self.model_performance = {}
        self.feature_columns = None
        self.location_clusters = None
        self.cluster_model = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def load_data(self) -> pd.DataFrame:
        """Load and merge relevant datasets for consumption prediction"""
        
        if self.data_path is None:
            self.data_path = Path(__file__).parent.parent.parent.parent / "data" / "synthetic"
        
        # Load main datasets
        routes = pd.read_csv(self.data_path / "routes.csv")
        fleet_info = pd.read_csv(self.data_path / "fleet_info.csv")
        weather = pd.read_csv(self.data_path / "weather.csv")
        
        # Merge routes with vehicle information (avoid duplicating driver_profile)
        # Routes already has driver_profile, so get other vehicle attributes from fleet_info
        vehicle_features = ['vehicle_id', 'model', 'battery_capacity', 'efficiency', 
                           'max_charging_speed', 'driving_style', 
                           'driver_personality', 'has_home_charging']
        
        data = routes.merge(fleet_info[vehicle_features], on='vehicle_id', how='left')
        
        # Merge with weather data
        data = data.merge(weather, on='date', how='left')
        
        # Remove consumption component features to avoid data leakage
        consumption_components = [
            'consumption_rolling_resistance_kwh',
            'consumption_aerodynamic_drag_kwh', 
            'consumption_elevation_change_kwh',
            'consumption_acceleration_kwh',
            'consumption_hvac_kwh',
            'consumption_auxiliary_kwh',
            'consumption_regenerative_braking_kwh',
            'consumption_battery_thermal_loss_kwh'
        ]
        
        # Drop consumption components if they exist
        existing_components = [col for col in consumption_components if col in data.columns]
        if existing_components:
            data = data.drop(columns=existing_components)
            self.logger.info(f"Removed {len(existing_components)} consumption component features to avoid data leakage")
        
        self.logger.info(f"Loaded {len(data)} records for training")
        return data
    
    def create_location_clusters(self, data: pd.DataFrame, n_clusters: int = 10) -> pd.DataFrame:
        """Create location clusters for origin and destination points"""
        
        # Combine all unique locations
        origins = data[['origin_lat', 'origin_lon']].drop_duplicates()
        destinations = data[['destination_lat', 'destination_lon']].drop_duplicates()
        
        # Rename columns for concatenation
        origins.columns = ['lat', 'lon']
        destinations.columns = ['lat', 'lon']
        
        all_locations = pd.concat([origins, destinations]).drop_duplicates()
        
        if len(all_locations) < n_clusters:
            n_clusters = max(1, len(all_locations) // 2)
        
        # Fit KMeans clustering
        self.cluster_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.cluster_model.fit(all_locations[['lat', 'lon']])
        
        # Predict clusters for origins and destinations
        # Need to rename columns to match training data
        origin_data = data[['origin_lat', 'origin_lon']].copy()
        origin_data.columns = ['lat', 'lon']
        origin_clusters = self.cluster_model.predict(origin_data)
        
        dest_data = data[['destination_lat', 'destination_lon']].copy()
        dest_data.columns = ['lat', 'lon']
        dest_clusters = self.cluster_model.predict(dest_data)
        
        data['origin_cluster'] = origin_clusters
        data['destination_cluster'] = dest_clusters
        data['same_cluster'] = (origin_clusters == dest_clusters).astype(int)
        
        return data
    
    def calculate_route_efficiency(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate route efficiency metrics"""
        
        # Calculate straight-line distance using simple approximation
        # For more accuracy, use geopy but add fallback for when it's not available
        try:
            # Try to use geopy for accurate distance calculation
            def haversine_distance(lat1, lon1, lat2, lon2):
                try:
                    return geodesic((lat1, lon1), (lat2, lon2)).kilometers
                except:
                    # Fallback to simple approximation
                    lat1_rad = np.radians(lat1)
                    lon1_rad = np.radians(lon1)
                    lat2_rad = np.radians(lat2)
                    lon2_rad = np.radians(lon2)
                    
                    dlat = lat2_rad - lat1_rad
                    dlon = lon2_rad - lon1_rad
                    
                    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
                    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
                    return 6371.0 * c  # Earth's radius in km
            
            data['straight_line_distance_km'] = data.apply(
                lambda row: haversine_distance(row['origin_lat'], row['origin_lon'], 
                                             row['destination_lat'], row['destination_lon']), axis=1
            )
        except:
            # Simple approximation fallback
            data['straight_line_distance_km'] = np.sqrt(
                (data['destination_lat'] - data['origin_lat'])**2 + 
                (data['destination_lon'] - data['origin_lon'])**2
            ) * 111.32  # Approximate km per degree
        
        # Route efficiency (actual distance vs straight line)
        data['route_efficiency'] = np.where(data['straight_line_distance_km'] > 0,
                                           data['total_distance_km'] / data['straight_line_distance_km'],
                                           1.0)
        
        # Cap extreme values
        data['route_efficiency'] = np.clip(data['route_efficiency'], 1.0, 10.0)
        
        return data
    
    def create_aggregate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create aggregate features based on historical patterns"""
        
        # Sort by vehicle and date for rolling calculations
        data = data.sort_values(['vehicle_id', 'date'])
        
        # Vehicle-level rolling averages (last 7 trips)
        data['vehicle_avg_consumption_7d'] = data.groupby('vehicle_id')['total_consumption_kwh'].rolling(
            window=7, min_periods=1).mean().reset_index(level=0, drop=True)
        
        data['vehicle_avg_efficiency_7d'] = data.groupby('vehicle_id')['efficiency_kwh_per_100km'].rolling(
            window=7, min_periods=1).mean().reset_index(level=0, drop=True)
        
        # Driver-level patterns
        data['driver_avg_consumption_7d'] = data.groupby('driver_profile')['total_consumption_kwh'].rolling(
            window=7, min_periods=1).mean().reset_index(level=0, drop=True)
        
        # Trip frequency features
        data['trips_last_7d'] = data.groupby('vehicle_id').cumcount() + 1
        data['trips_last_7d'] = np.minimum(data['trips_last_7d'], 7)
        
        # Distance patterns
        data['vehicle_total_distance_7d'] = data.groupby('vehicle_id')['total_distance_km'].rolling(
            window=7, min_periods=1).sum().reset_index(level=0, drop=True)
        
        return data
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create additional features for better prediction"""
        
        # Time-based features
        data['date'] = pd.to_datetime(data['date'])
        data['hour'] = data['date'].dt.hour
        data['day_of_week'] = data['date'].dt.dayofweek
        data['month'] = data['date'].dt.month
        data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
        data['is_rush_hour'] = ((data['hour'].between(7, 9)) | (data['hour'].between(17, 19))).astype(int)
        
        # Distance and speed features
        data['avg_speed_kmh'] = np.where(data['total_time_minutes'] > 0,
                                        data['total_distance_km'] / (data['total_time_minutes'] / 60),
                                        0)
        
        # Trip efficiency features
        data['distance_per_minute'] = np.where(data['total_time_minutes'] > 0,
                                              data['total_distance_km'] / data['total_time_minutes'],
                                              0)
        
        # Classify trip type based on speed and distance
        data['trip_type'] = 'local'
        data.loc[(data['avg_speed_kmh'] > 60) & (data['total_distance_km'] > 30), 'trip_type'] = 'highway'
        data.loc[(data['avg_speed_kmh'] < 25) & (data['total_distance_km'] < 10), 'trip_type'] = 'city'
        
        # Weather impact features
        data['temp_squared'] = data['temperature_celsius'] ** 2
        data['temp_deviation'] = abs(data['temperature_celsius'] - 20)  # Deviation from optimal temperature
        data['wind_impact'] = data['weather_wind_speed_kmh'] * data['total_distance_km']
        data['rain_impact'] = data['weather_is_raining'].astype(int) * data['total_distance_km']
        
        # Vehicle-specific features
        data['battery_usage_ratio'] = np.where(data['battery_capacity'] > 0,
                                             data['total_consumption_kwh'] / data['battery_capacity'],
                                             0)
        
        # Charging capability impact
        data['fast_charging_capable'] = (data['max_charging_speed'] > 50).astype(int)
        data['home_charging_available'] = data['has_home_charging'].astype(int)
        
        # Create location-based features
        data = self.create_location_clusters(data)
        data = self.calculate_route_efficiency(data)
        
        # Create aggregate features
        data = self.create_aggregate_features(data)
        
        return data
    
    def prepare_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target variable"""
        
        # Target variable
        target = data['total_consumption_kwh']
        
        # Feature selection - Updated with new features
        feature_cols = [
            # Trip characteristics
            'total_distance_km', 'total_time_minutes', 'avg_speed_kmh', 'distance_per_minute',
            
            # Time features
            'hour', 'day_of_week', 'month', 'is_weekend', 'is_rush_hour',
            
            # Weather features
            'temperature_celsius', 'temp_squared', 'temp_deviation', 'weather_wind_speed_kmh', 
            'weather_humidity', 'wind_impact', 'rain_impact', 'weather_is_raining',
            
            # Vehicle features
            'battery_capacity', 'efficiency', 'max_charging_speed', 'fast_charging_capable',
            'home_charging_available',
            
            # Location and route features
            'origin_cluster', 'destination_cluster', 'same_cluster', 'route_efficiency',
            'straight_line_distance_km',
            
            # Historical/aggregate features
            'vehicle_avg_consumption_7d', 'vehicle_avg_efficiency_7d', 'driver_avg_consumption_7d',
            'trips_last_7d', 'vehicle_total_distance_7d',
            
            # Temperature efficiency (if available)
            'temperature_efficiency_factor',
            
            # Categorical features
            'driver_profile', 'model', 'driving_style', 'driver_personality', 'weather_season', 'trip_type'
        ]
        
        # Only select features that exist in the data
        available_features = [col for col in feature_cols if col in data.columns]
        missing_features = [col for col in feature_cols if col not in data.columns]
        
        if missing_features:
            self.logger.warning(f"Missing features: {missing_features}")
        
        features = data[available_features].copy()
        
        # Handle categorical variables (only those that exist in features)
        categorical_cols = ['driver_profile', 'model', 'driving_style', 'driver_personality', 'weather_season', 'trip_type']
        available_categorical = [col for col in categorical_cols if col in features.columns]
        
        for col in available_categorical:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                features[col] = self.encoders[col].fit_transform(features[col].astype(str))
            else:
                features[col] = self.encoders[col].transform(features[col].astype(str))
        
        # Handle missing values
        features = features.fillna(features.median())
        
        # Store feature columns for prediction
        self.feature_columns = features.columns.tolist()
        
        return features, target
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series):
        """Train multiple models and compare performance"""
        
        # Scale features for linear model
        self.scalers['standard'] = StandardScaler()
        X_train_scaled = self.scalers['standard'].fit_transform(X_train)
        X_val_scaled = self.scalers['standard'].transform(X_val)
        
        # Define models
        models_config = {
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models_config['xgboost'] = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='rmse'
            )
        
        # Train and evaluate models
        for name, model in models_config.items():
            self.logger.info(f"Training {name}...")
            
            if name == 'linear_regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_val_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
            
            # Calculate metrics
            mae = mean_absolute_error(y_val, y_pred)
            mse = mean_squared_error(y_val, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_val, y_pred)
            
            self.models[name] = model
            self.model_performance[name] = {
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'R2': r2
            }
            
            # Feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = dict(zip(X_train.columns, model.feature_importances_))
            
            self.logger.info(f"{name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
    
    def get_best_model(self) -> str:
        """Select best model based on R2 score"""
        best_model = max(self.model_performance.keys(), 
                        key=lambda x: self.model_performance[x]['R2'])
        return best_model
    
    def predict(self, features: pd.DataFrame, model_name: str = None) -> np.ndarray:
        """Make predictions using trained model"""
        
        if model_name is None:
            model_name = self.get_best_model()
        
        model = self.models[model_name]
        
        if model_name == 'linear_regression':
            features_scaled = self.scalers['standard'].transform(features)
            return model.predict(features_scaled)
        else:
            return model.predict(features)
    
    def predict_trip_consumption(self, distance_km: float, time_minutes: float, 
                                temperature: float, vehicle_model: str = 'tesla_model_3',
                                driver_profile: str = 'commuter', **kwargs) -> Dict[str, Any]:
        """Predict consumption for a single trip"""
        
        # Create feature vector with all new features
        trip_data = {
            # Basic trip data
            'total_distance_km': distance_km,
            'total_time_minutes': time_minutes,
            'avg_speed_kmh': distance_km / (time_minutes / 60) if time_minutes > 0 else 0,
            'distance_per_minute': distance_km / time_minutes if time_minutes > 0 else 0,
            
            # Time features
            'hour': kwargs.get('hour', 12),
            'day_of_week': kwargs.get('day_of_week', 1),
            'month': kwargs.get('month', 6),
            'is_weekend': kwargs.get('is_weekend', 0),
            'is_rush_hour': kwargs.get('is_rush_hour', 0),
            
            # Weather features
            'temperature_celsius': temperature,
            'temp_squared': temperature ** 2,
            'temp_deviation': abs(temperature - 20),
            'weather_wind_speed_kmh': kwargs.get('wind_speed', 10),
            'weather_humidity': kwargs.get('humidity', 0.5),
            'wind_impact': kwargs.get('wind_speed', 10) * distance_km,
            'rain_impact': kwargs.get('rain', 0) * distance_km,
            'weather_is_raining': kwargs.get('rain', 0),
            'weather_season': kwargs.get('season', 'winter'),
            
            # Vehicle features
            'model': vehicle_model,
            'battery_capacity': kwargs.get('battery_capacity', 75),
            'efficiency': kwargs.get('efficiency', 15),
            'max_charging_speed': kwargs.get('max_charging_speed', 100),
            'fast_charging_capable': 1 if kwargs.get('max_charging_speed', 100) > 50 else 0,
            'has_home_charging': kwargs.get('has_home_charging', True),
            'home_charging_available': 1 if kwargs.get('has_home_charging', True) else 0,
            
            # Driver features
            'driver_profile': driver_profile,
            'driving_style': kwargs.get('driving_style', 'normal'),
            'driver_personality': kwargs.get('driver_personality', 'optimizer'),
            
            # Location features (defaults - will be updated if cluster model available)
            'origin_cluster': 0,
            'destination_cluster': 0,
            'same_cluster': 1,
            'route_efficiency': 1.2,
            'straight_line_distance_km': distance_km * 0.8,
            
            # Trip type
            'trip_type': 'local',
            
            # Historical features (defaults)
            'vehicle_avg_consumption_7d': kwargs.get('avg_consumption', distance_km * 0.15),
            'vehicle_avg_efficiency_7d': kwargs.get('avg_efficiency', 15),
            'driver_avg_consumption_7d': kwargs.get('driver_avg_consumption', distance_km * 0.15),
            'trips_last_7d': kwargs.get('trips_count', 5),
            'vehicle_total_distance_7d': kwargs.get('total_distance_7d', distance_km * 5),
            
            # Temperature efficiency
            'temperature_efficiency_factor': kwargs.get('temp_factor', 1.0)
        }
        
        # Update trip type based on speed and distance
        avg_speed = trip_data['avg_speed_kmh']
        if avg_speed > 60 and distance_km > 30:
            trip_data['trip_type'] = 'highway'
        elif avg_speed < 25 and distance_km < 10:
            trip_data['trip_type'] = 'city'
        
        # Convert to DataFrame
        trip_df = pd.DataFrame([trip_data])
        
        # Encode categorical variables
        categorical_cols = ['driver_profile', 'model', 'driving_style', 'driver_personality', 'weather_season', 'trip_type']
        for col in categorical_cols:
            if col in self.encoders:
                # Handle unseen labels by mapping to most common class
                encoder = self.encoders[col]
                values = trip_df[col].astype(str)
                encoded_values = []
                for val in values:
                    if val in encoder.classes_:
                        encoded_values.append(encoder.transform([val])[0])
                    else:
                        # Use the most frequent class (index 0) for unseen labels
                        encoded_values.append(0)
                trip_df[col] = encoded_values
        
        # Ensure we have all required features and in the correct order
        if self.feature_columns is not None:
            # Create DataFrame with all required features
            prediction_data = {}
            for col in self.feature_columns:
                if col in trip_df.columns:
                    prediction_data[col] = trip_df[col].iloc[0]
                else:
                    prediction_data[col] = 0  # Default value for missing features
            
            trip_df = pd.DataFrame([prediction_data])
        
        # Make prediction
        predicted_consumption = self.predict(trip_df)[0]
        
        # Calculate efficiency
        efficiency_kwh_per_100km = (predicted_consumption / distance_km * 100) if distance_km > 0 else 0
        
        return {
            'predicted_consumption_kwh': predicted_consumption,
            'efficiency_kwh_per_100km': efficiency_kwh_per_100km,
            'model_used': self.get_best_model(),
            'distance_km': distance_km,
            'estimated_time_minutes': time_minutes
        }
    
    def save_model(self, model_path: str):
        """Save trained models and preprocessing objects"""
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'encoders': self.encoders,
            'feature_importance': self.feature_importance,
            'model_performance': self.model_performance,
            'feature_columns': self.feature_columns,
            'cluster_model': self.cluster_model
        }
        
        joblib.dump(model_data, model_path)
        self.logger.info(f"Models saved to {model_path}")
    
    def load_model(self, model_path: str):
        """Load trained models and preprocessing objects"""
        model_data = joblib.load(model_path)
        
        self.models = model_data['models']
        self.scalers = model_data['scalers']
        self.encoders = model_data['encoders']
        self.feature_importance = model_data['feature_importance']
        self.model_performance = model_data['model_performance']
        self.feature_columns = model_data.get('feature_columns', None)
        self.cluster_model = model_data.get('cluster_model', None)
        
        self.logger.info(f"Models loaded from {model_path}")
    
    def get_feature_importance_summary(self) -> pd.DataFrame:
        """Get feature importance summary for tree-based models"""
        importance_data = []
        
        for model_name, importances in self.feature_importance.items():
            for feature, importance in importances.items():
                importance_data.append({
                    'model': model_name,
                    'feature': feature,
                    'importance': importance
                })
        
        return pd.DataFrame(importance_data)

def main():
    """Main training pipeline"""
    
    # Initialize predictor
    predictor = EVConsumptionPredictor()
    
    # Load and prepare data
    print("Loading data...")
    data = predictor.load_data()
    
    # Engineer features
    print("Engineering features...")
    data = predictor.engineer_features(data)
    
    # Prepare features and target
    X, y = predictor.prepare_features(data)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Further split training into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train models
    print("\nTraining models...")
    predictor.train_models(X_train, y_train, X_val, y_val)
    
    # Test best model
    best_model = predictor.get_best_model()
    print(f"\nBest model: {best_model}")
    
    test_predictions = predictor.predict(X_test, best_model)
    test_mae = mean_absolute_error(y_test, test_predictions)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    test_r2 = r2_score(y_test, test_predictions)
    
    print(f"Test performance - MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, R2: {test_r2:.4f}")
    
    # Save model
    model_path = Path(__file__).parent / "consumption_model.pkl"
    predictor.save_model(model_path)
    
    # Example prediction
    print("\nExample prediction:")
    example_prediction = predictor.predict_trip_consumption(
        distance_km=50.0,
        time_minutes=60.0,
        temperature=20.0,
        vehicle_model='tesla_model_3',
        driver_profile='commuter',
        season='winter'
    )
    
    print(f"50km trip in 60 minutes at 20°C:")
    print(f"Predicted consumption: {example_prediction['predicted_consumption_kwh']:.2f} kWh")
    print(f"Efficiency: {example_prediction['efficiency_kwh_per_100km']:.2f} kWh/100km")

if __name__ == "__main__":
    main()
