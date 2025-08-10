import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import logging
from pathlib import Path
from typing import Tuple, Dict, Any
import warnings
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import RandomizedSearchCV
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.utils import parallel_backend

warnings.filterwarnings('ignore')
#from lightgbm import LGBMRegressor
#from catboost import CatBoostRegressor

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available. Install with: pip install lightgbm")

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: CatBoost not available. Install with: pip install catboost")
    
def robust_parse_datetime(series):
    parsed = pd.to_datetime(series, errors='coerce')
    mask_failed = parsed.isna()
    if mask_failed.any():
        parsed2 = pd.to_datetime(series[mask_failed], format='%Y-%m-%dT%H:%M:%S.%f', errors='coerce')
        parsed[mask_failed] = parsed2
    return parsed

class SegmentEnergyPredictor:
    """
    Segment-level energy (kWh) prediction for EV fleet optimization.
    Only uses features available before the segment is driven (no look-ahead bias).
    """
    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or Path(__file__).parent.parent.parent.parent / "data" / "synthetic"
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = {}
        self.model_performance = {}
        self.feature_columns = None
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def load_data(self) -> pd.DataFrame:
        """Load and merge segment, vehicle, and weather data."""
        segments = pd.read_csv(self.data_dir / "segments.csv")
        fleet = pd.read_csv(self.data_dir / "fleet_info.csv")
        weather = pd.read_csv(self.data_dir / "weather.csv")
        # Merge with vehicle info
        data = segments.merge(fleet, on="vehicle_id", how="left", suffixes=("", "_veh"))
        # Merge with weather info (on date)
        data = data.merge(weather, left_on="date", right_on="date", how="left", suffixes=("", "_wx"))
        self.logger.info(f"Loaded {len(data)} segment records.")
        return data

    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Professional, detailed feature engineering for segment-level prediction."""
        # --- Spatial features ---
        data['distance_km'] = data['distance_m'] / 1000
        # --- Distance bins ---
        
        bins = [0, 50, 150, 400, 1200, np.inf]
        labels = ['very_low', 'low', 'medium', 'high', 'very_high']
        data['distance_bin'] = pd.cut(data['distance_m'], bins=bins, labels=labels, right=False, include_lowest=True)
        
        # Optionally encode as integer for ML models
        data['distance_bin_encoded'] = data['distance_bin'].map({l: i for i, l in enumerate(labels)}).astype(int)
    
        
        data['elevation_gain'] = data['end_elevation_m'] - data['start_elevation_m']
        # --- Temporal features ---
        
        # Parse ISO 8601 timestamps robustly
        # Remove 'format' to allow parsing both with and without microseconds
        data['start_time'] = robust_parse_datetime(data['start_time'])
        data['end_time'] = robust_parse_datetime(data['end_time'])
        # After this fix, check that most start_time values are valid timestamps (not NaT)
        # and that hour/weekday features show a realistic spread (not just a few values)
        data['duration_s'] = (data['end_time'] - data['start_time']).dt.total_seconds()
        # Only extract hour/weekday if start_time is valid
        data['hour'] = data['start_time'].dt.hour
        data['weekday'] = data['start_time'].dt.dayofweek
        data['is_weekend'] = (data['weekday'] >= 5).astype(int)
        print("Unique dates in the data:", data['date'].unique())
        print("Date value counts:\n", data['date'].value_counts())
        #print(data['start_time'].head(10))  # See if values are unique and parsed correctly
        #print(data['hour'].value_counts())  # Should show a spread, not just one value
        #print(data['weekday'].value_counts())
        # --- Speed features ---
        data['avg_speed_kmh'] = np.where(data['duration_s'] > 0, data['distance_km'] / (data['duration_s'] / 3600), 0)
        data['speed_delta'] = data['end_speed_kmh'] - data['start_speed_kmh']
        # --- Trip context ---
        data['segments_per_trip'] = data.groupby('trip_id')['segment_id'].transform('count')
        data['segment_idx_in_trip'] = data.groupby('trip_id').cumcount()
        # --- Energy efficiency ---
        data['energy_per_meter'] = np.where(data['distance_m'] > 0, data['energy_kwh'] * 1000 / data['distance_m'], 0)
        data['efficiency_wh_per_km'] = np.where(data['distance_km'] > 0, data['energy_kwh'] * 1000 / data['distance_km'], 0)
        # --- Weather features (already merged) ---
        # temperature, wind_speed_kmh, humidity, is_raining, season
        # --- Vehicle/driver features (already merged) ---
        # model, battery_capacity, efficiency, max_charging_speed, driver_profile, driving_style, driver_personality, has_home_charging
        # --- Categorical encoding ---
        label_encode_cols = ['model', 'driver_profile', 'driver_personality', 'season']
        for col in label_encode_cols:
            if col in data.columns:
                self.encoders[col] = LabelEncoder()
                data[col] = self.encoders[col].fit_transform(data[col].astype(str))
        # --- Drop columns not available before the segment (look-ahead bias) ---
        driving_style_order = [['eco_friendly', 'normal', 'aggressive']]

        ordinal_encoder = OrdinalEncoder(categories=driving_style_order)
        data['driving_style_encoded'] = ordinal_encoder.fit_transform(data[['driving_style']])
        drop_cols = [
            'end_lat','end_lon',
            'energy_rolling_resistance_kwh', 'energy_aerodynamic_drag_kwh',
            'start_speed_kmh','end_speed_kmh',
            'distance_km',
            'duration_s','is_weekend','avg_speed_kmh','speed_delta','segments_per_trip','segment_idx_in_trip','energy_per_meter','efficiency_wh_per_km',
            'energy_elevation_change_kwh', 'energy_acceleration_kwh', 'energy_regenerative_braking_kwh',
            'energy_hvac_kwh', 'energy_auxiliary_kwh', 'energy_battery_thermal_loss_kwh',
            'end_time', 'end_elevation_m', 'segment_id', 'trip_id', 'date','max_charging_speed','battery_capacity','preferred_start_hour','schedule_variability',
            'current_battery_soc', 'odometer', 'last_service', 'last_home_visit', 'home_location' , 'end_speed_kmh'
        ]
        # Only drop if present
        data = data.drop(columns=[c for c in drop_cols if c in data.columns], errors='ignore')
        # --- Handle missing values ---
        data = data.fillna(data.median(numeric_only=True))
        # --- Derived features ---
        data['log_distance_m'] = np.log1p(data['distance_m'])
        if 'weather_temp_c' in data.columns:
            data['temp_squared'] = (data['weather_temp_c'] - 15) ** 2
        # --- Energy per km target ---
        data['energy_per_km'] = data['energy_kwh'] / (data['distance_m'] / 1000 + 1e-6)
        return data

    def prepare_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target variable."""
        # Target: log1p(energy_per_km)
        target = np.log1p(data['energy_per_km'])  # Log-transform target
        # Features: all except target and columns not available before segment
        feature_cols = [col for col in data.columns if col not in ['energy_kwh', 'energy_per_km', 'distance_m']]
        features = data[feature_cols].copy()
        # Drop datetime columns
        features = features.select_dtypes(exclude=['datetime64[ns]', 'datetime64[ns, UTC]'])
        # Drop object/string columns (e.g., IDs)
        features = features.select_dtypes(include=[np.number])
        self.feature_columns = features.columns.tolist()
        # Print shape and save features/target for transparency
        print(f"Features shape: {features.shape}")
        print(f"Target shape: {target.shape}")
        print(features.columns)
        features.to_csv(Path(__file__).parent / 'segment_model_features.csv', index=False)
        target.to_csv(Path(__file__).parent / 'segment_model_target.csv', index=False)
        
        return features, target

    def smape(self, y_true, y_pred):
        return np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100

    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series):
        """Train multiple models and compare performance."""
        self.scalers['standard'] = StandardScaler()
        X_train_scaled = self.scalers['standard'].fit_transform(X_train)
        X_val_scaled = self.scalers['standard'].transform(X_val)
        models_config = {
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=100, max_depth=6, min_samples_split=5, min_samples_leaf=4, random_state=42, n_jobs=-1),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, max_depth=6,learning_rate=0.1, random_state=42)
        }
        if XGBOOST_AVAILABLE:
            models_config['xgboost'] = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='mae')
        
        if LIGHTGBM_AVAILABLE:
            models_config['lightgbm'] = LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
        if CATBOOST_AVAILABLE:
            models_config['catboost'] = CatBoostRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, verbose=0)
            
        for name, model in models_config.items():
            self.logger.info(f"Training {name}...")
            if name == 'linear_regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_val_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred)
            mse = mean_squared_error(y_val, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_val, y_pred)
            mape = mean_absolute_percentage_error(y_val, y_pred)
            smape = self.smape(y_val, y_pred)
            self.models[name] = model
            self.model_performance[name] = {
                'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2, 'MAPE': mape, 'SMAPE': smape
            }
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = dict(zip(X_train.columns, model.feature_importances_))
            self.logger.info(f"{name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}, MAPE: {mape:.4f}, SMAPE: {smape:.4f}")

    def get_best_model(self) -> str:
        """Select best model based on MAE."""
        best_model = min(self.model_performance.keys(), key=lambda x: self.model_performance[x].get('MAE', float('inf')))
        return best_model
    
    
    def tune_xgboost(self, X_train, y_train):
        """Hyperparameter tuning for XGBoost using RandomizedSearchCV."""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed.")
        param_dist = {
            'n_estimators': [100, 200, 400],
            'max_depth': [4, 6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.5, 0.7, 1.0],
            'min_child_weight': [1, 5, 10]
        }
        xgb_model = xgb.XGBRegressor(random_state=42, eval_metric='rmse')
        rs = RandomizedSearchCV(
            xgb_model,
            param_distributions=param_dist,
            n_iter=30,
            cv=3,
            scoring='neg_root_mean_squared_error',
            verbose=2,
            n_jobs=-1
        )
        rs.fit(X_train, y_train)
        self.models['xgboost_tuned'] = rs.best_estimator_
        self.logger.info(f"Best XGBoost params: {rs.best_params_}")
        return rs.best_estimator_
    
    def tune_xgboost_optuna(self, X_train, y_train, n_trials=30, timeout=600):
        """Hyperparameter tuning for XGBoost using Optuna."""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed.")

        def objective(trial):
            param = {
                'n_estimators': trial.suggest_categorical('n_estimators', [100, 200, 400]),
                'max_depth': trial.suggest_int('max_depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'random_state': 42,
                'eval_metric': 'mae',
                'tree_method': 'hist'
            }
            model = xgb.XGBRegressor(**param)
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
            return -scores.mean()

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
        best_params = study.best_trial.params
        self.logger.info(f"Best Optuna XGBoost params: {best_params}")
        best_model = xgb.XGBRegressor(**best_params)
        best_model.fit(X_train, y_train)
        self.models['xgboost_optuna'] = best_model
        return best_model

    
    
    
    

    def tune_random_forest(self, X_train, y_train):
        param_dist = {
            'n_estimators': [100, 200, 400],
            'max_depth': [4, 6, 8, 12, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2']
        }
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        rs = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=30, cv=3,
                                scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=0)
        with parallel_backend('threading'):
            rs.fit(X_train, y_train)
        self.models['random_forest_tuned'] = rs.best_estimator_
        self.logger.info(f"Best Random Forest params: {rs.best_params_}")
        return rs.best_estimator_

    def tune_gradient_boosting(self, X_train, y_train):
        param_dist = {
            'n_estimators': [100, 200, 400],
            'max_depth': [3, 5, 7, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2']
        }
        gb = GradientBoostingRegressor(random_state=42)
        rs = RandomizedSearchCV(gb, param_distributions=param_dist, n_iter=30, cv=3,
                                scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=0)
        with parallel_backend('threading'):
            rs.fit(X_train, y_train)
        self.models['gradient_boosting_tuned'] = rs.best_estimator_
        self.logger.info(f"Best Gradient Boosting params: {rs.best_params_}")
        return rs.best_estimator_

    def tune_lightgbm(self, X_train, y_train):
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not installed.")
        param_dist = {
            'verbosity': [-1],
            'n_estimators': [100, 200, 400],
            'max_depth': [4, 6, 8, -1],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [15, 31, 63, 127],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [0, 0.1, 1]
        }
        lgbm = LGBMRegressor(random_state=42, n_jobs=-1)
        rs = RandomizedSearchCV(lgbm, param_distributions=param_dist, n_iter=30, cv=3,
                                scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=0)
        with parallel_backend('threading'):
            rs.fit(X_train, y_train)
        self.models['lightgbm_tuned'] = rs.best_estimator_
        self.logger.info(f"Best LightGBM params: {rs.best_params_}")
        return rs.best_estimator_

    def tune_catboost(self, X_train, y_train):
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost is not installed.")
        param_dist = {
            'iterations': [100, 200, 400],
            'depth': [4, 6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1],
            'l2_leaf_reg': [1, 3, 5, 7, 9],
            'bagging_temperature': [0, 0.5, 1],
            'random_strength': [1, 2, 5, 10]
        }
        cb = CatBoostRegressor(random_state=42, verbose=0)
        rs = RandomizedSearchCV(cb, param_distributions=param_dist, n_iter=30, cv=3,
                                scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=0)
        with parallel_backend('threading'):
            rs.fit(X_train, y_train)
        self.models['catboost_tuned'] = rs.best_estimator_
        self.logger.info(f"Best CatBoost params: {rs.best_params_}")
        return rs.best_estimator_
    
    def print_model_metrics(self,model, X_val, y_val, name):
        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        mse = mean_squared_error(y_val, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val, preds)
        mape = mean_absolute_percentage_error(y_val, preds)
        smape = self.smape(y_val, preds)
        self.model_performance[name] = {
        'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2, 'MAPE': mape, 'SMAPE': smape}
        print(f"{name} tuned - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}, MAPE: {mape:.4f}, SMAPE: {smape:.4f}")
    
    
    
    def predict(self, features: pd.DataFrame, model_name: str = None, distance_m: np.ndarray = None) -> np.ndarray:
        if model_name is None:
            model_name = self.get_best_model()
        model = self.models[model_name]
        if model_name == 'linear_regression':
            features_scaled = self.scalers['standard'].transform(features)
            preds_log = model.predict(features_scaled)
        else:
            preds_log = model.predict(features)
        pred_energy_per_km = np.expm1(preds_log)
        # If distance_m is not provided, try to get from features (but we expect it to be passed explicitly)
        if distance_m is None:
            raise ValueError('distance_m must be provided for post-prediction scaling')
        pred_energy_kwh = pred_energy_per_km * (distance_m / 1000)
        return pred_energy_kwh

    def save_model(self, model_path: str):
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'encoders': self.encoders,
            'feature_importance': self.feature_importance,
            'model_performance': self.model_performance,
            'feature_columns': self.feature_columns
        }
        joblib.dump(model_data, model_path)
        self.logger.info(f"Models saved to {model_path}")

    def load_model(self, model_path: str):
        model_data = joblib.load(model_path)
        self.models = model_data['models']
        self.scalers = model_data['scalers']
        self.encoders = model_data['encoders']
        self.feature_importance = model_data['feature_importance']
        self.model_performance = model_data['model_performance']
        self.feature_columns = model_data.get('feature_columns', None)
        self.logger.info(f"Models loaded from {model_path}")

    def explain_with_shap(self, model_name: str, X_sample: pd.DataFrame):
        """Explain model predictions using SHAP values and plot beeswarm."""
        import shap
        model = self.models[model_name]
        # For linear regression, use scaled features
        if model_name == 'linear_regression':
            X_sample_ = self.scalers['standard'].transform(X_sample)
        else:
            X_sample_ = X_sample
        explainer = shap.Explainer(model, X_sample_)
        shap_values = explainer(X_sample_,check_additivity=False)
        shap.plots.beeswarm(shap_values, show=True)

def main():
    predictor = SegmentEnergyPredictor()
    print("Loading data...")
    data = predictor.load_data()
    print(data['distance_m'].describe(percentiles=[.1, .25, .5, .75, .9, .95, .99]))
    print("Engineering features...")
    data = predictor.engineer_features(data)
    print("Preparing features and target...")
    X, y = predictor.prepare_features(data)
    print("main")
    print(X['hour'].value_counts(dropna=False))
    print(X['weekday'].value_counts(dropna=False))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    print("Training models...")
    predictor.train_models(X_train, y_train, X_val, y_val)
    # --- Hyperparameter tuning for XGBoost with Optuna ---
    if XGBOOST_AVAILABLE:
        print("Tuning XGBoost hyperparameters with Optuna...")
        xgb_model=predictor.tune_xgboost_optuna(X_train, y_train, n_trials=200  , timeout=600)
        predictor.print_model_metrics(xgb_model, X_val, y_val, "xgboost_optuna")
        
    print("Tuning Random Forest...")
    rf_model=predictor.tune_random_forest(X_train, y_train)
    predictor.print_model_metrics(rf_model, X_val, y_val, "Random Forest")
    print("Tuning Gradient Boosting...")
    """
    gb_model=predictor.tune_gradient_boosting(X_train, y_train)
    predictor.print_model_metrics(gb_model, X_val, y_val, "Gradient Boosting")
    """

    if LIGHTGBM_AVAILABLE:
        print("Tuning LightGBM...")
        lgbm_model=predictor.tune_lightgbm(X_train, y_train)
        predictor.print_model_metrics(lgbm_model, X_val, y_val, "LightGBM")

    if CATBOOST_AVAILABLE:
        print("Tuning CatBoost...")
        cb_model=predictor.tune_catboost(X_train, y_train)
        predictor.print_model_metrics(cb_model, X_val, y_val, "CatBoost")
        
    best_model = predictor.get_best_model()
    print(f"Best model: {best_model}")
    # SHAP feature analysis on validation set
    predictor.explain_with_shap(best_model, X_val)
    # For test set, get distance_m for each sample from the original data
    test_distances = data.loc[X_test.index, 'distance_m'].values if 'distance_m' in data.columns else None
    test_predictions = predictor.predict(X_test, best_model, distance_m=test_distances)
    y_test_exp = data.loc[X_test.index, 'energy_kwh'].values if 'energy_kwh' in data.columns else np.nan
    test_mae = mean_absolute_error(y_test_exp, test_predictions)
    test_mse = mean_squared_error(y_test_exp, test_predictions)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test_exp, test_predictions)
    test_mape = mean_absolute_percentage_error(y_test_exp, test_predictions)
    test_smape = predictor.smape(y_test_exp, test_predictions)
    print(f"Test performance - MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, R2: {test_r2:.4f}, MAPE: {test_mape:.4f}, SMAPE: {test_smape:.4f}")
    model_path = Path(__file__).parent / "segment_energy_model.pkl"
    predictor.save_model(model_path)

if __name__ == "__main__":
    main()
