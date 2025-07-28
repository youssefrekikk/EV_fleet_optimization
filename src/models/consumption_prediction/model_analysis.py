"""
Comprehensive XGBoost Model Analysis
=====================================

This script provides a thorough evaluation of the EV consumption prediction model
including overfitting analysis, feature importance, correlation analysis, and 
performance diagnostics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

from consumption_model import EVConsumptionPredictor

class ModelAnalyzer:
    """Comprehensive model analysis class"""
    
    def __init__(self, model_path: str = None):
        self.predictor = EVConsumptionPredictor()
        self.model_path = model_path
        self.analysis_results = {}
        
    def load_and_prepare_data(self):
        """Load data and prepare for analysis"""
        print("Loading and preparing data for analysis...")
        
        # Load data
        data = self.predictor.load_data()
        data = self.predictor.engineer_features(data)
        
        # Prepare features and target
        X, y = self.predictor.prepare_features(data)
        
        # Split data (same as training)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.25, random_state=42
        )
        
        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test
        self.feature_names = X.columns.tolist()
        
        print(f"Data loaded: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test samples")
        return X, y
    
    def analyze_overfitting(self):
        """Comprehensive overfitting analysis"""
        print("\n" + "="*60)
        print("OVERFITTING ANALYSIS")
        print("="*60)
        
        # Load model if not already loaded
        if self.model_path and 'xgboost' not in self.predictor.models:
            self.predictor.load_model(self.model_path)
        
        xgb_model = self.predictor.models.get('xgboost')
        if xgb_model is None:
            print("XGBoost model not found. Training models first...")
            self.predictor.train_models(self.X_train, self.y_train, self.X_val, self.y_val)
            xgb_model = self.predictor.models['xgboost']
        
        # 1. Performance on different sets
        train_pred = xgb_model.predict(self.X_train)
        val_pred = xgb_model.predict(self.X_val)
        test_pred = xgb_model.predict(self.X_test)
        
        # Calculate metrics for each set
        performance_data = []
        for name, y_true, y_pred in [
            ('Training', self.y_train, train_pred),
            ('Validation', self.y_val, val_pred),
            ('Test', self.y_test, test_pred)
        ]:
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            
            performance_data.append({
                'Dataset': name,
                'MAE': mae,
                'RMSE': rmse,
                'R²': r2,
                'MAPE': mape
            })
        
        perf_df = pd.DataFrame(performance_data)
        print("\nPerformance Comparison:")
        print(perf_df.round(4))
        
        # Overfitting indicators
        train_r2 = perf_df[perf_df['Dataset'] == 'Training']['R²'].iloc[0]
        val_r2 = perf_df[perf_df['Dataset'] == 'Validation']['R²'].iloc[0]
        test_r2 = perf_df[perf_df['Dataset'] == 'Test']['R²'].iloc[0]
        
        overfitting_gap = train_r2 - val_r2
        generalization_gap = val_r2 - test_r2
        
        print(f"\nOverfitting Analysis:")
        print(f"Train-Validation Gap: {overfitting_gap:.4f}")
        print(f"Validation-Test Gap: {generalization_gap:.4f}")
        
        if overfitting_gap > 0.02:
            print("WARNING: Potential overfitting detected (train-val gap > 0.02)")
        elif overfitting_gap > 0.01:
            print("CAUTION: Mild overfitting detected (train-val gap > 0.01)")
        else:
            print("Good: No significant overfitting detected")
        
        if abs(generalization_gap) > 0.01:
            print("WARNING: Generalization gap detected")
        else:
            print("Good: Model generalizes well to test set")
        
        # 2. Learning curves analysis
        self._plot_learning_curves(xgb_model)
        
        self.analysis_results['overfitting'] = {
            'performance': perf_df,
            'overfitting_gap': overfitting_gap,
            'generalization_gap': generalization_gap
        }
        
        return perf_df
    
    def _plot_learning_curves(self, model):
        """Plot learning curves to visualize overfitting"""
        print("\nGenerating learning curves...")
        
        # Create combined training data for learning curve
        X_combined = pd.concat([self.X_train, self.X_val])
        y_combined = pd.concat([self.y_train, self.y_val])
        
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_combined, y_combined, 
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=5, scoring='r2', n_jobs=-1
        )
        
        plt.figure(figsize=(12, 8))
        
        # Learning curve
        plt.subplot(2, 2, 1)
        plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training Score', color='blue')
        plt.plot(train_sizes, np.mean(val_scores, axis=1), 'o-', label='Validation Score', color='red')
        plt.fill_between(train_sizes, np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                         np.mean(train_scores, axis=1) + np.std(train_scores, axis=1), alpha=0.1, color='blue')
        plt.fill_between(train_sizes, np.mean(val_scores, axis=1) - np.std(val_scores, axis=1),
                         np.mean(val_scores, axis=1) + np.std(val_scores, axis=1), alpha=0.1, color='red')
        plt.xlabel('Training Set Size')
        plt.ylabel('R² Score')
        plt.title('Learning Curves - Overfitting Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Validation curve for max_depth
        plt.subplot(2, 2, 2)
        param_range = range(3, 11)
        train_scores, val_scores = validation_curve(
            type(model)(), X_combined, y_combined,
            param_name='max_depth', param_range=param_range,
            cv=3, scoring='r2'
        )
        
        plt.plot(param_range, np.mean(train_scores, axis=1), 'o-', label='Training', color='blue')
        plt.plot(param_range, np.mean(val_scores, axis=1), 'o-', label='Validation', color='red')
        plt.xlabel('Max Depth')
        plt.ylabel('R² Score')
        plt.title('Validation Curve - Max Depth')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Residuals plot
        plt.subplot(2, 2, 3)
        test_pred = model.predict(self.X_test)
        residuals = self.y_test - test_pred
        plt.scatter(test_pred, residuals, alpha=0.6, color='purple')
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals Plot - Test Set')
        plt.grid(True, alpha=0.3)
        
        # Q-Q plot for residuals normality
        plt.subplot(2, 2, 4)
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('Q-Q Plot - Residuals Normality')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('overfitting_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_feature_importance(self):
        """Comprehensive feature importance analysis"""
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        xgb_model = self.predictor.models['xgboost']
        
        # Get feature importance
        importance_gain = xgb_model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance_gain': importance_gain
        }).sort_values('importance_gain', ascending=False)
        
        # Top 20 features
        top_features = importance_df.head(20)
        
        print("Top 20 Most Important Features:")
        for i, (_, row) in enumerate(top_features.iterrows(), 1):
            print(f"{i:2d}. {row['feature']:30s} : {row['importance_gain']:.4f}")
        
        # Visualize feature importance
        plt.figure(figsize=(15, 10))
        
        # Plot top 15 features
        plt.subplot(2, 2, 1)
        top_15 = importance_df.head(15)
        plt.barh(range(len(top_15)), top_15['importance_gain'])
        plt.yticks(range(len(top_15)), top_15['feature'])
        plt.xlabel('Feature Importance (Gain)')
        plt.title('Top 15 Feature Importance - XGBoost')
        plt.gca().invert_yaxis()
        
        # Feature importance distribution
        plt.subplot(2, 2, 2)
        plt.hist(importance_gain, bins=30, alpha=0.7, color='skyblue')
        plt.xlabel('Feature Importance')
        plt.ylabel('Count')
        plt.title('Feature Importance Distribution')
        plt.axvline(np.mean(importance_gain), color='red', linestyle='--', label='Mean')
        plt.legend()
        
        # Cumulative importance
        plt.subplot(2, 2, 3)
        cumsum = np.cumsum(importance_df['importance_gain'].values)
        plt.plot(range(1, len(cumsum) + 1), cumsum / cumsum[-1])
        plt.xlabel('Number of Features')
        plt.ylabel('Cumulative Importance Ratio')
        plt.title('Cumulative Feature Importance')
        plt.grid(True, alpha=0.3)
        
        # Feature categories analysis
        plt.subplot(2, 2, 4)
        categories = self._categorize_features()
        cat_importance = {}
        for feature, importance in zip(self.feature_names, importance_gain):
            cat = self._get_feature_category(feature, categories)
            cat_importance[cat] = cat_importance.get(cat, 0) + importance
        
        cats, vals = zip(*sorted(cat_importance.items(), key=lambda x: x[1], reverse=True))
        plt.bar(cats, vals, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57'])
        plt.xlabel('Feature Category')
        plt.ylabel('Total Importance')
        plt.title('Importance by Feature Category')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('feature_importance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Analyze feature importance distribution
        total_importance = importance_gain.sum()
        top_5_importance = top_features.head(5)['importance_gain'].sum()
        top_10_importance = top_features.head(10)['importance_gain'].sum()
        
        print(f"\nFeature Importance Distribution:")
        print(f"Top 5 features account for: {top_5_importance/total_importance*100:.1f}% of total importance")
        print(f"Top 10 features account for: {top_10_importance/total_importance*100:.1f}% of total importance")
        
        if top_5_importance/total_importance > 0.8:
            print("WARNING: Model heavily relies on few features (>80% from top 5)")
        elif top_5_importance/total_importance > 0.6:
            print("CAUTION: Model moderately concentrated on top features")
        else:
            print("Good: Feature importance is well distributed")
        
        self.analysis_results['feature_importance'] = importance_df
        return importance_df
    
    def analyze_feature_correlations(self):
        """Analyze feature correlations and multicollinearity"""
        print("\n" + "="*60)
        print("FEATURE CORRELATION ANALYSIS")
        print("="*60)
        
        # Calculate correlation matrix
        corr_matrix = self.X_train.corr()
        
        # Find highly correlated feature pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.8:  # High correlation threshold
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_val
                    })
        
        if high_corr_pairs:
            print("High Correlation Pairs (|correlation| > 0.8):")
            for pair in sorted(high_corr_pairs, key=lambda x: abs(x['correlation']), reverse=True):
                print(f"  {pair['feature1']} <-> {pair['feature2']}: {pair['correlation']:.3f}")
        else:
            print("No highly correlated feature pairs found (threshold: 0.8)")
        
        # Correlation heatmap for top features
        top_features = self.analysis_results['feature_importance'].head(20)['feature'].values
        corr_subset = corr_matrix.loc[top_features, top_features]
        
        plt.figure(figsize=(14, 12))
        
        # Correlation heatmap
        plt.subplot(2, 2, 1)
        mask = np.triu(np.ones_like(corr_subset, dtype=bool))
        sns.heatmap(corr_subset, mask=mask, annot=True, cmap='coolwarm', center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": .8}, fmt='.2f')
        plt.title('Feature Correlation Heatmap (Top 20 Features)')
        
        # Correlation with target
        plt.subplot(2, 2, 2)
        target_corr = []
        for feature in self.X_train.columns:
            corr, _ = pearsonr(self.X_train[feature], self.y_train)
            target_corr.append(abs(corr))
        
        target_corr_df = pd.DataFrame({
            'feature': self.X_train.columns,
            'target_correlation': target_corr
        }).sort_values('target_correlation', ascending=False)
        
        top_target_corr = target_corr_df.head(15)
        plt.barh(range(len(top_target_corr)), top_target_corr['target_correlation'])
        plt.yticks(range(len(top_target_corr)), top_target_corr['feature'])
        plt.xlabel('|Correlation with Target|')
        plt.title('Top 15 Features - Target Correlation')
        plt.gca().invert_yaxis()
        
        # Variance Inflation Factor analysis
        plt.subplot(2, 2, 3)
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        
        # Calculate VIF for top 10 features (VIF calculation is expensive)
        top_10_features = top_features[:10]
        X_vif = self.X_train[top_10_features]
        
        vif_data = []
        for i, feature in enumerate(X_vif.columns):
            try:
                vif = variance_inflation_factor(X_vif.values, i)
                vif_data.append({'feature': feature, 'VIF': vif})
            except:
                vif_data.append({'feature': feature, 'VIF': np.nan})
        
        vif_df = pd.DataFrame(vif_data).sort_values('VIF', ascending=False)
        vif_df = vif_df.dropna()
        
        plt.bar(range(len(vif_df)), vif_df['VIF'])
        plt.xticks(range(len(vif_df)), vif_df['feature'], rotation=45)
        plt.ylabel('Variance Inflation Factor')
        plt.title('VIF Analysis (Top 10 Features)')
        plt.axhline(y=5, color='red', linestyle='--', label='VIF = 5 (concern)')
        plt.axhline(y=10, color='red', linestyle='-', label='VIF = 10 (problem)')
        plt.legend()
        
        # Feature redundancy analysis
        plt.subplot(2, 2, 4)
        redundant_features = []
        if high_corr_pairs:
            # Analyze which features in high correlation pairs are less important
            importance_dict = dict(zip(self.analysis_results['feature_importance']['feature'],
                                     self.analysis_results['feature_importance']['importance_gain']))
            
            for pair in high_corr_pairs:
                f1_imp = importance_dict.get(pair['feature1'], 0)
                f2_imp = importance_dict.get(pair['feature2'], 0)
                if f1_imp < f2_imp:
                    redundant_features.append(pair['feature1'])
                else:
                    redundant_features.append(pair['feature2'])
        
        if redundant_features:
            redundant_importance = [importance_dict.get(f, 0) for f in redundant_features]
            plt.bar(range(len(redundant_features)), redundant_importance)
            plt.xticks(range(len(redundant_features)), redundant_features, rotation=45)
            plt.ylabel('Feature Importance')
            plt.title('Potentially Redundant Features')
        else:
            plt.text(0.5, 0.5, 'No redundant features detected', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Redundancy Analysis')
        
        plt.tight_layout()
        plt.savefig('correlation_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Multicollinearity warnings
        high_vif_features = vif_df[vif_df['VIF'] > 10] if not vif_df.empty else pd.DataFrame()
        
        if not high_vif_features.empty:
            print(f"\nWARNING: {len(high_vif_features)} features with high VIF (>10):")
            for _, row in high_vif_features.iterrows():
                print(f"  {row['feature']}: VIF = {row['VIF']:.2f}")
        
        if len(high_corr_pairs) > 5:
            print(f"\nWARNING: {len(high_corr_pairs)} highly correlated feature pairs detected")
            print("  Consider feature selection or dimensionality reduction")
        
        self.analysis_results['correlations'] = {
            'high_corr_pairs': high_corr_pairs,
            'target_correlation': target_corr_df,
            'vif_analysis': vif_df
        }
        
        return high_corr_pairs, target_corr_df
    
    def analyze_prediction_errors(self):
        """Analyze prediction errors and identify patterns"""
        print("\n" + "="*60)
        print("PREDICTION ERROR ANALYSIS")
        print("="*60)
        
        xgb_model = self.predictor.models['xgboost']
        test_pred = xgb_model.predict(self.X_test)
        errors = self.y_test - test_pred
        
        # Error statistics
        print(f"Error Statistics:")
        print(f"  Mean Error: {errors.mean():.4f}")
        print(f"  Std Error: {errors.std():.4f}")
        print(f"  Min Error: {errors.min():.4f}")
        print(f"  Max Error: {errors.max():.4f}")
        print(f"  Median Error: {errors.median():.4f}")
        
        # Error analysis by prediction magnitude
        pred_bins = pd.qcut(test_pred, q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        error_by_magnitude = pd.DataFrame({
            'prediction_bin': pred_bins,
            'absolute_error': np.abs(errors),
            'relative_error': np.abs(errors / self.y_test) * 100
        })
        
        print(f"\nError by Prediction Magnitude:")
        magnitude_stats = error_by_magnitude.groupby('prediction_bin').agg({
            'absolute_error': ['mean', 'std'],
            'relative_error': ['mean', 'std']
        }).round(4)
        print(magnitude_stats)
        
        # Visualize error patterns
        plt.figure(figsize=(16, 12))
        
        # Error distribution
        plt.subplot(3, 3, 1)
        plt.hist(errors, bins=50, alpha=0.7, color='lightcoral')
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title('Error Distribution')
        plt.axvline(0, color='red', linestyle='--', alpha=0.8)
        
        # Absolute error vs prediction
        plt.subplot(3, 3, 2)
        plt.scatter(test_pred, np.abs(errors), alpha=0.6, color='lightblue')
        plt.xlabel('Predicted Value')
        plt.ylabel('Absolute Error')
        plt.title('Absolute Error vs Predicted Value')
        
        # Relative error vs prediction
        plt.subplot(3, 3, 3)
        relative_errors = np.abs(errors / self.y_test) * 100
        plt.scatter(test_pred, relative_errors, alpha=0.6, color='lightgreen')
        plt.xlabel('Predicted Value')
        plt.ylabel('Relative Error (%)')
        plt.title('Relative Error vs Predicted Value')
        
        # Error by feature values (top 3 important features)
        top_3_features = self.analysis_results['feature_importance'].head(3)['feature'].values
        
        for i, feature in enumerate(top_3_features):
            plt.subplot(3, 3, 4 + i)
            plt.scatter(self.X_test[feature], np.abs(errors), alpha=0.6)
            plt.xlabel(feature)
            plt.ylabel('Absolute Error')
            plt.title(f'Error vs {feature}')
        
        # Error by prediction bins
        plt.subplot(3, 3, 7)
        error_by_magnitude.boxplot(column='absolute_error', by='prediction_bin', ax=plt.gca())
        plt.title('Absolute Error by Prediction Magnitude')
        plt.suptitle('')  # Remove default title
        
        # Residuals vs fitted
        plt.subplot(3, 3, 8)
        plt.scatter(test_pred, errors, alpha=0.6, color='purple')
        plt.axhline(0, color='red', linestyle='--', alpha=0.8)
        plt.xlabel('Fitted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Fitted')
        
        # Scale-location plot
        plt.subplot(3, 3, 9)
        standardized_residuals = errors / errors.std()
        plt.scatter(test_pred, np.sqrt(np.abs(standardized_residuals)), alpha=0.6, color='orange')
        plt.xlabel('Fitted Values')
        plt.ylabel('√|Standardized Residuals|')
        plt.title('Scale-Location Plot')
        
        plt.tight_layout()
        plt.savefig('error_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Outlier analysis
        z_scores = np.abs(stats.zscore(errors))
        outliers = z_scores > 3
        outlier_count = outliers.sum()
        
        print(f"\nOutlier Analysis:")
        print(f"  Outliers (|z-score| > 3): {outlier_count} ({outlier_count/len(errors)*100:.1f}%)")
        
        if outlier_count > len(errors) * 0.05:  # More than 5% outliers
            print("WARNING: High number of outliers detected")
        else:
            print("Outlier rate is acceptable")
        
        # Check for bias
        from scipy.stats import ttest_1samp
        t_stat, p_value = ttest_1samp(errors, 0)
        
        print(f"\nBias Analysis:")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.6f}")
        
        if p_value < 0.05:
            print("WARNING: Significant bias detected in predictions")
        else:
            print("No significant bias detected")
        
        self.analysis_results['errors'] = {
            'error_stats': errors.describe(),
            'outlier_count': outlier_count,
            'bias_test': {'t_stat': t_stat, 'p_value': p_value}
        }
        
        return errors
    
    def _categorize_features(self):
        """Categorize features by type"""
        return {
            'trip': ['total_distance_km', 'total_time_minutes', 'avg_speed_kmh', 'distance_per_minute'],
            'temporal': ['hour', 'day_of_week', 'month', 'is_weekend', 'is_rush_hour'],
            'weather': ['temperature_celsius', 'temp_squared', 'temp_deviation', 'weather_wind_speed_kmh', 
                       'weather_humidity', 'wind_impact', 'rain_impact', 'weather_is_raining'],
            'vehicle': ['battery_capacity', 'efficiency', 'max_charging_speed', 'fast_charging_capable',
                       'home_charging_available'],
            'location': ['origin_cluster', 'destination_cluster', 'same_cluster', 'route_efficiency',
                        'straight_line_distance_km'],
            'historical': ['vehicle_avg_consumption_7d', 'vehicle_avg_efficiency_7d', 'driver_avg_consumption_7d',
                          'trips_last_7d', 'vehicle_total_distance_7d']
        }
    
    def _get_feature_category(self, feature, categories):
        """Get category for a feature"""
        for category, features in categories.items():
            if feature in features:
                return category
        return 'other'
    
    def generate_recommendations(self):
        """Generate recommendations based on analysis"""
        print("\n" + "="*60)
        print("MODEL IMPROVEMENT RECOMMENDATIONS")
        print("="*60)
        
        recommendations = []
        
        # Overfitting recommendations
        overfitting_gap = self.analysis_results['overfitting']['overfitting_gap']
        if overfitting_gap > 0.02:
            recommendations.append("- Reduce model complexity (lower max_depth, increase min_child_weight)")
            recommendations.append("- Add more regularization (increase reg_alpha, reg_lambda)")
            recommendations.append("- Use early stopping with more patience")
        
        # Feature importance recommendations
        importance_df = self.analysis_results['feature_importance']
        top_5_ratio = importance_df.head(5)['importance_gain'].sum() / importance_df['importance_gain'].sum()
        
        if top_5_ratio > 0.8:
            recommendations.append("- Consider feature selection - model relies heavily on few features")
            recommendations.append("- Investigate if top features have data leakage")
        
        # Correlation recommendations
        high_corr_pairs = self.analysis_results['correlations']['high_corr_pairs']
        if len(high_corr_pairs) > 5:
            recommendations.append("- Remove highly correlated features to reduce multicollinearity")
            recommendations.append("- Consider PCA or feature engineering to combine correlated features")
        
        # Error pattern recommendations
        error_stats = self.analysis_results['errors']['error_stats']
        outlier_count = self.analysis_results['errors']['outlier_count']
        bias_p_value = self.analysis_results['errors']['bias_test']['p_value']
        
        if outlier_count > len(self.y_test) * 0.05:
            recommendations.append("- Investigate and handle outliers in training data")
            recommendations.append("- Consider robust scaling or outlier removal")
        
        if bias_p_value < 0.05:
            recommendations.append("- Address systematic bias in predictions")
            recommendations.append("- Check for missing important features")
        
        # Data recommendations
        if len(self.X_train) < 10000:
            recommendations.append("- Collect more training data if possible")
        
        # Model ensemble recommendations
        recommendations.append("- Consider ensemble methods (combine XGBoost with other models)")
        recommendations.append("- Try hyperparameter tuning (Bayesian optimization)")
        
        if recommendations:
            print("Based on the analysis, here are improvement recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"{i:2d}. {rec}")
        else:
            print("Model appears to be well-tuned. No major issues detected.")
        
        return recommendations
    
    def run_full_analysis(self):
        """Run complete model analysis"""
        print("COMPREHENSIVE XGBOOST MODEL ANALYSIS")
        print("="*80)
        
        # Load data
        self.load_and_prepare_data()
        
        # Run all analyses
        self.analyze_overfitting()
        self.analyze_feature_importance()
        self.analyze_feature_correlations()
        self.analyze_prediction_errors()
        
        # Generate recommendations
        self.generate_recommendations()
        
        # Summary
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        
        perf = self.analysis_results['overfitting']['performance']
        test_r2 = perf[perf['Dataset'] == 'Test']['R²'].iloc[0]
        
        print(f"Model Performance: R² = {test_r2:.4f}")
        print(f"Overfitting Gap: {self.analysis_results['overfitting']['overfitting_gap']:.4f}")
        print(f"High Correlation Pairs: {len(self.analysis_results['correlations']['high_corr_pairs'])}")
        print(f"Outlier Rate: {self.analysis_results['errors']['outlier_count']/len(self.y_test)*100:.1f}%")
        
        print(f"\nAnalysis visualizations saved:")
        print(f"   - overfitting_analysis.png")
        print(f"   - feature_importance_analysis.png") 
        print(f"   - correlation_analysis.png")
        print(f"   - error_analysis.png")
        
        return self.analysis_results

def main():
    """Main analysis execution"""
    analyzer = ModelAnalyzer()
    results = analyzer.run_full_analysis()
    return results

if __name__ == "__main__":
    main()
