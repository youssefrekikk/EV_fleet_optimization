"""
Visualization script for XGBoost model analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from consumption_model import EVConsumptionPredictor
import warnings
warnings.filterwarnings('ignore')

def create_analysis_visualizations():
    """Create comprehensive visualizations for model analysis"""
    
    # Load predictor and data
    predictor = EVConsumptionPredictor()
    data = predictor.load_data()
    data = predictor.engineer_features(data)
    X, y = predictor.prepare_features(data)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    
    # Train models
    predictor.train_models(X_train, y_train, X_val, y_val)
    xgb_model = predictor.models['xgboost']
    
    # Make predictions
    train_pred = xgb_model.predict(X_train)
    val_pred = xgb_model.predict(X_val)
    test_pred = xgb_model.predict(X_test)
    
    # =============================================================================
    # VISUALIZATION 1: Model Performance Overview
    # =============================================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Performance comparison
    datasets = ['Training', 'Validation', 'Test']
    r2_scores = [r2_score(y_train, train_pred), r2_score(y_val, val_pred), r2_score(y_test, test_pred)]
    rmse_scores = [np.sqrt(mean_squared_error(y_train, train_pred)), 
                   np.sqrt(mean_squared_error(y_val, val_pred)), 
                   np.sqrt(mean_squared_error(y_test, test_pred))]
    
    x = np.arange(len(datasets))
    width = 0.35
    
    ax1 = axes[0, 0]
    bars1 = ax1.bar(x - width/2, r2_scores, width, label='R² Score', color='skyblue')
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, rmse_scores, width, label='RMSE', color='lightcoral')
    
    ax1.set_xlabel('Dataset')
    ax1.set_ylabel('R² Score', color='skyblue')
    ax2.set_ylabel('RMSE (kWh)', color='lightcoral')
    ax1.set_title('Model Performance Across Datasets')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets)
    
    # Add value labels
    for bar, val in zip(bars1, r2_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', va='bottom')
    for bar, val in zip(bars2, rmse_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom')
    
    # 2. Feature importance
    importance_gain = xgb_model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': importance_gain
    }).sort_values('importance', ascending=False)
    
    top_15 = importance_df.head(15)
    axes[0, 1].barh(range(len(top_15)), top_15['importance'], color='lightgreen')
    axes[0, 1].set_yticks(range(len(top_15)))
    axes[0, 1].set_yticklabels(top_15['feature'])
    axes[0, 1].set_xlabel('Feature Importance')
    axes[0, 1].set_title('Top 15 Feature Importance')
    axes[0, 1].invert_yaxis()
    
    # 3. Predictions vs Actual (Test Set)
    axes[0, 2].scatter(y_test, test_pred, alpha=0.6, color='purple')
    axes[0, 2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 2].set_xlabel('Actual Consumption (kWh)')
    axes[0, 2].set_ylabel('Predicted Consumption (kWh)')
    axes[0, 2].set_title('Predictions vs Actual (Test Set)')
    axes[0, 2].grid(alpha=0.3)
    
    # Add R² annotation
    test_r2 = r2_score(y_test, test_pred)
    axes[0, 2].text(0.05, 0.95, f'R² = {test_r2:.4f}', transform=axes[0, 2].transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 4. Residuals plot
    errors = y_test - test_pred
    axes[1, 0].scatter(test_pred, errors, alpha=0.6, color='orange')
    axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.8)
    axes[1, 0].set_xlabel('Predicted Values')
    axes[1, 0].set_ylabel('Residuals')
    axes[1, 0].set_title('Residuals Plot')
    axes[1, 0].grid(alpha=0.3)
    
    # 5. Error distribution
    axes[1, 1].hist(errors, bins=50, alpha=0.7, color='lightcoral')
    axes[1, 1].axvline(0, color='red', linestyle='--', alpha=0.8)
    axes[1, 1].set_xlabel('Prediction Error (kWh)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Error Distribution')
    
    # Add statistics
    mean_error = errors.mean()
    std_error = errors.std()
    axes[1, 1].text(0.05, 0.95, f'Mean: {mean_error:.4f}\nStd: {std_error:.4f}', 
                    transform=axes[1, 1].transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 6. Feature importance concentration
    cumsum = np.cumsum(importance_df['importance'].values)
    axes[1, 2].plot(range(1, len(cumsum) + 1), cumsum / cumsum[-1], marker='o')
    axes[1, 2].axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='80%')
    axes[1, 2].axhline(y=0.95, color='orange', linestyle='--', alpha=0.7, label='95%')
    axes[1, 2].set_xlabel('Number of Features')
    axes[1, 2].set_ylabel('Cumulative Importance Ratio')
    axes[1, 2].set_title('Cumulative Feature Importance')
    axes[1, 2].legend()
    axes[1, 2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('xgboost_analysis_overview.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # =============================================================================
    # VISUALIZATION 2: Correlation Analysis
    # =============================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Correlation heatmap (top 15 features)
    top_15_features = importance_df.head(15)['feature'].values
    corr_matrix = X_train[top_15_features].corr()
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8}, fmt='.2f', ax=axes[0, 0])
    axes[0, 0].set_title('Feature Correlation Heatmap (Top 15)')
    
    # 2. Target correlation
    target_corr = []
    for feature in top_15_features:
        corr = np.corrcoef(X_train[feature], y_train)[0, 1]
        target_corr.append(abs(corr))
    
    axes[0, 1].barh(range(len(top_15_features)), target_corr, color='lightblue')
    axes[0, 1].set_yticks(range(len(top_15_features)))
    axes[0, 1].set_yticklabels(top_15_features)
    axes[0, 1].set_xlabel('|Correlation with Target|')
    axes[0, 1].set_title('Target Correlation (Top 15 Features)')
    axes[0, 1].invert_yaxis()
    
    # 3. Feature importance by category
    categories = {
        'Trip': ['total_distance_km', 'total_time_minutes', 'avg_speed_kmh', 'distance_per_minute'],
        'Weather': ['temperature_celsius', 'temp_deviation', 'weather_wind_speed_kmh', 
                   'weather_humidity', 'wind_impact', 'rain_impact', 'weather_is_raining'],
        'Vehicle': ['battery_capacity', 'efficiency', 'max_charging_speed', 'model'],
        'Location': ['origin_cluster', 'destination_cluster', 'same_cluster', 'route_efficiency',
                    'straight_line_distance_km'],
        'Historical': ['vehicle_avg_consumption_7d', 'vehicle_avg_efficiency_7d', 
                      'driver_avg_consumption_7d', 'trips_last_7d'],
        'Driver': ['driver_profile', 'driving_style', 'driver_personality', 'trip_type']
    }
    
    cat_importance = {}
    for cat, features in categories.items():
        total_imp = 0
        for feature in features:
            if feature in importance_df['feature'].values:
                idx = importance_df[importance_df['feature'] == feature].index[0]
                total_imp += importance_df.iloc[idx]['importance']
        cat_importance[cat] = total_imp
    
    cats, vals = zip(*sorted(cat_importance.items(), key=lambda x: x[1], reverse=True))
    axes[1, 0].bar(cats, vals, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3'])
    axes[1, 0].set_xlabel('Feature Category')
    axes[1, 0].set_ylabel('Total Importance')
    axes[1, 0].set_title('Importance by Feature Category')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Error analysis by top feature
    top_feature = importance_df.iloc[0]['feature']
    feature_values = X_test[top_feature]
    
    # Bin the feature values
    try:
        feature_bins = pd.qcut(feature_values, q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    except ValueError:
        # Handle case where there are duplicate bin edges
        feature_bins = pd.qcut(feature_values, q=5, duplicates='drop')
        # Create appropriate labels based on actual number of bins
        n_bins = len(feature_bins.cat.categories)
        if n_bins == 5:
            feature_bins = feature_bins.cat.rename_categories(['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        elif n_bins == 4:
            feature_bins = feature_bins.cat.rename_categories(['Low', 'Medium', 'High', 'Very High'])
        elif n_bins == 3:
            feature_bins = feature_bins.cat.rename_categories(['Low', 'Medium', 'High'])
        else:
            feature_bins = feature_bins.cat.rename_categories([f'Bin_{i+1}' for i in range(n_bins)])
    error_by_bin = pd.DataFrame({
        'bin': feature_bins,
        'abs_error': np.abs(errors)
    })
    
    error_stats = error_by_bin.groupby('bin')['abs_error'].agg(['mean', 'std']).reset_index()
    
    axes[1, 1].bar(error_stats['bin'], error_stats['mean'], 
                   yerr=error_stats['std'], capsize=5, color='lightcoral')
    axes[1, 1].set_xlabel(f'{top_feature} Bins')
    axes[1, 1].set_ylabel('Mean Absolute Error')
    axes[1, 1].set_title(f'Error by {top_feature} Level')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('xgboost_correlation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # =============================================================================
    # VISUALIZATION 3: Detailed Error Analysis
    # =============================================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Q-Q plot for error normality
    from scipy import stats
    stats.probplot(errors, dist="norm", plot=axes[0, 0])
    axes[0, 0].set_title('Q-Q Plot - Error Normality')
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Error vs prediction magnitude
    pred_quartiles = pd.qcut(test_pred, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    error_by_quartile = pd.DataFrame({
        'quartile': pred_quartiles,
        'abs_error': np.abs(errors),
        'rel_error': np.abs(errors / y_test) * 100
    })
    
    error_by_quartile.boxplot(column='abs_error', by='quartile', ax=axes[0, 1])
    axes[0, 1].set_title('Absolute Error by Prediction Quartile')
    axes[0, 1].set_xlabel('Prediction Quartile')
    axes[0, 1].set_ylabel('Absolute Error (kWh)')
    
    # 3. Relative error by quartile
    error_by_quartile.boxplot(column='rel_error', by='quartile', ax=axes[0, 2])
    axes[0, 2].set_title('Relative Error by Prediction Quartile')
    axes[0, 2].set_xlabel('Prediction Quartile')
    axes[0, 2].set_ylabel('Relative Error (%)')
    
    # 4. Error vs top 3 features
    top_3_features = importance_df.head(3)['feature'].values
    
    for i, feature in enumerate(top_3_features):
        ax = axes[1, i]
        ax.scatter(X_test[feature], np.abs(errors), alpha=0.6)
        ax.set_xlabel(feature)
        ax.set_ylabel('Absolute Error (kWh)')
        ax.set_title(f'Error vs {feature}')
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('xgboost_error_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Analysis visualizations saved:")
    print("- xgboost_analysis_overview.png")
    print("- xgboost_correlation_analysis.png") 
    print("- xgboost_error_analysis.png")

if __name__ == "__main__":
    create_analysis_visualizations()
