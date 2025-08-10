"""
Quick but comprehensive XGBoost model analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
from consumption_model import EVConsumptionPredictor
import warnings
warnings.filterwarnings('ignore')

def run_quick_analysis():
    """Run a comprehensive but faster analysis"""
    
    print("COMPREHENSIVE XGBOOST MODEL ANALYSIS")
    print("="*80)
    
    # Load predictor and data
    predictor = EVConsumptionPredictor()
    data = predictor.load_data()
    data = predictor.engineer_features(data)
    X, y = predictor.prepare_features(data)
    
    # Split data (same as training)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    
    # Train models
    predictor.train_models(X_train, y_train, X_val, y_val)
    xgb_model = predictor.models['xgboost']
    
    print(f"\nData: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test samples")
    
    # =============================================================================
    # 1. OVERFITTING ANALYSIS
    # =============================================================================
    print("\n" + "="*60)
    print("1. OVERFITTING ANALYSIS")
    print("="*60)
    
    # Performance on different sets
    train_pred = xgb_model.predict(X_train)
    val_pred = xgb_model.predict(X_val)
    test_pred = xgb_model.predict(X_test)
    
    # Calculate metrics
    performance_data = []
    for name, y_true, y_pred in [('Training', y_train, train_pred), 
                                ('Validation', y_val, val_pred), 
                                ('Test', y_test, test_pred)]:
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        performance_data.append({
            'Dataset': name, 'MAE': mae, 'RMSE': rmse, 'R²': r2, 'MAPE': mape
        })
    
    perf_df = pd.DataFrame(performance_data)
    print("\nPerformance Comparison:")
    print(perf_df.round(4))
    
    # Overfitting analysis
    train_r2 = perf_df[perf_df['Dataset'] == 'Training']['R²'].iloc[0]
    val_r2 = perf_df[perf_df['Dataset'] == 'Validation']['R²'].iloc[0]
    test_r2 = perf_df[perf_df['Dataset'] == 'Test']['R²'].iloc[0]
    
    overfitting_gap = train_r2 - val_r2
    generalization_gap = val_r2 - test_r2
    
    print(f"\nOverfitting Metrics:")
    print(f"Train-Validation Gap: {overfitting_gap:.4f}")
    print(f"Validation-Test Gap: {generalization_gap:.4f}")
    
    if overfitting_gap > 0.02:
        print("WARNING: Potential overfitting detected (train-val gap > 0.02)")
    elif overfitting_gap > 0.01:
        print("CAUTION: Mild overfitting detected (train-val gap > 0.01)")
    else:
        print("GOOD: No significant overfitting detected")
    
    # =============================================================================
    # 2. FEATURE IMPORTANCE ANALYSIS
    # =============================================================================
    print("\n" + "="*60)
    print("2. FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    # Get feature importance
    importance_gain = xgb_model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': importance_gain
    }).sort_values('importance', ascending=False)
    
    print("\nTop 15 Most Important Features:")
    for i, (_, row) in enumerate(importance_df.head(15).iterrows(), 1):
        print(f"{i:2d}. {row['feature']:30s} : {row['importance']:.4f}")
    
    # Importance concentration analysis
    total_importance = importance_gain.sum()
    top_5_importance = importance_df.head(5)['importance'].sum()
    top_10_importance = importance_df.head(10)['importance'].sum()
    
    print(f"\nFeature Concentration:")
    print(f"Top 5 features: {top_5_importance/total_importance*100:.1f}% of total importance")
    print(f"Top 10 features: {top_10_importance/total_importance*100:.1f}% of total importance")
    
    if top_5_importance/total_importance > 0.8:
        print("WARNING: Model heavily relies on few features (>80% from top 5)")
        print("RECOMMENDATION: Consider feature selection or investigate data leakage")
    elif top_5_importance/total_importance > 0.6:
        print("CAUTION: Model moderately concentrated on top features")
    else:
        print("GOOD: Feature importance is well distributed")
    
    # =============================================================================
    # 3. CORRELATION ANALYSIS
    # =============================================================================
    print("\n" + "="*60)
    print("3. CORRELATION ANALYSIS")
    print("="*60)
    
    # Calculate correlation matrix for top 20 features
    top_20_features = importance_df.head(20)['feature'].values
    corr_matrix = X_train[top_20_features].corr()
    
    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(top_20_features)):
        for j in range(i+1, len(top_20_features)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.8:
                high_corr_pairs.append({
                    'feature1': top_20_features[i],
                    'feature2': top_20_features[j],
                    'correlation': corr_val
                })
    
    if high_corr_pairs:
        print(f"High Correlation Pairs (|correlation| > 0.8):")
        for pair in sorted(high_corr_pairs, key=lambda x: abs(x['correlation']), reverse=True):
            print(f"  {pair['feature1']} <-> {pair['feature2']}: {pair['correlation']:.3f}")
        print(f"\nWARNING: {len(high_corr_pairs)} highly correlated feature pairs detected")
        print("RECOMMENDATION: Consider removing redundant features")
    else:
        print("GOOD: No highly correlated feature pairs found (threshold: 0.8)")
    
    # Correlation with target
    target_correlations = []
    for feature in top_20_features:
        corr = np.corrcoef(X_train[feature], y_train)[0, 1]
        target_correlations.append({'feature': feature, 'target_corr': abs(corr)})
    
    target_corr_df = pd.DataFrame(target_correlations).sort_values('target_corr', ascending=False)
    print(f"\nTop 10 Features by Target Correlation:")
    for i, (_, row) in enumerate(target_corr_df.head(10).iterrows(), 1):
        print(f"{i:2d}. {row['feature']:30s} : {row['target_corr']:.4f}")
    
    # =============================================================================
    # 4. ERROR ANALYSIS
    # =============================================================================
    print("\n" + "="*60)
    print("4. PREDICTION ERROR ANALYSIS")
    print("="*60)
    
    errors = y_test - test_pred
    
    # Error statistics
    print(f"Error Statistics:")
    print(f"  Mean Error: {errors.mean():.4f}")
    print(f"  Std Error: {errors.std():.4f}")
    print(f"  Min Error: {errors.min():.4f}")
    print(f"  Max Error: {errors.max():.4f}")
    print(f"  Median Error: {errors.median():.4f}")
    
    # Outlier analysis
    z_scores = np.abs(stats.zscore(errors))
    outliers = z_scores > 3
    outlier_count = outliers.sum()
    
    print(f"\nOutlier Analysis:")
    print(f"  Outliers (|z-score| > 3): {outlier_count} ({outlier_count/len(errors)*100:.1f}%)")
    
    if outlier_count > len(errors) * 0.05:
        print("WARNING: High number of outliers detected (>5%)")
        print("RECOMMENDATION: Investigate outliers and consider robust scaling")
    else:
        print("GOOD: Outlier rate is acceptable")
    
    # Bias test
    from scipy.stats import ttest_1samp
    t_stat, p_value = ttest_1samp(errors, 0)
    
    print(f"\nBias Analysis:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.6f}")
    
    if p_value < 0.05:
        print("WARNING: Significant bias detected in predictions")
        print("RECOMMENDATION: Check for systematic errors or missing features")
    else:
        print("GOOD: No significant bias detected")
    
    # Error patterns by prediction magnitude
    pred_quartiles = pd.qcut(test_pred, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    error_by_quartile = pd.DataFrame({
        'quartile': pred_quartiles,
        'abs_error': np.abs(errors),
        'rel_error': np.abs(errors / y_test) * 100
    })
    
    quartile_stats = error_by_quartile.groupby('quartile').agg({
        'abs_error': ['mean', 'std'],
        'rel_error': ['mean', 'std']
    }).round(4)
    
    print(f"\nError by Prediction Quartile:")
    print(quartile_stats)
    
    # =============================================================================
    # 5. MODEL DIAGNOSTICS
    # =============================================================================
    print("\n" + "="*60)
    print("5. MODEL DIAGNOSTICS")
    print("="*60)
    
    # Model complexity
    print(f"Model Parameters:")
    print(f"  Max Depth: {xgb_model.max_depth}")
    print(f"  Number of Estimators: {xgb_model.n_estimators}")
    print(f"  Learning Rate: {xgb_model.learning_rate}")
    
    # Feature usage efficiency
    zero_importance = (importance_gain == 0).sum()
    low_importance = (importance_gain < 0.001).sum()
    
    print(f"\nFeature Usage:")
    print(f"  Total Features: {len(importance_gain)}")
    print(f"  Zero Importance: {zero_importance}")
    print(f"  Low Importance (<0.001): {low_importance}")
    print(f"  Effective Features: {len(importance_gain) - low_importance}")
    
    if zero_importance > len(importance_gain) * 0.2:
        print("WARNING: Many features have zero importance")
        print("RECOMMENDATION: Consider feature selection")
    
    # =============================================================================
    # 6. RECOMMENDATIONS
    # =============================================================================
    print("\n" + "="*60)
    print("6. IMPROVEMENT RECOMMENDATIONS")
    print("="*60)
    
    recommendations = []
    
    # Based on overfitting
    if overfitting_gap > 0.02:
        recommendations.extend([
            "Reduce model complexity (lower max_depth, increase min_child_weight)",
            "Add more regularization (increase reg_alpha, reg_lambda)",
            "Use early stopping with more patience"
        ])
    
    # Based on feature concentration
    if top_5_importance/total_importance > 0.8:
        recommendations.extend([
            "Investigate top features for potential data leakage",
            "Consider feature selection to remove redundant features",
            "Analyze if model is too dependent on few features"
        ])
    
    # Based on correlations
    if len(high_corr_pairs) > 3:
        recommendations.extend([
            "Remove highly correlated features to reduce multicollinearity",
            "Consider PCA or feature engineering to combine correlated features"
        ])
    
    # Based on errors
    if outlier_count > len(errors) * 0.05:
        recommendations.extend([
            "Investigate and handle outliers in training data",
            "Consider robust scaling or outlier removal techniques"
        ])
    
    if p_value < 0.05:
        recommendations.extend([
            "Address systematic bias in predictions",
            "Check for missing important features or data quality issues"
        ])
    
    # General recommendations
    recommendations.extend([
        "Try hyperparameter tuning (Bayesian optimization)",
        "Consider ensemble methods (combine with other models)",
        "Validate on completely unseen data from different time periods"
    ])
    
    print("Key Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i:2d}. {rec}")
    
    # =============================================================================
    # 7. SUMMARY
    # =============================================================================
    print("\n" + "="*60)
    print("7. ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"Model Performance:")
    print(f"  Test R²: {test_r2:.4f}")
    print(f"  Test RMSE: {perf_df[perf_df['Dataset'] == 'Test']['RMSE'].iloc[0]:.4f} kWh")
    print(f"  Test MAPE: {perf_df[perf_df['Dataset'] == 'Test']['MAPE'].iloc[0]:.1f}%")
    
    print(f"\nModel Health:")
    print(f"  Overfitting Risk: {'Low' if overfitting_gap < 0.01 else 'Medium' if overfitting_gap < 0.02 else 'High'}")
    print(f"  Feature Concentration: {'High' if top_5_importance/total_importance > 0.8 else 'Medium' if top_5_importance/total_importance > 0.6 else 'Good'}")
    print(f"  Multicollinearity: {'High' if len(high_corr_pairs) > 5 else 'Medium' if len(high_corr_pairs) > 2 else 'Low'}")
    print(f"  Outlier Rate: {outlier_count/len(errors)*100:.1f}%")
    print(f"  Bias: {'Detected' if p_value < 0.05 else 'None'}")
    
    # Overall assessment
    issues = 0
    if overfitting_gap > 0.02: issues += 1
    if top_5_importance/total_importance > 0.8: issues += 1
    if len(high_corr_pairs) > 5: issues += 1
    if outlier_count > len(errors) * 0.05: issues += 1
    if p_value < 0.05: issues += 1
    
    print(f"\nOverall Assessment:")
    if issues == 0:
        print("EXCELLENT: Model appears well-tuned with no major issues")
    elif issues <= 2:
        print("GOOD: Model performs well with minor areas for improvement")
    elif issues <= 3:
        print("FAIR: Model has moderate issues that should be addressed")
    else:
        print("POOR: Model has several issues requiring attention")
    
    print(f"\nTop 3 Priority Actions:")
    priority_actions = []
    if top_5_importance/total_importance > 0.8:
        priority_actions.append("1. Investigate feature concentration and potential data leakage")
    if len(high_corr_pairs) > 3:
        priority_actions.append("2. Address multicollinearity by removing redundant features")
    if overfitting_gap > 0.02:
        priority_actions.append("3. Reduce overfitting through regularization")
    
    if not priority_actions:
        priority_actions = [
            "1. Fine-tune hyperparameters for optimal performance",
            "2. Validate on external datasets",
            "3. Consider ensemble methods for robustness"
        ]
    
    for action in priority_actions[:3]:
        print(f"  {action}")
    
    return {
        'performance': perf_df,
        'overfitting_gap': overfitting_gap,
        'feature_importance': importance_df,
        'high_correlations': high_corr_pairs,
        'error_stats': errors.describe(),
        'recommendations': recommendations
    }

if __name__ == "__main__":
    results = run_quick_analysis()
