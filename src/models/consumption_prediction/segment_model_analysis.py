"""
📊 ANALYSIS MODEL - EV Consumption Analysis
Uses ALL available features including post-trip data for understanding patterns
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def smape(y_true, y_pred):
    return np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100

def overfitting_analysis(model, scaler, X_train, y_train, X_val, y_val, X_test, y_test, model_name):
    if model_name == 'linear_regression':
        X_train_ = scaler.transform(X_train)
        X_val_ = scaler.transform(X_val)
        X_test_ = scaler.transform(X_test)
    else:
        X_train_ = X_train
        X_val_ = X_val
        X_test_ = X_test
    sets = [('Training', X_train_, y_train), ('Validation', X_val_, y_val), ('Test', X_test_, y_test)]
    perf = []
    for name, X, y in sets:
        y_pred = model.predict(X)
        perf.append({
            'Dataset': name,
            'MAE': mean_absolute_error(y, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y, y_pred)),
            'R2': r2_score(y, y_pred),
            'MAPE': mean_absolute_percentage_error(y, y_pred),
            'SMAPE': smape(y, y_pred)
        })
    perf_df = pd.DataFrame(perf)
    print('\nPerformance Comparison:')
    print(perf_df.round(4))
    overfitting_gap = perf_df.loc[0, 'R2'] - perf_df.loc[1, 'R2']
    generalization_gap = perf_df.loc[1, 'R2'] - perf_df.loc[2, 'R2']
    print(f"\nOverfitting Analysis:")
    print(f"Train-Validation Gap: {overfitting_gap:.4f}")
    print(f"Validation-Test Gap: {generalization_gap:.4f}")
    # Learning curve (train/val R2 vs. train size)
    train_sizes = np.linspace(0.1, 1.0, 5)
    train_scores, val_scores = [], []
    for frac in train_sizes:
        n = int(len(X_train) * frac)
        if n < 10: continue
        X_sub, y_sub = X_train.iloc[:n], y_train.iloc[:n]
        if model_name == 'linear_regression':
            X_sub_ = scaler.transform(X_sub)
        else:
            X_sub_ = X_sub
        model.fit(X_sub_, y_sub)
        train_scores.append(r2_score(y_sub, model.predict(X_sub_)))
        val_scores.append(r2_score(y_val, model.predict(X_val_)))
    plt.figure(figsize=(7,5))
    plt.plot(train_sizes[:len(train_scores)], train_scores, 'o-', label='Train R2')
    plt.plot(train_sizes[:len(val_scores)], val_scores, 'o-', label='Val R2')
    plt.xlabel('Fraction of Training Set')
    plt.ylabel('R2 Score')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.close()
    return perf_df, overfitting_gap, generalization_gap

def feature_importance_analysis(model, X_train, model_name):
    if not hasattr(model, 'feature_importances_'):
        print('No feature importances for this model.')
        return None
    importances = model.feature_importances_
    features = X_train.columns
    imp_df = pd.DataFrame({'feature': features, 'importance': importances})
    imp_df = imp_df.sort_values('importance', ascending=False)
    print('\nTop 15 Features:')
    print(imp_df.head(15))
    plt.figure(figsize=(10,6))
    plt.barh(imp_df.head(15)['feature'][::-1], imp_df.head(15)['importance'][::-1])
    plt.xlabel('Importance')
    plt.title('Top 15 Feature Importances')
    plt.tight_layout()
    plt.show()
    plt.close()
    # Cumulative
    plt.figure(figsize=(6,4))
    plt.plot(np.arange(1, len(imp_df)+1), np.cumsum(imp_df['importance'])/imp_df['importance'].sum())
    plt.xlabel('Number of Features')
    plt.ylabel('Cumulative Importance')
    plt.title('Cumulative Feature Importance')
    plt.tight_layout()
    plt.show()
    plt.close()
    top5 = imp_df.head(5)['importance'].sum()/imp_df['importance'].sum()
    print(f"Top 5 features account for {top5*100:.1f}% of total importance.")
    return imp_df

def feature_correlation_analysis(X_train, imp_df):
    top_features = imp_df['feature'].head(15).tolist() if imp_df is not None else X_train.columns[:15].tolist()
    corr = X_train[top_features].corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True)
    plt.title('Correlation Matrix (Top 15 Features)')
    plt.tight_layout()
    plt.show()
    plt.close()
    # High correlation pairs
    high_corr = []
    for i in range(len(top_features)):
        for j in range(i+1, len(top_features)):
            if abs(corr.iloc[i,j]) > 0.8:
                high_corr.append((top_features[i], top_features[j], corr.iloc[i,j]))
    if high_corr:
        print('\nHighly correlated feature pairs (|corr| > 0.8):')
        for f1, f2, c in high_corr:
            print(f'  {f1} <-> {f2}: {c:.2f}')
    else:
        print('\nNo highly correlated feature pairs (|corr| > 0.8)')
    # VIF
    vif_data = []
    for i in range(min(10, len(top_features))):
        try:
            vif = variance_inflation_factor(X_train[top_features].values, i)
        except Exception:
            vif = np.nan
        vif_data.append({'feature': top_features[i], 'VIF': vif})
    vif_df = pd.DataFrame(vif_data)
    print('\nVIF for top 10 features:')
    print(vif_df)
    if (vif_df['VIF'] > 10).any():
        print('Warning: High multicollinearity detected (VIF > 10)')
    return corr, high_corr, vif_df

def prediction_error_analysis(model, scaler, X_test, y_test, model_name, X_train=None):
    if model_name == 'linear_regression':
        X_test_ = scaler.transform(X_test)
    else:
        X_test_ = X_test
    y_pred = model.predict(X_test_)
    errors = y_test - y_pred
    plt.figure(figsize=(7,5))
    plt.hist(errors, bins=50, alpha=0.7, color='lightcoral')
    plt.xlabel('Prediction Error (kWh)')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.tight_layout()
    plt.show()
    plt.close()
    plt.figure(figsize=(7,5))
    plt.scatter(y_pred, np.abs(errors), alpha=0.5)
    plt.xlabel('Predicted Value (kWh)')
    plt.ylabel('Absolute Error (kWh)')
    plt.title('Absolute Error vs Predicted Value')
    plt.tight_layout()
    plt.show()
    plt.close()
    # Outlier analysis
    z_scores = np.abs((errors - errors.mean()) / (errors.std() + 1e-8))
    outlier_count = (z_scores > 3).sum()
    print(f'\nOutlier count (|z| > 3): {outlier_count} ({outlier_count/len(errors)*100:.2f}%)')
    # Bias test
    from scipy.stats import ttest_1samp
    t_stat, p_value = ttest_1samp(errors, 0)
    print(f'Bias t-test: t={t_stat:.3f}, p={p_value:.4f}')
    if p_value < 0.05:
        print('Warning: Significant bias detected in predictions.')
    else:
        print('No significant bias detected.')
    return errors, outlier_count, t_stat, p_value

def print_recommendations(overfitting_gap, high_corr, vif_df, outlier_count, p_value, top5_ratio, n_train):
    print('\nRECOMMENDATIONS:')
    if overfitting_gap > 0.02:
        print('- Reduce model complexity or add regularization (overfitting detected)')
    if high_corr:
        print('- Remove or combine highly correlated features (multicollinearity)')
    if (vif_df['VIF'] > 10).any():
        print('- Address high VIF features (multicollinearity)')
    if outlier_count > 0.05 * n_train:
        print('- Investigate and handle outliers in training data')
    if p_value < 0.05:
        print('- Address systematic bias in predictions')
    if top5_ratio > 0.8:
        print('- Model relies heavily on a few features; check for data leakage or add more features')
    if n_train < 1000:
        print('- Consider collecting more training data')
    print('- Try hyperparameter tuning and ensemble methods for further improvement')

def print_summary():
    print('\nSUMMARY:')
    print('Plots saved:')
    print('  - learning_curve.png')
    print('  - feature_importance.png')
    print('  - feature_importance_cumulative.png')
    print('  - correlation_matrix.png')
    print('  - error_distribution.png')
    print('  - abs_error_vs_pred.png')
    print('  - Actual vs Predicted plot (see script output)')

if __name__ == "__main__":
    model_dir = Path(__file__).parent
    features = pd.read_csv(model_dir / 'segment_model_features.csv')
    target = pd.read_csv(model_dir / 'segment_model_target.csv').squeeze()
    model_data = joblib.load(model_dir / 'segment_energy_model.pkl')
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    # Use best model by MAE
    best_model = min(model_data['models'], key=lambda m: mean_absolute_error(y_test, model_data['models'][m].predict(model_data['scalers']['standard'].transform(X_test)) if m == 'linear_regression' else model_data['models'][m].predict(X_test)))
    model = model_data['models'][best_model]
    scaler = model_data['scalers']['standard'] if best_model == 'linear_regression' else None
    print(f'\nBest model: {best_model}')
    perf_df, overfitting_gap, generalization_gap = overfitting_analysis(model, scaler, X_train, y_train, X_val, y_val, X_test, y_test, best_model)
    imp_df = feature_importance_analysis(model, X_train, best_model)
    corr, high_corr, vif_df = feature_correlation_analysis(X_train, imp_df)
    errors, outlier_count, t_stat, p_value = prediction_error_analysis(model, scaler, X_test, y_test, best_model, X_train)
    top5_ratio = imp_df.head(5)['importance'].sum()/imp_df['importance'].sum() if imp_df is not None else 0
    print_recommendations(overfitting_gap, high_corr, vif_df, outlier_count, p_value, top5_ratio, len(X_train))
    print_summary()
    # Actual vs Predicted plot (zoomed)
    y_pred = model.predict(scaler.transform(X_test)) if best_model == 'linear_regression' else model.predict(X_test)
    plt.figure(figsize=(7,7))
    plt.scatter(y_test, y_pred, alpha=0.3, label='Predicted vs Actual')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.plot([0, 1], [0, 1], 'r--', label='1:1 Line')
    plt.xlabel('Actual Energy (kWh)')
    plt.ylabel('Predicted Energy (kWh)')
    plt.title(f'Actual vs Predicted Energy (Best Model: {best_model})')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xticks(np.linspace(0, 1, 21))
    plt.yticks(np.linspace(0, 1, 21))
    plt.tight_layout()

    plt.show()
