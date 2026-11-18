import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

from src.features import build_features

def main():
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    
    print("Loading and building features...")
    df = build_features()
    
    # Ensure 'date' column is datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Part A: Strict Chronological Split (No Shuffling)
    print("Splitting dataset chronologically...")
    train_df = df[df['date'] < '2025-01-01'].copy()
    test_df = df[df['date'] >= '2025-01-01'].copy()
    
    # Define feature columns (X)
    feature_cols = [
        'tomato_price_lag1', 'tomato_price_lag2', 
        'onion_price_lag1', 'onion_price_lag2', 
        'diesel_price_lag1', 'petrol_price_lag1', 
        'usd_ngn_lag1', 'north_rainfall_lag2', 
        'month', 'is_rainy_season'
    ]
    
    # Drop rows with missing values in features or targets for the split datasets
    targets = ['tomato_price_ngn', 'onion_price_ngn']
    train_df = train_df.dropna(subset=feature_cols + targets)
    test_df = test_df.dropna(subset=feature_cols + targets)
    
    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    
    commodities = {
        'tomato': {
            'target': 'tomato_price_ngn',
            'baseline_col': 'tomato_price_lag1'
        },
        'onion': {
            'target': 'onion_price_ngn',
            'baseline_col': 'onion_price_lag1'
        }
    }
    
    metrics_records = []
    
    # Part B & C: Model Training, Benchmarking & Selection
    for comm_name, cfg in commodities.items():
        target_col = cfg['target']
        baseline_col = cfg['baseline_col']
        
        y_train = train_df[target_col]
        y_test = test_df[target_col]
        
        print(f"\n--- Training models for {comm_name.upper()} ({target_col}) ---")
        
        # 1. Persistence Baseline
        preds_baseline = test_df[baseline_col]
        mae_base = mean_absolute_error(y_test, preds_baseline)
        rmse_base = np.sqrt(mean_squared_error(y_test, preds_baseline))
        
        models_perf = {
            'Persistence Baseline': {
                'model': None,
                'mae': mae_base,
                'rmse': rmse_base
            }
        }
        
        # 2. Linear Regression
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        preds_lr = lr.predict(X_test)
        models_perf['Linear Regression'] = {
            'model': lr,
            'mae': mean_absolute_error(y_test, preds_lr),
            'rmse': np.sqrt(mean_squared_error(y_test, preds_lr))
        }
        
        # 3. Random Forest Regressor
        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        rf.fit(X_train, y_train)
        preds_rf = rf.predict(X_test)
        models_perf['Random Forest'] = {
            'model': rf,
            'mae': mean_absolute_error(y_test, preds_rf),
            'rmse': np.sqrt(mean_squared_error(y_test, preds_rf))
        }
        
        # 4. XGBoost Regressor
        xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)
        xgb.fit(X_train, y_train)
        preds_xgb = xgb.predict(X_test)
        models_perf['XGBoost'] = {
            'model': xgb,
            'mae': mean_absolute_error(y_test, preds_xgb),
            'rmse': np.sqrt(mean_squared_error(y_test, preds_xgb))
        }
        
        # Record metrics and find the best model based on lowest MAE
        best_model_name = None
        min_mae = float('inf')
        best_model_obj = None
        
        for m_name, perf in models_perf.items():
            metrics_records.append({
                'commodity': comm_name,
                'model': m_name,
                'MAE': perf['mae'],
                'RMSE': perf['rmse']
            })
            if perf['mae'] < min_mae:
                min_mae = perf['mae']
                best_model_name = m_name
                best_model_obj = perf['model']
                
        print(f"Best model for {comm_name}: {best_model_name} (MAE: {min_mae:.4f})")
        
        # Save the winning model to joblib
        if best_model_name == 'Persistence Baseline':
            joblib.dump({'type': 'persistence', 'column': baseline_col}, f'models/{comm_name}_model.joblib')
        else:
            joblib.dump(best_model_obj, f'models/{comm_name}_model.joblib')

    # Save metrics comparison table to CSV
    metrics_df = pd.DataFrame(metrics_records)
    metrics_path = 'models/model_metrics.csv'
    metrics_df.to_csv(metrics_path, index=False)
    
    print("\n================ 2025 TEST COMPARISON TABLE ================")
    print(metrics_df.to_string(index=False))
    print(f"\nMetrics successfully saved to {metrics_path}")
    print("Winning models saved to models/tomato_model.joblib and models/onion_model.joblib")

if __name__ == '__main__':
    main()