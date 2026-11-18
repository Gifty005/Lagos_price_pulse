# src/features.py

import pandas as pd

def build_features(filepath="data/processed/master_mile12_data.csv"):
    """
    Takes the master Mile 12 dataset, creates lagged features, 
    seasonality flags, and returns a modeling-ready DataFrame.
    """
    df = pd.read_csv(filepath)
    
    # Ensure date column is datetime and sorted chronologically
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
    # 1. Price Momentum Lags
    df['tomato_price_lag1'] = df['tomato_price_ngn'].shift(1)
    df['tomato_price_lag2'] = df['tomato_price_ngn'].shift(2)
    df['onion_price_lag1'] = df['onion_price_ngn'].shift(1)
    df['onion_price_lag2'] = df['onion_price_ngn'].shift(2)
    
    # 2. Cost & Climate Shock Lags
    df['diesel_price_lag1'] = df['diesel_price_ngn'].shift(1)
    df['petrol_price_lag1'] = df['petrol_price_ngn'].shift(1)
    df['usd_ngn_lag1'] = df['usd_ngn_rate'].shift(1)
    df['north_rainfall_lag2'] = df['north_rainfall_mm'].shift(2)
    
    # 3. Seasonality Features
    if 'date' in df.columns:
        df['month'] = df['date'].dt.month
    else:
        df['month'] = (df.index % 12) + 1
        
    # Rainy season flag: 1 for May–September (5 to 9), 0 otherwise
    df['is_rainy_season'] = df['month'].apply(lambda m: 1 if 5 <= m <= 9 else 0)
    
    # Drop initial rows with NaN values resulting from lagging
    df_model = df.dropna().reset_index(drop=True)
    
    return df_model

if __name__ == "__main__":
    df_processed = build_features()
    print(f"Processed shape: {df_processed.shape}")
    print(df_processed.head())