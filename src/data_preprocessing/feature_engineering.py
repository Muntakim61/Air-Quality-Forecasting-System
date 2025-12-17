import pandas as pd
import numpy as np

def create_features(df):
    """Enhanced feature engineering with Target Lags and Rolling Volatility."""
    df_features = df.copy()
    
    if 'DateTime' in df_features.columns:
        df_features['DateTime'] = pd.to_datetime(df_features['DateTime'])
        df_features['hour'] = df_features['DateTime'].dt.hour
        df_features['day_of_week'] = df_features['DateTime'].dt.dayofweek
        
        # Cyclical encoding
        df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
        df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)

    # 1. Target Lags: What happened 1 hour ago is the best predictor for now
    targets = ['co', 'no2', 'nox', 'benzene']
    for t in targets:
        if t in df_features.columns:
            df_features[f'{t}_lag1'] = df_features[t].shift(1)
            # Rolling volatility: captures sudden spikes
            df_features[f'{t}_std_3h'] = df_features[t].rolling(window=3).std()

    # 2. Sensor Features
    sensor_cols = [col for col in df_features.columns if 'PT08' in col]
    for col in sensor_cols:
        df_features[f'{col}_lag1'] = df_features[col].shift(1)
        df_features[f'{col}_diff'] = df_features[col].diff()
    
    # 3. Environmental Interactions
    if 'T' in df_features.columns and 'RH' in df_features.columns:
        df_features['heat_index'] = df_features['T'] * df_features['RH']
        
    # Crucial: fill the NaNs created by lagging/diffing
    return df_features.ffill().bfill()