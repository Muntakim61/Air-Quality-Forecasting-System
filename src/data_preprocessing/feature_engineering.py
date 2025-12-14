import pandas as pd
import numpy as np

def create_features(df):
    """Create additional features for modeling"""
    df_features = df.copy()
    
    # Extract datetime features
    if 'DateTime' in df_features.columns:
        df_features['DateTime'] = pd.to_datetime(df_features['DateTime'])
        df_features['hour'] = df_features['DateTime'].dt.hour
        df_features['day_of_week'] = df_features['DateTime'].dt.dayofweek
        df_features['month'] = df_features['DateTime'].dt.month
        df_features['day_of_year'] = df_features['DateTime'].dt.dayofyear
        df_features['is_weekend'] = df_features['day_of_week'].isin([5, 6]).astype(int)
        
        # Cyclical encoding for hour
        df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
        df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)
        
        # Cyclical encoding for day of week
        df_features['dow_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
        df_features['dow_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)
        
        # Cyclical encoding for month
        df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
        df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
    
    # Create lag features for sensor readings
    sensor_cols = [col for col in df_features.columns if 'PT08' in col]
    for col in sensor_cols:
        if col in df_features.columns:
            df_features[f'{col}_lag1'] = df_features[col].shift(1)
            df_features[f'{col}_lag2'] = df_features[col].shift(2)
            df_features[f'{col}_rolling_mean_3'] = df_features[col].rolling(window=3, min_periods=1).mean()
    
    # Temperature and humidity interactions
    if 'T' in df_features.columns and 'RH' in df_features.columns:
        df_features['T_RH_interaction'] = df_features['T'] * df_features['RH']
        df_features['T_squared'] = df_features['T'] ** 2
    
    # Fill NaN values created by lag features
    df_features.fillna(method='bfill', inplace=True)
    
    # Rename target columns to simpler names
    rename_map = {
        'CO(GT)': 'co',
        'NOx(GT)': 'nox',
        'NO2(GT)': 'no2',
        'C6H6(GT)': 'benzene'
    }
    df_features.rename(columns=rename_map, inplace=True)
    
    print(f"✓ Features created: {df_features.shape[1]} total features")
    return df_features