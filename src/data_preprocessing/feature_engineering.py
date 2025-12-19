import pandas as pd
import numpy as np

def create_features(df):
    """
    Generates temporal, lagged, and interaction features for air quality forecasting.
    
    This function transforms raw sensor data into a feature set suitable for 
    time-series machine learning models by capturing seasonality and trends.

    Args:
        df (pd.DataFrame): Cleaned dataframe containing 'DateTime' and pollutant levels.

    Returns:
        pd.DataFrame: Augmented dataframe with cyclical time features, 
                     rolling statistics, and lagged values.
    """
    df_features = df.copy()
    
    # --- 1. Temporal Feature Engineering ---
    if 'DateTime' in df_features.columns:
        df_features['DateTime'] = pd.to_datetime(df_features['DateTime'])
        df_features['hour'] = df_features['DateTime'].dt.hour
        df_features['day_of_week'] = df_features['DateTime'].dt.dayofweek
        
        # Cyclical Encoding: Convert hour to sin/cos to preserve the 
        # relationship between 23:00 and 00:00 (proximity in time).
        df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
        df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)

    # --- 2. Pollutant Lag & Rolling Statistics ---
    # Capturing the immediate history and local volatility of target pollutants.
    targets = ['co', 'no2', 'nox', 'benzene']
    for t in targets:
        if t in df_features.columns:
            # Lag 1: Provides the previous hour's value as a predictor.
            df_features[f'{t}_lag1'] = df_features[t].shift(1)
            # 3h Standard Deviation: Captures local variance/spikes in levels.
            df_features[f'{t}_std_3h'] = df_features[t].rolling(window=3).std()

    # --- 3. Sensor-Specific Features (PT08 Series) ---
    # These features track the rate of change in the raw sensor responses.
    sensor_cols = [col for col in df_features.columns if 'PT08' in col]
    for col in sensor_cols:
        # lag1: Previous sensor state.
        df_features[f'{col}_lag1'] = df_features[col].shift(1)
        # diff: Measures the velocity of change (current - previous).
        df_features[f'{col}_diff'] = df_features[col].diff()

    # --- 4. Environmental Interaction Features ---
    # Simple interaction term between Temperature (T) and Relative Humidity (RH).
    if 'T' in df_features.columns and 'RH' in df_features.columns:
        df_features['heat_index'] = df_features['T'] * df_features['RH']

    # --- 5. Data Integrity Cleanup ---
    # Rolling windows and shifts introduce NaNs at the start of the series.
    # ffill/bfill ensures no missing values are passed to the model.
    return df_features.ffill().bfill()