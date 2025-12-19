import pandas as pd
import numpy as np

def clean_data(df):
    """
    Performs end-to-end cleaning of the Air Quality dataset.
    
    Processing steps:
    1. Standardizes DateTime formatting.
    2. Handles missing values (removes sensor-specific -200 flags).
    3. Imputes data using linear interpolation and median fallback.
    4. Clips outliers using the 10th and 90th percentiles.
    5. Renames chemical sensor columns to standardized shorthand.

    Args:
        df (pd.DataFrame): The raw AirQualityUCI dataset.

    Returns:
        pd.DataFrame: A cleaned, interpolated, and feature-mapped DataFrame.
    """
    df_clean = df.copy()

    # --- 1. DateTime Standardization ---
    if 'Date' in df_clean.columns and 'Time' in df_clean.columns:
        # Replace '.' with ':' to ensure Time follows HH:MM:SS format
        df_clean['Time'] = df_clean['Time'].astype(str).str.replace('.', ':', regex=False)
        
        # Concatenate Date and Time into a single string series
        datetime_series = df_clean['Date'].astype(str).str.strip() + ' ' + df_clean['Time'].astype(str).str.strip()
        
        # Parse strings into true Pandas DateTime objects
        df_clean['DateTime'] = pd.to_datetime(datetime_series, errors='coerce')
        
        # Remove legacy columns to maintain a tidy dataset
        df_clean = df_clean.drop(columns=['Date', 'Time'], errors='ignore')

    # --- 2. Handling Missing Values & Sensor Errors ---
    # The UCI dataset uses negative values (e.g., -200) to represent missing data.
    # We mask these as NaN to allow for proper interpolation.
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[numeric_cols] = df_clean[numeric_cols].mask(df_clean[numeric_cols] < 0, np.nan)

    # --- 3. Imputation Pipeline ---
    # Perform linear interpolation to fill gaps based on time-series trends.
    # limit_direction='both' ensures leading and trailing NaNs are also filled.
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[numeric_cols] = df_clean[numeric_cols].interpolate(method='linear', limit_direction='both')

    # Fallback: fill any remaining NaNs (where interpolation wasn't possible) with column medians.
    for col in numeric_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    # --- 4. Outlier Handling & Column Mapping ---
    # Map raw sensor names to standardized chemical keys used by the alert system.
    target_mapping = {'CO(GT)': 'co', 'NOx(GT)': 'nox', 'NO2(GT)': 'no2', 'C6H6(GT)': 'benzene'}
    
    for old_col, new_col in target_mapping.items():
        if old_col in df_clean.columns:
            # We use a strict 10-90 percentile clip to minimize the impact of 
            # sensor noise or extreme environmental anomalies.
            Q1 = df_clean[old_col].quantile(0.10) 
            Q3 = df_clean[old_col].quantile(0.90) 
            df_clean[old_col] = df_clean[old_col].clip(lower=Q1, upper=Q3)
            
            # Rename to short-form key
            df_clean = df_clean.rename(columns={old_col: new_col})

    # --- 5. Deduplication ---
    # Ensure each timestamp/record is unique to prevent bias during model training.
    df_clean = df_clean.drop_duplicates()
    
    return df_clean