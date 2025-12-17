import pandas as pd
import numpy as np

def clean_data(df):
    """Clean and preprocess air quality data with Interpolation and Clipping."""
    df_clean = df.copy()

    # 1. Standardize DateTime
    if 'Date' in df_clean.columns and 'Time' in df_clean.columns:
        df_clean['Time'] = df_clean['Time'].astype(str).str.replace('.', ':', regex=False)
        datetime_series = df_clean['Date'].astype(str).str.strip() + ' ' + df_clean['Time'].astype(str).str.strip()
        df_clean['DateTime'] = pd.to_datetime(datetime_series, errors='coerce')
        df_clean = df_clean.drop(columns=['Date', 'Time'], errors='ignore')

    # 2. Advanced Imputation: Replace -200 with NaN and use Linear Interpolation
    # This captures the smooth transition of air quality better than forward-fill
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[numeric_cols] = df_clean[numeric_cols].mask(df_clean[numeric_cols] < 0, np.nan)
    
    # Interpolate numeric columns only
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[numeric_cols] = df_clean[numeric_cols].interpolate(method='linear', limit_direction='both')

    # 3. Handle remaining NaNs (at edges) with median
    for col in numeric_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    # 4. Outlier Handling: Clipping (Winsorization)
    # Instead of dropping rows, we cap extreme values to reduce data loss
    target_mapping = {'CO(GT)': 'co', 'NOx(GT)': 'nox', 'NO2(GT)': 'no2', 'C6H6(GT)': 'benzene'}
    
    for old_col, new_col in target_mapping.items():
        if old_col in df_clean.columns:
            Q1 = df_clean[old_col].quantile(0.10) # Using 10th percentile
            Q3 = df_clean[old_col].quantile(0.90) # Using 90th percentile
            df_clean[old_col] = df_clean[old_col].clip(lower=Q1, upper=Q3)
            df_clean = df_clean.rename(columns={old_col: new_col})

    df_clean = df_clean.drop_duplicates()
    return df_clean