import pandas as pd
import numpy as np

def clean_data(df):
    """Clean and preprocess air quality data (safe assignments, UI time fix)"""
    df_clean = df.copy()

    # Combine Date and Time columns if they exist separately
    if 'Date' in df_clean.columns and 'Time' in df_clean.columns:
        # UI-specific: replace dots with colons in Time (e.g., "18.00.00" -> "18:00:00")
        df_clean['Time'] = df_clean['Time'].astype(str).str.replace('.', ':', regex=False)
        # Build a datetime series with explicit parsing where possible
        datetime_series = df_clean['Date'].astype(str).str.strip() + ' ' + df_clean['Time'].astype(str).str.strip()
        try:
            dt = pd.to_datetime(datetime_series, format='%d/%m/%Y %H:%M:%S', errors='coerce')
        except Exception:
            dt = pd.to_datetime(datetime_series, errors='coerce')
        df_clean = df_clean.assign(DateTime=dt)
        # Drop original Date/Time columns (non-inplace to avoid chained-assignment issues)
        df_clean = df_clean.drop(columns=['Date', 'Time'], errors='ignore')

    # Replace -200 values with NaN (missing data indicator in this dataset)
    df_clean = df_clean.replace(-200, np.nan)

    # Handle missing values for numeric columns (safe assignments)
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_clean[col].isnull().sum() > 0:
            series = df_clean[col].ffill().bfill()
            # If still NaN, use median if available
            if series.isnull().sum() > 0:
                median = series.median()
                if not np.isnan(median):
                    series = series.fillna(median)
            df_clean[col] = series

    # Remove duplicates (non-inplace approach)
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    removed = initial_rows - len(df_clean)
    if removed > 0:
        print(f"✓ Removed {removed} duplicate rows")

    # Remove outliers using IQR method (only for target variables)
    target_cols = ['CO(GT)', 'NOx(GT)', 'NO2(GT)', 'C6H6(GT)']
    for col in target_cols:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]

    print(f"✓ Data cleaned: {len(df_clean)} rows remaining")
    return df_clean