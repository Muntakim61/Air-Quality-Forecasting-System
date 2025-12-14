import pandas as pd
import numpy as np

def clean_data(df):
    """Clean and preprocess air quality data"""
    df_clean = df.copy()
    
    # Combine Date and Time columns if they exist separately
    if 'Date' in df_clean.columns and 'Time' in df_clean.columns:
        df_clean['DateTime'] = pd.to_datetime(df_clean['Date'] + ' ' + df_clean['Time'])
        df_clean.drop(['Date', 'Time'], axis=1, inplace=True)
    
    # Replace -200 values with NaN (missing data indicator in this dataset)
    df_clean.replace(-200, np.nan, inplace=True)
    
    # Handle missing values for numeric columns
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_clean[col].isnull().sum() > 0:
            # Use forward fill then backward fill for time series
            df_clean[col].fillna(method='ffill', inplace=True)
            df_clean[col].fillna(method='bfill', inplace=True)
            # If still NaN, use median
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
    
    # Remove duplicates
    initial_rows = len(df_clean)
    df_clean.drop_duplicates(inplace=True)
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