"""
Standalone training script - Place this in project root directory
Run: python train_models.py
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import joblib
import os

# Set random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def load_raw_data(file_path):
    """Load raw CSV data"""
    df = pd.read_csv(file_path)
    print(f"✓ Loaded {len(df)} rows")
    return df

def clean_data(df):
    """Clean data"""
    df_clean = df.copy()
    
    # Combine Date and Time
    if 'Date' in df_clean.columns and 'Time' in df_clean.columns:
        df_clean['DateTime'] = pd.to_datetime(df_clean['Date'] + ' ' + df_clean['Time'])
        df_clean.drop(['Date', 'Time'], axis=1, inplace=True)
    
    # Replace -200 with NaN
    df_clean.replace(-200, np.nan, inplace=True)
    
    # Fill missing values
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col].fillna(method='ffill', inplace=True)
            df_clean[col].fillna(method='bfill', inplace=True)
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
    
    # Remove duplicates
    df_clean.drop_duplicates(inplace=True)
    
    # Remove outliers for target columns
    target_cols = ['CO(GT)', 'NOx(GT)', 'NO2(GT)', 'C6H6(GT)']
    for col in target_cols:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 3 * IQR
            upper = Q3 + 3 * IQR
            df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
    
    print(f"✓ Cleaned: {len(df_clean)} rows")
    return df_clean

def create_features(df):
    """Create features"""
    df_features = df.copy()
    
    if 'DateTime' in df_features.columns:
        df_features['DateTime'] = pd.to_datetime(df_features['DateTime'])
        df_features['hour'] = df_features['DateTime'].dt.hour
        df_features['day_of_week'] = df_features['DateTime'].dt.dayofweek
        df_features['month'] = df_features['DateTime'].dt.month
        df_features['day_of_year'] = df_features['DateTime'].dt.dayofyear
        df_features['is_weekend'] = df_features['day_of_week'].isin([5, 6]).astype(int)
        
        # Cyclical features
        df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
        df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)
        df_features['dow_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
        df_features['dow_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)
        df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
        df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
    
    # Lag features
    sensor_cols = [col for col in df_features.columns if 'PT08' in col]
    for col in sensor_cols:
        df_features[f'{col}_lag1'] = df_features[col].shift(1)
        df_features[f'{col}_lag2'] = df_features[col].shift(2)
        df_features[f'{col}_rolling_mean_3'] = df_features[col].rolling(window=3, min_periods=1).mean()
    
    # T and RH interactions
    if 'T' in df_features.columns and 'RH' in df_features.columns:
        df_features['T_RH_interaction'] = df_features['T'] * df_features['RH']
        df_features['T_squared'] = df_features['T'] ** 2
    
    df_features.fillna(method='bfill', inplace=True)
    
    # Rename targets
    rename_map = {
        'CO(GT)': 'co',
        'NOx(GT)': 'nox',
        'NO2(GT)': 'no2',
        'C6H6(GT)': 'benzene'
    }
    df_features.rename(columns=rename_map, inplace=True)
    
    print(f"✓ Features: {df_features.shape[1]} columns")
    return df_features

def train_ensemble_models(X, y, target_names, output_dir='outputs/models'):
    """Train ensemble models"""
    os.makedirs(output_dir, exist_ok=True)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    
    results = {}
    
    for i, target in enumerate(target_names):
        print(f"\n{'='*60}")
        print(f"Training: {target.upper()}")
        print(f"{'='*60}")
        
        y_train_target = y_train.iloc[:, i]
        y_test_target = y_test.iloc[:, i]
        
        models = {
            'RandomForest': RandomForestRegressor(
                n_estimators=100, max_depth=15, random_state=RANDOM_SEED, n_jobs=-1
            ),
            'XGBoost': XGBRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1, 
                random_state=RANDOM_SEED, n_jobs=-1
            ),
            'LightGBM': LGBMRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                random_state=RANDOM_SEED, n_jobs=-1, verbose=-1
            )
        }
        
        predictions = {}
        for model_name, model in models.items():
            print(f"  Training {model_name}...")
            model.fit(X_train, y_train_target)
            y_pred = model.predict(X_test)
            predictions[model_name] = y_pred
            
            rmse = np.sqrt(mean_squared_error(y_test_target, y_pred))
            r2 = r2_score(y_test_target, y_pred)
            print(f"    RMSE: {rmse:.4f} | R²: {r2:.4f}")
        
        # Ensemble
        ensemble_pred = np.mean(list(predictions.values()), axis=0)
        ensemble_rmse = np.sqrt(mean_squared_error(y_test_target, ensemble_pred))
        ensemble_r2 = r2_score(y_test_target, ensemble_pred)
        
        print(f"\n  Ensemble: RMSE: {ensemble_rmse:.4f} | R²: {ensemble_r2:.4f}")
        
        # Save models
        model_path = os.path.join(output_dir, f'{target}.joblib')
        joblib.dump(models, model_path)
        print(f"  ✓ Saved to {model_path}")
        
        results[target] = {'rmse': ensemble_rmse, 'r2': ensemble_r2}
    
    return results

def main():
    print("="*60)
    print("AIR QUALITY FORECASTING - MODEL TRAINING")
    print("="*60)
    
    # Create directories
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('outputs/models', exist_ok=True)
    
    # Check for processed data
    processed_path = 'data/processed/processed_data.csv'
    
    if not os.path.exists(processed_path):
        print("\n⚠ Processing raw data...")
        
        raw_path = 'data/raw/AirQualityUCI.csv'
        if not os.path.exists(raw_path):
            print(f"\n✗ Error: {raw_path} not found!")
            print("  Place your CSV in data/raw/ directory")
            return
        
        # Process pipeline
        df = load_raw_data(raw_path)
        df_clean = clean_data(df)
        df_features = create_features(df_clean)
        df_features.to_csv(processed_path, index=False)
        print(f"✓ Saved to {processed_path}")
    else:
        print(f"\n✓ Loading {processed_path}")
        df_features = pd.read_csv(processed_path)
    
    # Prepare data
    target_cols = ['co', 'no2', 'nox', 'benzene']
    missing = [col for col in target_cols if col not in df_features.columns]
    if missing:
        print(f"\n✗ Missing columns: {missing}")
        return
    
    exclude_cols = target_cols + ['DateTime']
    feature_cols = [col for col in df_features.columns if col not in exclude_cols]
    
    X = df_features[feature_cols]
    y = df_features[target_cols]
    
    print(f"\n✓ Features: {X.shape[1]} | Targets: {len(target_cols)}")
    
    # Train
    results = train_ensemble_models(X, y, target_cols)
    
    # Summary
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    for target, metrics in results.items():
        print(f"{target.upper():12} | RMSE: {metrics['rmse']:.4f} | R²: {metrics['r2']:.4f}")
    
    print(f"\n✓ Models saved to outputs/models/")
    print(f"✓ Run dashboard: streamlit run src/dashboard/app.py")

if __name__ == '__main__':
    main()