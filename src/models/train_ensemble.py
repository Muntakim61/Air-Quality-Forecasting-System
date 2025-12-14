import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import joblib
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data_preprocessing.load_data import load_raw_data, save_processed_data
from src.data_preprocessing.clean_data import clean_data
from src.data_preprocessing.feature_engineering import create_features

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def train_ensemble_models(X, y, target_names, output_dir='outputs/models'):
    """Train best model per pollutant based on prior evaluation"""
    os.makedirs(output_dir, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    results = {}

    # Final model mapping based on validated results
    best_models = {
        'co': RandomForestRegressor(
            n_estimators=300,
            max_depth=15,
            random_state=RANDOM_SEED,
            n_jobs=-1
        ),
        'no2': LGBMRegressor(
            n_estimators=450,
            learning_rate=0.05,
            max_depth=-1,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbose=-1
        ),
        'nox': LGBMRegressor(
            n_estimators=450,
            learning_rate=0.05,
            max_depth=-1,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbose=-1
        ),
        'benzene': RandomForestRegressor(
            n_estimators=300,
            max_depth=15,
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
    }

    for i, target in enumerate(target_names):
        print(f"\n{'='*60}")
        print(f"Training model for: {target.upper()}")
        print(f"{'='*60}")

        y_train_target = y_train.iloc[:, i]
        y_test_target = y_test.iloc[:, i]

        model = best_models[target]

        print(f"\n  Training {model.__class__.__name__}...")
        model.fit(X_train, y_train_target)

        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test_target, y_pred))
        r2 = r2_score(y_test_target, y_pred)

        print(f"    RMSE: {rmse:.4f}")
        print(f"    R²:   {r2:.4f}")

        model_path = os.path.join(output_dir, f'{target}.joblib')
        joblib.dump(model, model_path)

        print(f"\n  ✓ Saved model to {model_path}")

        results[target] = {
            'rmse': rmse,
            'r2': r2
        }

    return results


def main():
    """Main training pipeline"""
    print("="*60)
    print("AIR QUALITY FORECASTING - MODEL TRAINING")
    print("="*60)
    
    # Check if processed data exists
    processed_path = 'data/processed/processed_data.csv'
    
    if not os.path.exists(processed_path):
        print(f"\n⚠ Processed data not found. Running data processing pipeline...")
        
        # Load raw data
        raw_path = 'data/raw/AirQualityUCI.csv'
        if not os.path.exists(raw_path):
            print(f"\n✗ Error: Raw data not found at {raw_path}")
            print("  Please place your data file in data/raw/ directory.")
            return
        
        df = load_raw_data(raw_path)
        df_clean = clean_data(df)
        df_features = create_features(df_clean)
        save_processed_data(df_features, processed_path)
    else:
        print(f"\n✓ Loading processed data from {processed_path}")
        df_features = pd.read_csv(processed_path)
    
    print(f"✓ Loaded {len(df_features)} rows")
    
    # Define target pollutants
    target_cols = ['co', 'no2', 'nox', 'benzene']
    
    # Verify targets exist
    missing_targets = [col for col in target_cols if col not in df_features.columns]
    if missing_targets:
        print(f"\n✗ Error: Missing target columns: {missing_targets}")
        return
    
    # Prepare features and targets
    exclude_cols = target_cols + ['DateTime']
    feature_cols = [col for col in df_features.columns if col not in exclude_cols]
    
    X = df_features[feature_cols]
    y = df_features[target_cols]
    
    print(f"\n✓ Features: {X.shape[1]} columns")
    print(f"✓ Targets: {len(target_cols)} pollutants")
    
    # Train models
    results = train_ensemble_models(X, y, target_cols)
    
    # Summary
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE - SUMMARY")
    print(f"{'='*60}")
    for target, metrics in results.items():
        print(f"{target.upper():12} | RMSE: {metrics['rmse']:.4f} | R²: {metrics['r2']:.4f}")
    
    print(f"\n✓ All models saved to outputs/models/")
    print(f"✓ Ready to run dashboard: streamlit run src/dashboard/app.py")

if __name__ == '__main__':
    main()