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

from src.data_processing.load_data import load_raw_data, save_processed_data
from src.data_processing.clean_data import clean_data
from src.data_processing.feature_engineering import create_features

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def train_ensemble_models(X, y, target_names, output_dir='outputs/models'):
    """Train ensemble models for each pollutant"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    
    results = {}
    
    for i, target in enumerate(target_names):
        print(f"\n{'='*60}")
        print(f"Training models for: {target.upper()}")
        print(f"{'='*60}")
        
        y_train_target = y_train.iloc[:, i] if isinstance(y_train, pd.DataFrame) else y_train[:, i]
        y_test_target = y_test.iloc[:, i] if isinstance(y_test, pd.DataFrame) else y_test[:, i]
        
        # Initialize models
        models = {
            'RandomForest': RandomForestRegressor(
                n_estimators=100, 
                max_depth=15, 
                random_state=RANDOM_SEED,
                n_jobs=-1
            ),
            'XGBoost': XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=RANDOM_SEED,
                n_jobs=-1
            ),
            'LightGBM': LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=RANDOM_SEED,
                n_jobs=-1,
                verbose=-1
            )
        }
        
        # Train and evaluate each model
        predictions = {}
        for model_name, model in models.items():
            print(f"\n  Training {model_name}...")
            model.fit(X_train, y_train_target)
            y_pred = model.predict(X_test)
            predictions[model_name] = y_pred
            
            rmse = np.sqrt(mean_squared_error(y_test_target, y_pred))
            r2 = r2_score(y_test_target, y_pred)
            
            print(f"    RMSE: {rmse:.4f}")
            print(f"    R²:   {r2:.4f}")
        
        # Create ensemble prediction (average)
        ensemble_pred = np.mean(list(predictions.values()), axis=0)
        ensemble_rmse = np.sqrt(mean_squared_error(y_test_target, ensemble_pred))
        ensemble_r2 = r2_score(y_test_target, ensemble_pred)
        
        print(f"\n  Ensemble Performance:")
        print(f"    RMSE: {ensemble_rmse:.4f}")
        print(f"    R²:   {ensemble_r2:.4f}")
        
        # Save all models for this target
        target_models = {}
        for model_name, model in models.items():
            target_models[model_name] = model
        
        model_path = os.path.join(output_dir, f'{target}.joblib')
        joblib.dump(target_models, model_path)
        print(f"\n  ✓ Saved models to {model_path}")
        
        results[target] = {
            'rmse': ensemble_rmse,
            'r2': ensemble_r2
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