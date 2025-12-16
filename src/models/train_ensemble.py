# train_ensemble.py
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor # Included for completeness if used later

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def train_best_models(X: pd.DataFrame, y: pd.DataFrame, target_names: list, output_dir: str = 'outputs/models'):
    """
    Objective: Train the final, production-ready model for each pollutant 
               using hyperparameter-tuned settings.
    
    Accountability:
    - Uses the best-performing model (RandomForest/LightGBM) and optimized 
      parameters derived from prior tuning efforts (provided by the user).
    - Splits data for final evaluation and saves the trained model objects.
    
    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.DataFrame): Target matrix.
        target_names (list): List of target column names.
        output_dir (str): Directory to save the trained models.

    Returns:
        dict: Summary of evaluation metrics (RMSE, R²) for each target.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Split Data for Evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    results = {}

    # 2. Define Final Models with Hyperparameter Tuning Results
    # Incorporating FINAL BEST MODEL and its parameters from the tuning output.
    best_models = {
        # CO: Final Best Model: RandomForest (RMSE: 11.8993)
        'co': RandomForestRegressor(
            n_estimators=200,
            max_depth=18,
            min_samples_split=4,
            random_state=RANDOM_SEED,
            n_jobs=-1
        ),
        # NO2: Final Best Model: LightGBM (RMSE: 38.8345)
        'no2': LGBMRegressor(
            n_estimators=450,
            learning_rate=0.062147422437610705,
            num_leaves=31,
            min_data_in_leaf=65,
            feature_fraction=0.6299924747454009,
            bagging_fraction=0.7337498258560773,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbose=-1 # Suppress logging
        ),
        # NOx: Final Best Model: LightGBM (RMSE: 88.1821)
        'nox': LGBMRegressor(
            n_estimators=450,
            learning_rate=0.07322370567394015,
            num_leaves=35,
            min_data_in_leaf=51,
            feature_fraction=0.8976634677873652,
            bagging_fraction=0.6002336297523043,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbose=-1
        ),
        # Benzene: Final Best Model: RandomForest (RMSE: 0.0175)
        'benzene': RandomForestRegressor(
            n_estimators=200,
            max_depth=18,
            min_samples_split=4,
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
    }

    # 3. Training Loop
    for i, target in enumerate(target_names):
        print(f"\n{'='*60}")
        print(f"🎯 Training final model for: {target.upper()}")
        print(f"{'='*60}")

        y_train_target = y_train.iloc[:, i]
        y_test_target = y_test.iloc[:, i]

        model = best_models[target]
        model_name = model.__class__.__name__

        # Training the model
        print(f"  > Model: {model_name}...")
        model.fit(X_train, y_train_target)
        
        # Predicting and Evaluating
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test_target, y_pred))
        r2 = r2_score(y_test_target, y_pred)

        # Log results
        print(f"    - Metrics (Test Set):")
        print(f"    - RMSE: {rmse:.4f}")
        print(f"    - R²:   {r2:.4f}")

        # Saving the model
        model_path = os.path.join(output_dir, f'{target}_{model_name}.joblib')
        joblib.dump(model, model_path)

        print(f"  ✓ Saved model to {model_path}")

        results[target] = {
            'model': model_name,
            'rmse': rmse,
            'r2': r2
        }

    return results