import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import TimeSeriesSplit 
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor 

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def train_best_models(X: pd.DataFrame, y: pd.DataFrame, target_names: list, output_dir: str = 'outputs/models'):
    os.makedirs(output_dir, exist_ok=True)
    tscv = TimeSeriesSplit(n_splits=5)
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    results = {}

    best_models = {
        'co': LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbose=-1
        ),
        'no2': RandomForestRegressor(
            n_estimators=300,
            max_depth=15,
            random_state=RANDOM_SEED,
            n_jobs=-1
        ),
        'nox': XGBRegressor(
            n_estimators=400,
            max_depth=8,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_SEED,
            n_jobs=-1
        ),
        'benzene': RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=2,
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
    }

    for i, target in enumerate(target_names):
        print(f"\n{'='*60}")
        print(f"Training final model for: {target.upper()}")
        print(f"{'='*60}")
        y_train_target = y_train.iloc[:, i]
        y_test_target = y_test.iloc[:, i]
        model = best_models[target]
        model_name = model.__class__.__name__
        print(f"  > Model: {model_name}...")
        model.fit(X_train, y_train_target)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test_target, y_pred))
        r2 = r2_score(y_test_target, y_pred)
        print(f"    - Metrics (Final Time-Series Fold):")
        print(f"    - RMSE: {rmse:.4f}")
        print(f"    - R²:   {r2:.4f}")
        model_path = os.path.join(output_dir, f'{target}_{model_name}.joblib')
        joblib.dump(model, model_path)
        print(f"Saved model to {model_path}")
        results[target] = {
            'model': model_name,
            'rmse': rmse,
            'r2': r2
        }
    return results