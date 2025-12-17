import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import settings

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

TARGET_COLUMNS = ["co", "no2", "nox", "benzene"]

# Derived paths from settings
RAW_DATA_PATH: Path = settings.RAW_AIR_QUALITY_CSV
PROCESSED_DATA_PATH: Path = settings.PROCESSED_DATA_DIR / "processed_dataset.csv"
MODEL_OUTPUT_DIR: Path = settings.MODELS_OUTPUT_DIR

# Ensure required directories exist
for d in settings.REQUIRED_DIRS:
    os.makedirs(d, exist_ok=True)


# Import project modules (fail fast with helpful message)
try:
    from src.data_preprocessing.load_data import load_raw_data, save_processed_data
    from src.data_preprocessing.clean_data import clean_data
    from src.data_preprocessing.feature_engineering import create_features
    from src.models.train_ensemble import train_best_models
except Exception as e:
    print(f"✗ FATAL ERROR: Required module import failed: {e}")
    print("  - Ensure you're running from the repository root and 'src' is importable.")
    sys.exit(1)


def main() -> int:
    print("=" * 70)
    print("AIR QUALITY PREDICTION PIPELINE - EXECUTION START")
    print("=" * 70)

    # Create any additional dirs used by this run
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Stage 1: Data preprocessing / loading
    if not PROCESSED_DATA_PATH.exists():
        print("\n--- STAGE 1: Data Processing (raw -> processed) ---")
        try:
            df_raw = load_raw_data(RAW_DATA_PATH)
        except FileNotFoundError:
            print(f"\n✗ FATAL ERROR: Raw data not found at {RAW_DATA_PATH}")
            print("   Pipeline terminated. Please ensure the raw data file is present.")
            return 2
        except Exception as e:
            print(f"\n✗ FATAL ERROR during raw data load: {e}")
            return 2

        # Run cleaning and feature engineering
        df_clean = clean_data(df_raw)
        df_features = create_features(df_clean)

        # Drop helper temporal columns if present
        for col in ["hour", "day_of_week", "month", "day_of_year", "DateTime"]:
            if col in df_features.columns:
                df_features.drop(columns=[col], inplace=True, errors="ignore")

        # Save processed dataset
        try:
            save_processed_data(df_features, str(PROCESSED_DATA_PATH))
        except Exception as e:
            print(f"\n✗ FATAL ERROR saving processed data: {e}")
            return 2
    else:
        print(f"\n--- STAGE 1: Processed data found. Loading from {PROCESSED_DATA_PATH} ---")
        try:
            df_features = pd.read_csv(PROCESSED_DATA_PATH)
        except Exception as e:
            print(f"\n✗ FATAL ERROR loading processed data: {e}")
            return 2

    # Stage 2: Prepare inputs for modeling
    print("\n--- STAGE 2: Data Preparation ---")
    missing_targets = [c for c in TARGET_COLUMNS if c not in df_features.columns]
    if missing_targets:
        print(f"\n✗ FATAL ERROR: Target columns missing after processing: {missing_targets}")
        return 2

    feature_cols = [c for c in df_features.columns if c not in TARGET_COLUMNS]
    X = df_features[feature_cols]
    y = df_features[TARGET_COLUMNS]

    print(f"✓ Model Input Ready. Features (X) shape: {X.shape} | Targets (Y) shape: {y.shape}")

    # Stage 3: Model training & evaluation
    print("\n--- STAGE 3: Model Training ---")
    try:
        # train_best_models expects output_dir as a string path
        results = train_best_models(X, y, TARGET_COLUMNS, str(MODEL_OUTPUT_DIR))
    except Exception as e:
        print(f"\n✗ FATAL ERROR during model training: {e}")
        return 2

    # Stage 4: Summary
    print(f"\n{'=' * 70}")
    print("PIPELINE EXECUTION COMPLETE")
    print(f"{'=' * 70}")
    for target, metrics in results.items():
        print(f"[{target.upper():8}] | Model: {metrics['model']:15} | RMSE: {metrics['rmse']:.4f} | R²: {metrics['r2']:.4f}")

    print(f"\nNext Step: Models saved in {MODEL_OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())