"""
Central path settings for the Air-Quality-Forecasting-System.

Place this file at: src/config/settings.py

All paths are Path objects and are computed relative to this file's location
so they can be imported and used directly from `train_models.py` or other modules.
"""
from pathlib import Path

# Repository layout assumptions:
# - this file will live at: <repo>/src/config/settings.py
# - repo root is therefore two parents above this file
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

# Top-level dirs
SRC_DIR: Path = REPO_ROOT / "src"
DATA_DIR: Path = REPO_ROOT / "data"
OUTPUTS_DIR: Path = REPO_ROOT / "outputs"

# Raw and processed data
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
RAW_AIR_QUALITY_CSV: Path = RAW_DATA_DIR / "AirQualityUCI.csv"

# Configs
CONFIG_DIR: Path = SRC_DIR / "config"
CONFIGS_SUBDIR: Path = CONFIG_DIR / "configs"
ALERTS_CONFIG_YAML: Path = CONFIGS_SUBDIR / "alerts.yaml"
FEATURES_CONFIG_YAML: Path = CONFIGS_SUBDIR / "features.yaml"
MODELS_CONFIG_YAML: Path = CONFIGS_SUBDIR / "models.yaml"

# Source modules / scripts
# Note: train_models.py at repo root and model training scripts under src/models
TRAIN_MODELS_SCRIPT: Path = REPO_ROOT / "train_models.py"
SRC_MODELS_DIR: Path = SRC_DIR / "models"
TRAIN_ENSEMBLE_SCRIPT: Path = SRC_MODELS_DIR / "train_ensemble.py"

# Outputs (predictions, model artifacts, alerts, visualizations)
FORECASTS_DIR: Path = OUTPUTS_DIR / "forecasts"
MODELS_OUTPUT_DIR: Path = OUTPUTS_DIR / "models"
ALERTS_OUTPUT_DIR: Path = OUTPUTS_DIR / "alerts"
VISUALIZATIONS_DIR: Path = OUTPUTS_DIR / "visualizations"

# Dashboard / app
DASHBOARD_APP: Path = SRC_DIR / "dashboard" / "app.py"

# Data preprocessing utilities
DATA_PREPROCESSING_DIR: Path = SRC_DIR / "data_preprocessing"
CLEAN_DATA_SCRIPT: Path = DATA_PREPROCESSING_DIR / "clean_data.py"
FEATURE_ENGINEERING_SCRIPT: Path = DATA_PREPROCESSING_DIR / "feature_engineering.py"
LOAD_DATA_SCRIPT: Path = DATA_PREPROCESSING_DIR / "load_data.py"

# Other helpers
ALERTS_PACKAGE_DIR: Path = SRC_DIR / "alerts"
ALERT_MANAGER_SCRIPT: Path = ALERTS_PACKAGE_DIR / "alert_manager.py"

# Convenience: list of directories that should exist at runtime
REQUIRED_DIRS = [
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    OUTPUTS_DIR,
    FORECASTS_DIR,
    MODELS_OUTPUT_DIR,
    ALERTS_OUTPUT_DIR,
    VISUALIZATIONS_DIR,
]

# Example usage:
# from src.config import settings
# df = pd.read_csv(settings.RAW_AIR_QUALITY_CSV)