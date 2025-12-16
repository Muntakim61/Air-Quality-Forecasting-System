# Air Quality Forecasting and Alert System

## Overview

This project is an end-to-end Air Quality Forecasting and Alert System that:
- Processes historical air quality data
- Trains multi-output regression models
- Generates pollutant forecasts for new data
- Triggers alerts based on configurable thresholds
- Presents results via an interactive Streamlit dashboard

The codebase separates offline training from online prediction: models are trained once (offline) and reused by the dashboard for inference.

---

## Key Features

- Centralized, shared data preprocessing pipeline (`src/data_preprocessing`)
- Consistent feature engineering used by both training and dashboard
- Multi-output regression for multiple pollutants
- Support for RandomForest, LightGBM (and XGBoost if added)
- Configurable alert thresholds with severity levels (`src/alerts/alert_manager.py`)
- Streamlit dashboard (`src/dashboard/app.py`) for CSV upload, prediction, visualizations, and alerts
- JSON-based alert logging to `outputs/alerts/`
- Clean, modular project structure for maintainability

---

## Predicted Pollutants

The system predicts the following air quality parameters:

- CO
- NO2
- NOx
- Benzene
---

## Project Structure

```
air_quality_project/
│
├── data/
│   ├── raw/            # Raw input datasets
│       └── AirQualityUCI.csv
│   └── processed/      # Cleaned and feature-engineered data
│
├── src/
│   ├── data_preprocessing/   # load_data.py, clean_data.py, feature_engineering.p
│       └── __init__.py
        └── clean_data.py
        └── load_data.py
        └── feature_engineering.py
│   ├── models/               # train_ensemble.py (training & evaluation)
        └── __init__.py
        └── train_ensemble.py
│   ├── alerts/               # alert_manager.py (thresholds + evaluation
│       └── __init__.py
│       └── alert_manager.py
│   ├── dashboard/
│       └── app.py            # (Streamlit dashboard)
│   └── config/
│       └── configs/
│           └── alerts.yaml   # Alert thresholds
│        └── settings.py
├── outputs/
│   ├── models/       # Trained models saved as .joblib
│       └── __init__.py/
│           └── alerts.yaml
│   └── alerts/       # Generated alert JSON files
│
├── requirements.txt
└── README.md
└── train_models.py
```

---

## System Workflow

```
Raw Air Quality Data
    |
    v
Data Processing (src/data_preprocessing/load_data.py,
                 src/data_preprocessing/clean_data.py,
                 src/data_preprocessing/feature_engineering.py)
    |
    v
Processed Dataset
    |
    v
Model Training & Evaluation (src/models/train_ensemble.py)
    |
    v
Trained models saved to outputs/models/
    |
    v
Streamlit Dashboard (src/dashboard/app.py)
    |
    v
User uploads new CSV -> Click Predict
    |
    v
Predictions generated using trained models
    |
    v
Alerts evaluated using src/alerts/alert_manager.py and alerts.yaml
    |
    v
Alerts saved to outputs/alerts/ and displayed in the UI
```

---

## Installation

### 1. Create a virtual environment (recommended)

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Windows (cmd)
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Model Training

Model training is performed offline. Prepare processed data, then run:

```bash
python src/models/train_ensemble.py
```

What this does:
- Trains a final model per pollutant (RandomForest / LightGBM)
- Evaluates using RMSE and R²
- Saves trained model files to `outputs/models/` as `.joblib`

The dashboard requires these saved model files to perform inference.

---

## Alert Configuration

Alert thresholds live at:

```
src/config/configs/alerts.yaml
```

Example structure:

```yaml
thresholds:
  co:
    low: 2.0
    medium: 4.0
    high: 9.0
    unit: "mg/m³"
    description: "Carbon Monoxide"
  no2:
    low: 40
    medium: 100
    high: 200
  nox:
    low: 150
    medium: 250
    high: 350
  benzene:
    low: 1.5
    medium: 3.0
    high: 5.0
  pm2.5:
    low: 25
    medium: 35
    high: 55
  pm10:
    low: 40
    medium: 50
    high: 100
  o3:
    low: 50
    medium: 70
    high: 120
```

`src/alerts/alert_manager.py` provides robust loading and fallback defaults.

---

## Running the Dashboard

After training (and ensuring `outputs/models/` contains model `.joblib` files), start the dashboard:

```bash
python -m streamlit run src/dashboard/app.py
```

---

## Dashboard Usage

1. Upload a CSV file containing raw air quality data (use the same format used for training).
2. Click the **Generate Forecasts** (Predict) button.
3. View predicted pollutant concentrations and severity-labeled alerts.
4. Alerts are saved automatically to `outputs/alerts/`.

Notes:
- Uploaded CSV must contain columns expected by the preprocessing pipeline (`Date`, `Time`, sensor columns, `T`, `RH`, etc.).
- The dashboard imports the shared preprocessing functions from `src/data_preprocessing/` to ensure feature parity with training.

---

## Troubleshooting

- If models are not found, confirm `.joblib` files exist in `outputs/models/` (filenames are like `co_RandomForestRegressor.joblib`).
- If feature mismatch errors occur at prediction time, ensure the uploaded CSV yields the same features as the training pipeline (use the same preprocessing functions).
- For pip install warnings about script locations, prefer using a virtual environment (`.venv`) to isolate installs.

---

## Future Enhancements

- In-dashboard model retraining and model selection UI
- Automated hyperparameter tuning pipelines
- Geospatial visualizations and mapping
- Real-time ingestion (API/webhook) and streaming predictions
- Authentication & role-based access for the dashboard

---

## License

This project is intended for academic and educational use. Add a license file if you intend to redistribute or publish this work.

# Air Quality Forecasting and Alert System

## Overview

This project is an end-to-end Air Quality Forecasting and Alert System designed to process historical air quality data, train machine learning models, generate pollutant forecasts, trigger alerts based on configurable thresholds, and present results through an interactive Streamlit dashboard.

The system is structured to separate offline model training from online prediction. Models are trained once using historical data and then reused by the dashboard to generate predictions on new, user-uploaded air quality data.

---

## Key Features

* **Centralized Data Pipeline:** Robust and modular data preprocessing pipeline (`clean_data`, `create_features`) is shared between training and dashboard to ensure feature consistency.
* Multi-output regression models for air quality prediction
* Support for multiple machine learning models (RandomForest, XGBoost, LightGBM)
* Configurable alert thresholds with severity levels
* **Interactive Streamlit Dashboard:** Supports CSV upload, real-time prediction, **Pollutant Trend Visualizations (Plotly)**, **Distribution Analysis (Matplotlib)**, and **Correlation Heatmaps (Seaborn)**.
* JSON-based alert logging
* Clean and scalable project structure

---

## Predicted Pollutants

The system predicts the following air quality parameters:

* CO (Carbon Monoxide)
* NO2 (Nitrogen Dioxide)
* NOx (Nitrogen Oxides)
* Benzene

---

## Project Structure


air\_quality\_project/
│
├── data/
│   ├── raw/                     \# Raw input datasets
│   └── processed/               \# Cleaned and feature-engineered data
│
├── src/
│   ├── data\_preprocessing/      \# **(NEW)** Centralized data loading, cleaning, feature engineering (Used by training and dashboard)
│   ├── models/                  \# Model training and evaluation scripts (e.g., train\_ensemble.py)
│   ├── alerts/                  \# **(NEW)** Centralized Alert generation logic (e.g., alert\_manager.py)
│   ├── dashboard/
│   │   └── app.py               \# Streamlit dashboard (now imports all utilities from src/ subfolders)
│   └── config/
│       └── configs/
│           └── alerts.yaml      \# Alert thresholds
│
├── outputs/
│   ├── models/                  \# Trained models saved as .joblib
│   └── alerts/                  \# Generated alert JSON files
│
├── requirements.txt
└── README.md

```

---

## System Workflow

The workflow has been updated to emphasize the modularity of the data processing steps.



```

Raw Air Quality Data
|
v
Data Preprocessing (src/data\_preprocessing/clean\_data.py):

  - Clean missing values (-200 -\> NaN, ffill/bfill)
  - **Fix time format (e.g., "18.00.00" -\> "18:00:00")**
  - Remove outliers and duplicates
    |
    v
    Feature Engineering (src/data\_preprocessing/feature\_engineering.py):
  - Create time-based features (hour, day\_of\_week, cyclical encodings)
  - Create lag/rolling mean features for sensors
    |
    v
    Processed Dataset
    |
    v
    Model Training (src/models/train\_ensemble.py)
    |
    v
    Trained Models saved to outputs/models/
    |
    v
    Streamlit Dashboard (app.py)
    |
    v
    User uploads new CSV and clicks Predict
    |
    v
    Predictions generated using imported trained models
    |
    v
    Alerts evaluated using imported logic (src/alerts/alert\_manager.py) and alerts.yaml thresholds
    |
    v
    Alerts saved to outputs/alerts/, displayed to user, and Visualizations generated

<!-- end list -->

````

---

## Installation

### 1. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # Linux or macOS
venv\Scripts\activate      # Windows
````

### 2\. Install dependencies

```bash
pip install -r requirements.txt
```

-----

## Model Training

Model training is performed offline and is not triggered by the dashboard.

Run the training script (ensure the paths are correct for your command, e.g., if you run it from the root):

```bash
python src/models/train_ensemble.py
```

This step will:

  * Train models for each pollutant
  * Evaluate performance using RMSE and R²
  * Save trained models to `outputs/models/` as `.joblib` files

These saved models are required for the dashboard to function correctly.

-----

## Alert Configuration

Alert thresholds are defined in:

```
src/config/configs/alerts.yaml
```

The system uses the `load_alert_thresholds` function from `src/alerts/alert_manager.py` to ensure robust loading, falling back to safe defaults if the file is missing or malformed.

Example (showing pollutant, unit, and default thresholds):

```yaml
thresholds:
  co: 
    low: 2.0
    medium: 4.0
    high: 9.0
    unit: "mg/m³"
    description: "Carbon Monoxide"
  # ... other pollutants follow this structure
```

Severity levels are automatically assigned based on configured `low`, `medium`, and `high` values.

-----

## Running the Dashboard

Once models are trained, start the Streamlit application:

```bash
python -m streamlit run src/dashboard/app.py
```

-----

## Dashboard Usage

1.  Upload a CSV file containing new air quality data.
2.  Click the **Generate Forecasts** button.
3.  View predicted pollutant concentrations and generated alerts.
4.  Expand sections to view **Interactive Trends (Plotly)**, **Distribution Analysis (Matplotlib)**, and **Correlation Heatmaps (Seaborn)**.
5.  Alerts are automatically logged to `outputs/alerts/`.

The uploaded CSV should contain feature columns consistent with the training data schema, including robust `Date` and `Time` columns.

-----

## Notes

  * **Single Source of Truth:** All data cleaning and feature engineering logic is centralized and imported into both training and dashboard scripts for consistency.
  * The dashboard does not train models. It only loads pre-trained models.
  * If model files are missing, the dashboard will display a clear setup error.
  * Ensure column names in uploaded CSV files match the expected feature set.

-----

## Future Enhancements

  * Automated model retraining from the dashboard
  * Hyperparameter tuning and model selection
  * Geospatial visualization of air quality
  * Real-time data ingestion via APIs
  * User authentication and role-based access

-----

## License

This project is intended for academic and educational use. Licensing can be added as needed.

```
```