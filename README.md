# Air Quality Forecasting and Alert System

## Overview

This project is an end-to-end Air Quality Forecasting and Alert System designed to process historical air quality data, train machine learning models, generate pollutant forecasts, trigger alerts based on configurable thresholds, and present results through an interactive Streamlit dashboard.

The system is structured to separate offline model training from online prediction. Models are trained once using historical data and then reused by the dashboard to generate predictions on new, user-uploaded air quality data.

---

## Key Features

* Modular data preprocessing pipeline
* Multi-output regression models for air quality prediction
* Support for multiple machine learning models (RandomForest, XGBoost, LightGBM)
* Configurable alert thresholds with severity levels
* Streamlit dashboard for CSV upload and real-time prediction
* JSON-based alert logging
* Clean and scalable project structure

---

## Predicted Pollutants

The system predicts the following air quality parameters:

* CO
* NO2
* NOx
* Benzene

---

## Project Structure

```
air_quality_project/
│
├── data/
│   ├── raw/                     # Raw input datasets
│   └── processed/               # Cleaned and feature-engineered data
│
├── src/
│   ├── data_processing/         # Data loading, cleaning, feature engineering
│   ├── models/                  # Model training and evaluation scripts
│   ├── alerts/                  # Alert generation logic
│   ├── dashboard/
│   │   └── app.py               # Streamlit dashboard
│   └── config/
│       └── configs/
│           └── alerts.yaml      # Alert thresholds
│
├── outputs/
│   ├── models/                  # Trained models saved as .joblib
│   └── alerts/                  # Generated alert JSON files
│
├── requirements.txt
└── README.md
```

---

## System Workflow

```
Raw Air Quality Data
        |
        v
Data Processing (load, clean, feature engineering)
        |
        v
Processed Dataset
        |
        v
Model Training and Evaluation
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
Predictions generated using trained models
        |
        v
Alerts evaluated using alerts.yaml thresholds
        |
        v
Alerts saved to outputs/alerts/ and displayed to user
```

---

## Installation

### 1. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # Linux or macOS
venv\Scripts\activate      # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Model Training

Model training is performed offline and is not triggered by the dashboard.

Run the training scripts after preparing processed data:

```bash
python src/models/train_models.py
```

This step will:

* Train models for each pollutant
* Evaluate performance using RMSE and R²
* Save trained models to `outputs/models/` as `.joblib` files

These saved models are required for the dashboard to function correctly.

---

## Alert Configuration

Alert thresholds are defined in:

```
src/config/configs/alerts.yaml
```

Example:

```yaml
thresholds:
  co: 10
  no2: 40
  nox: 200
  benzene: 5
```

Severity levels are automatically assigned based on threshold multiples.

---

## Running the Dashboard

Once models are trained, start the Streamlit application:

```bash
streamlit run src/dashboard/app.py
```

---

## Dashboard Usage

1. Upload a CSV file containing new air quality data.
2. Click the **Predict** button.
3. View predicted pollutant concentrations.
4. Inspect generated alerts and severity levels.
5. Alerts are automatically saved to `outputs/alerts/`.

The uploaded CSV should contain feature columns consistent with the training data schema.

---

## Notes

* The dashboard does not train models. It only loads pre-trained models.
* If model files are missing, the dashboard will raise an error.
* Ensure column names in uploaded CSV files match the expected feature set.

---

## Future Enhancements

* Automated model retraining from the dashboard
* Hyperparameter tuning and model selection
* Geospatial visualization of air quality
* Real-time data ingestion via APIs
* User authentication and role-based access

---

## License

This project is intended for academic and educational use. Licensing can be added as needed.

