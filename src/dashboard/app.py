# src/dashboard/app.py
import os
import sys
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
import yaml
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================================
# === REFACTORING: IMPORT CENTRALIZED UTILITIES (Copilot Changes) =========
# =========================================================================
# Note: Assuming your project structure aligns with these imports (e.g.,
# src/data_preprocessing/clean_data.py, src/alerts/alert_manager.py, etc.)

# Data Preprocessing
from src.data_preprocessing.clean_data import clean_data
from src.data_preprocessing.feature_engineering import create_features
# Load Data (load_raw_data not strictly needed but included for completeness)
# from src.data_preprocessing.load_data import load_raw_data 

# Alerts Manager
from src.alerts.alert_manager import load_alert_thresholds, evaluate_alerts, save_alerts
# Training module (train_best_models not directly used in app.py but sometimes imported)
# from src.models.train_ensemble import train_best_models 

# =========================================================================
# === END OF REFACTORING CHANGES ==========================================
# =========================================================================


# ---------- Utility: Paths ----------
REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "outputs" / "models"
ALERTS_CONFIG_PATH = REPO_ROOT / "src" / "config" / "configs" / "alerts.yaml"
ALERTS_OUTPUT_DIR = REPO_ROOT / "outputs" / "alerts"

# ---------- CONSTANTS FIX: Define TARGET_COLUMNS here ----------
TARGET_COLUMNS = ["co", "no2", "nox", "benzene"]
# ----------------------------------------------------------------

# =========================================================================
# === REMOVED: Duplicated local `clean_data` and `create_features` =======
# === REMOVED: Duplicated local `load_alert_thresholds` ===================
# === REMOVED: Duplicated local `evaluate_alerts` =========================
# === REMOVED: Duplicated local `save_alerts` =============================
# === Functions are now imported from centralized modules. =================
# =========================================================================


# ---------- Fixed: load_models now finds files created by train_ensemble ----------
@st.cache_resource
def load_models(model_dir: Path | None = None):
    """
    Load trained model files from outputs/models.
    
    Returns: dict[pollutant] -> dict[model_name] = model_object
    """
    models = {}
    md = Path(model_dir) if model_dir else MODEL_DIR
    if not md.exists():
        return models

    for file in md.glob("*.joblib"):
        stem = file.stem 
        parts = stem.split("_", 1)
        if len(parts) == 2:
            pollutant, model_name = parts
        else:
            # Fallback for models without an explicit name (e.g., just 'co.joblib')
            pollutant, model_name = parts[0], parts[0] 
        try:
            mdl = joblib.load(file)
            # Old app.py expects models[pollutant] to be a dictionary of models (ensemble),
            # so we ensure it's a dict even if only one model is loaded.
            models.setdefault(pollutant, {})[model_name] = mdl 
        except Exception as e:
            print(f"Warning: failed to load model file {file}: {e}", file=sys.stderr)

    return models


# ---------- Model prediction helpers (No changes here) ----------
def _expected_feature_names(model) -> list | None:
    
    # sklearn
    if hasattr(model, "feature_names_in_"):
        return list(getattr(model, "feature_names_in_"))
    # lightgbm
    if hasattr(model, "feature_name_"):
        return list(getattr(model, "feature_name_"))
    # lightgbm Booster
    try:
        if hasattr(model, "booster_"):
            feat = model.booster_.feature_name()
            return list(feat)
    except Exception:
        pass
    # fallback: None
    return None


def make_predictions(models: dict, X: pd.DataFrame) -> pd.DataFrame:
    
    preds = {}
    for pollutant, model_dict in models.items():
        preds_list = []
        for name, mdl in model_dict.items():
            try:
                expected = _expected_feature_names(mdl)
                if expected is not None:
                    # Reindex X to match training features: missing -> fill 0, extra -> dropped
                    X_model = X.reindex(columns=expected)
                    # Fill any missing columns with zeros and ensure numeric dtype
                    X_model = X_model.fillna(0).astype(float)
                else:
                    # If we can't get expected names, try to use numeric columns ordered as-is
                    X_model = X.select_dtypes(include=[np.number]).fillna(0).astype(float)

                # Predict
                p = mdl.predict(X_model)
                preds_list.append(np.asarray(p).reshape(-1))
            except Exception as e:
                print(f"Warning: model {name} for {pollutant} failed to predict: {e}", file=sys.stderr)
                # continue to next model

        if preds_list:
            arr = np.vstack(preds_list)  # shape (n_models, n_samples)
            # compute mean across models but avoid warnings when columns are all-NaN
            with np.errstate(invalid="ignore"):
                avg = np.nanmean(arr, axis=0)
            # where all entries are nan, set result to nan explicitly (nanmean gives nan but may warn)
            all_nan_mask = np.all(np.isnan(arr), axis=0)
            if all_nan_mask.any():
                avg[all_nan_mask] = np.nan
            preds[pollutant] = avg
    if not preds:
        return pd.DataFrame()
    return pd.DataFrame(preds)


# --- Plotting function (using the old app.py's detailed version for UI match) ---
def plot_pollutant_trends(predictions_df, thresholds):
    """Create interactive plotly charts for pollutants including thresholds"""
    
    # Ensure all required pollutants are in the predictions_df, filling with NaN if missing
    pollutants = TARGET_COLUMNS
    for p in pollutants:
        if p not in predictions_df.columns:
            predictions_df[p] = np.nan
            
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('CO (Carbon Monoxide)', 'NO2 (Nitrogen Dioxide)', 
                       'NOx (Nitrogen Oxides)', 'Benzene'),
        vertical_spacing=0.20,
        horizontal_spacing=0.1
    )
    
    positions = [(1,1), (1,2), (2,1), (2,2)]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, (pollutant, pos, color) in enumerate(zip(pollutants, positions, colors)):
        # Main prediction line
        fig.add_trace(
            go.Scatter(
                x=list(range(len(predictions_df))),
                y=predictions_df[pollutant],
                name=pollutant.upper(),
                line=dict(color=color, width=2),
                showlegend=True
            ),
            row=pos[0], col=pos[1]
        )
        
        # Threshold lines
        if pollutant in thresholds:
            threshold = thresholds[pollutant]
            
            # High threshold
            fig.add_hline(
                y=threshold['high'],
                line_dash="dash",
                line_color="red",
                annotation_text="High",
                row=pos[0], col=pos[1],
                annotation_position="top left"
            )
            
            # Medium threshold
            fig.add_hline(
                y=threshold['medium'],
                line_dash="dash",
                line_color="orange",
                annotation_text="Medium",
                row=pos[0], col=pos[1],
                annotation_position="bottom left"
            )
            
            # Low threshold
            fig.add_hline(
                y=threshold['low'],
                line_dash="dash",
                line_color="yellow",
                annotation_text="Low",
                row=pos[0], col=pos[1],
                annotation_position="top right"
            )
    
    fig.update_layout(
        height=750,
        showlegend=False,
        title_text="Pollutant Forecasts with Alert Thresholds",
        title_font_size=20
    )
    
    fig.update_xaxes(title_text="Sample Index")
    fig.update_yaxes(title_text="Concentration")
    
    return fig


# ---------- Streamlit UI ----------

# Custom CSS for the old app.py look
st.markdown("""
    <style>
    .main-header {
        font-size: 2.8rem;
        color: #1f77b4;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
    }
    /* === IMPROVED METRIC CARD STYLES === */
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        color: black;
        min-height: 120px; /* Ensure consistent height */
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        
        /* Layout control for responsiveness */
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    
    /* Dynamic Border Colors based on Pollutant */
    .metric-card-co { border-left: 5px solid #ff4444; }
    .metric-card-no2 { border-left: 5px solid #ffaa00; }
    .metric-card-nox { border-left: 5px solid #2ca02c; }
    .metric-card-benzene { border-left: 5px solid #764ba2; }

    /* Internal element styling */
    .metric-card h4 {
        margin-top: 0;
        margin-bottom: 5px; 
        font-size: 1.1rem;
    }

    .metric-card .value {
        font-size: 2.0rem; 
        font-weight: bold;
        margin-bottom: 5px; 
        margin-top: 5px;
    }

    .metric-card .std-dev {
        color: #6c757d;
        margin-top: 0px;
        font-size: 0.9rem;
    }
    /* ================================== */

    .alert-high {
        background-color: #ff4444;
        color: white;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        font-weight: bold;
    }
    .alert-medium {
        background-color: #ffaa00;
        color: white;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        font-weight: bold;
    }
    .alert-low {
        background-color: #44aaff;
        color: white;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        font-weight: bold;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title="Air Quality Forecasting System",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

def display_alerts(alerts):
    """Display alerts using custom CSS classes"""
    st.subheader(f"⚠️ Generated Alerts ({len(alerts)})")
    
    # Use st.expander for a cleaner look
    with st.expander("View Detailed Alerts", expanded=True):
        if not alerts:
            st.info("No elevated alerts were triggered.")
            return

        for alert in alerts:
            severity_class = f"alert-{alert['severity']}"
            st.markdown(
                f'<div class="{severity_class}">{alert["message"]}</div>', 
                unsafe_allow_html=True
            )


def main():
    # Header: Use the custom CSS class 'main-header'
    st.markdown('<div class="main-header">🌍 Air Quality Forecasting & Alert System</div>', 
                unsafe_allow_html=True)
    
    # Sidebar Setup
    st.sidebar.title("System Settings")
    st.sidebar.markdown("---")

    # Load models
    models = load_models()
    
    if not models:
        st.error("**No trained models found!**")
        st.info(f"""
        ### To get started:
        1. Place your raw data in `data/raw/AirQualityUCI.csv`
        2. Run the training script:
        ```bash
        python src/models/train_ensemble.py
        ```
        3. Refresh this dashboard.
        """)
        return
    
    # Sidebar: Model Info
    st.sidebar.success(f"{sum(len(v) for v in models.values())} model files loaded")
    st.sidebar.markdown(f"**Pollutants:** {', '.join([m.upper() for m in models.keys()])}")
    
    # Load thresholds (Now using the imported centralized function)
    # The imported function should handle the ALERTS_CONFIG_PATH logic internally
    thresholds = load_alert_thresholds() 
    print(f"DEBUG: Value of 'thresholds' loaded in app.py: {thresholds}")
    
    # === CRITICAL FIX: Ensure 'thresholds' is a dictionary for the UI loop ===
    if thresholds is None:
        st.error("Error: Could not load alert thresholds from file. Using hardcoded defaults.")
        # Provide a hardcoded dictionary structure to prevent the crash
        thresholds = { 
            'co': {'low': 2.0, 'medium': 4.0, 'high': 9.0, 'unit': 'mg/m³', 'description': 'Carbon Monoxide (Default)'},
            'no2': {'low': 100, 'medium': 200, 'high': 400, 'unit': 'µg/m³', 'description': 'Nitrogen Dioxide (Default)'},
            'nox': {'low': 150, 'medium': 300, 'high': 600, 'unit': 'µg/m³', 'description': 'Nitrogen Oxides (Default)'},
            'benzene': {'low': 5.0, 'medium': 10.0, 'high': 20.0, 'unit': 'µg/m³', 'description': 'Benzene (Default)'}
        }
    # Sidebar: Alert Thresholds
    st.sidebar.markdown("---")
    st.sidebar.subheader("Alert Thresholds")
    for pollutant, threshold in thresholds.items():
        with st.sidebar.expander(f"{pollutant.upper()}"):
            st.write(f"**Description:** {threshold.get('description', 'N/A')}")
            st.write(f"**Unit:** {threshold.get('unit', 'N/A')}")
            st.write(f"🟢 Low: {threshold['low']} {threshold.get('unit', '')}")
            st.write(f"🟡 Medium: {threshold['medium']} {threshold.get('unit', '')}")
            st.write(f"🔴 High: {threshold['high']} {threshold.get('unit', '')}")

    st.markdown("---")

    # Main Content: File Upload and Info
    st.subheader("Upload Air Quality Data")
    
    col1, col2 = st.columns([2, 1])
    uploaded_file = None

    with col1:
        uploaded_file = st.file_uploader(
            "Upload CSV file with raw air quality measurements",
            type=['csv'],
            help="File should contain columns: Date, Time, sensor readings, etc."
        )
    
    with col2:
        st.info("""
        **Required columns:**
        - Date, Time
        - Sensor readings (PT08.S1, etc.)
        - Temperature (T), Humidity (RH)
        """)

    if uploaded_file is None:
        st.info("Upload raw dataset CSV to generate predictions.")
        return

    # Load and process data (now outside the button click for metric display)
    try:
        # Try different separators to match the old app.py robustness (sep=None/python engine, then default)
        try:
            df_input = pd.read_csv(uploaded_file, sep=None, engine="python")
        except Exception:
            df_input = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return

    st.success(f"Successfully loaded {len(df_input)} rows")
    
    # Show data preview
    with st.expander("View Raw Data Preview (First 10 Rows)", expanded=False):
        st.dataframe(df_input.head(10), use_container_width=True)
    
    # Data metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", len(df_input))
    col2.metric("Total Columns", len(df_input.columns))
    # Calculate and display memory usage (matches old app.py)
    mem_usage = df_input.memory_usage(deep=True).sum() / 1024
    col3.metric("Memory Usage", f"{mem_usage:.2f} KB")
    
    st.markdown("---")
    
    # Prediction Section
    st.subheader("Generate Predictions")
    
    col_pred_left, col_pred_center, col_pred_right = st.columns([1, 2, 1])
    
    with col_pred_center:
        predict_button = st.button(
            "Generate Forecasts",
            type="primary",
            use_container_width=True
        )

    # Prediction Logic
    if predict_button:
        with st.spinner("Processing data and generating predictions..."):
            try:
                # 1. Preprocess data (Now using the imported centralized functions)
                df_features = clean_data(df_input)
                df_features = create_features(df_features)

                # 2. Prepare features for prediction
                X = df_features.select_dtypes(include=[np.number]).copy()
                
                # --- FIX: Drop target columns using the now-defined global constant ---
                for t in TARGET_COLUMNS:
                    if t in X.columns:
                        X = X.drop(columns=[t])
                # ---------------------------------------------------------------------
                
                if X.shape[0] == 0:
                    st.error("No numeric feature columns available after preprocessing.")
                    return

                # 3. Make predictions
                preds_df = make_predictions(models, X)
                
                if preds_df.empty:
                    st.error("No predictions produced by loaded models.")
                    return

                # 4. Evaluation and Display
                st.success("Predictions generated")
                
                # Show predictions dataframe
                st.markdown("#### Forecast Data Preview")
                st.dataframe(preds_df.head(20), use_container_width=True)
                
                # 5. Alert Generation and Display (Now using the imported centralized functions)
                alerts = evaluate_alerts(preds_df, thresholds)
                if alerts:
                    # save_alerts is now the imported function
                    out_path = save_alerts(alerts, output_dir=ALERTS_OUTPUT_DIR) 
                    display_alerts(alerts) # Use the custom display function
                else:
                    st.info("No alerts generated for the prediction batch.")

                # --- START OF NEW VISUALIZATIONS AND DOWNLOAD SECTIONS ---
                
                st.markdown("---")
                st.header("Detailed Analysis Features")
                
                # 1. Interactive Visualizations (Plotly Trends)
                with st.expander("View Interactive Pollutant Forecast Trends (Plotly)", expanded=False):
                    st.markdown("### Interactive Visualizations")
                    # Note: Using preds_df
                    fig = plot_pollutant_trends(preds_df, thresholds) 
                    st.plotly_chart(fig, use_container_width=True)
                
                # 2. Distribution plots (Matplotlib)
                with st.expander("View Pollutant Distribution Analysis (Matplotlib)", expanded=False):
                    st.markdown("### Pollutant Distribution Analysis")
                    
                    # Create the figure for Matplotlib/Seaborn
                    fig_dist, axes = plt.subplots(2, 2, figsize=(15, 10))
                    axes = axes.flatten()
                    
                    pollutants_dist = TARGET_COLUMNS
                    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # Consistent colors
                    
                    for idx, (pollutant, color) in enumerate(zip(pollutants_dist, colors)):
                        # Check if pollutant exists in predictions
                        if pollutant not in preds_df.columns:
                             axes[idx].set_title(f'{pollutant.upper()} - Data Missing', fontsize=14)
                             # Set axis limits to ensure blank plots are consistent
                             axes[idx].set_xlim(0, 1) 
                             axes[idx].set_ylim(0, 1)
                             continue
                             
                        # Histogram
                        # Use dropna() for the distribution plot to handle potential NaNs
                        axes[idx].hist(preds_df[pollutant].dropna(), bins=30, alpha=0.7, 
                                       color=color, edgecolor='black')
                        axes[idx].set_title(f'{pollutant.upper()} Distribution', 
                                            fontsize=14, fontweight='bold')
                        axes[idx].set_xlabel('Concentration')
                        axes[idx].set_ylabel('Frequency')
                        axes[idx].grid(True, alpha=0.3)
                        
                        # Add mean line
                        mean_val = preds_df[pollutant].mean()
                        axes[idx].axvline(mean_val, color='red', linestyle='--', 
                                          linewidth=2, label=f'Mean: {mean_val:.2f}')
                        axes[idx].legend()
                    
                    plt.tight_layout()
                    st.pyplot(fig_dist) # Display Matplotlib figure
                
                # 3. Correlation heatmap (Seaborn)
                with st.expander("View Pollutant Correlation Matrix (Seaborn)", expanded=False):
                    st.markdown("### Pollutant Correlation Matrix")
                    
                    fig_corr, ax = plt.subplots(figsize=(10, 8))
                    corr_matrix = preds_df.corr() # Note: Using preds_df
                    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                                center=0, square=True, linewidths=1, ax=ax,
                                cbar_kws={"shrink": 0.8})
                    ax.set_title('Correlation Between Predicted Pollutants', 
                                 fontsize=16, fontweight='bold')
                    st.pyplot(fig_corr) # Display Seaborn plot
                
                # Download section
                st.markdown("---")
                st.subheader("Download Results")
                
                col1_dl, col2_dl = st.columns(2) 

                with col1_dl:
                    # Download predictions
                    csv = preds_df.to_csv(index=False) # Note: Using preds_df
                    st.download_button(
                        label="📥 Download Predictions CSV",
                        data=csv,
                        file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col2_dl:
                    # Download full results with datetime
                    if 'DateTime' in df_features.columns: # df_features still holds the DateTime column
                        full_results = pd.concat([
                            df_features[['DateTime']].reset_index(drop=True),
                            preds_df.reset_index(drop=True)
                        ], axis=1)
                        csv_full = full_results.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Full Results with Timestamps",
                            data=csv_full,
                            file_name=f"full_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    else:
                        st.info("Timestamped results not available (DateTime column missing).")
                
                # --- END OF NEW VISUALIZATIONS AND DOWNLOAD SECTIONS ---

            except Exception as e:
                # Catch any unexpected errors during the prediction path
                st.error(f"An error occurred during prediction: {e}")


if __name__ == "__main__":
    main()