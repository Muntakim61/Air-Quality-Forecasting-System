import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_preprocessing.clean_data import clean_data
from src.data_preprocessing.feature_engineering import create_features
from src.alerts.alert_manager import load_alert_thresholds, evaluate_alerts, save_alerts
REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "outputs" / "models"
ALERTS_CONFIG_PATH = REPO_ROOT / "src" / "config" / "configs" / "alerts.yaml"
ALERTS_OUTPUT_DIR = REPO_ROOT / "outputs" / "alerts"
TARGET_COLUMNS = ["co", "no2", "nox", "benzene"]
@st.cache_resource
def load_models(model_dir: Path | None = None):
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
            pollutant, model_name = parts[0], parts[0] 
        try:
            mdl = joblib.load(file)
            models.setdefault(pollutant, {})[model_name] = mdl 
        except Exception as e:
            print(f"Warning: failed to load model file {file}: {e}", file=sys.stderr)
    return models

def _expected_feature_names(model) -> list | None:
    if hasattr(model, "feature_names_in_"):
        return list(getattr(model, "feature_names_in_"))
    if hasattr(model, "feature_name_"):
        return list(getattr(model, "feature_name_"))
    try:
        if hasattr(model, "booster_"):
            feat = model.booster_.feature_name()
            return list(feat)
    except Exception:
        pass
    return None


def make_predictions(models: dict, X: pd.DataFrame) -> pd.DataFrame:
    
    preds = {}
    for pollutant, model_dict in models.items():
        preds_list = []
        for name, mdl in model_dict.items():
            try:
                expected = _expected_feature_names(mdl)
                if expected is not None:
                    X_model = X.reindex(columns=expected)
                    X_model = X_model.fillna(0).astype(float)
                else:
                    X_model = X.select_dtypes(include=[np.number]).fillna(0).astype(float)
                p = mdl.predict(X_model)
                preds_list.append(np.asarray(p).reshape(-1))
            except Exception as e:
                print(f"Warning: model {name} for {pollutant} failed to predict: {e}", file=sys.stderr)

        if preds_list:
            arr = np.vstack(preds_list)
            with np.errstate(invalid="ignore"):
                avg = np.nanmean(arr, axis=0)
            
            all_nan_mask = np.all(np.isnan(arr), axis=0)
            if all_nan_mask.any():
                avg[all_nan_mask] = np.nan
            preds[pollutant] = avg
    if not preds:
        return pd.DataFrame()
    return pd.DataFrame(preds)

def plot_pollutant_trends(predictions_df, thresholds):
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
        if pollutant in thresholds:
            threshold = thresholds[pollutant]
            fig.add_hline(
                y=threshold['high'],
                line_dash="dash",
                line_color="red",
                annotation_text="High",
                row=pos[0], col=pos[1],
                annotation_position="top left"
            )
            fig.add_hline(
                y=threshold['medium'],
                line_dash="dash",
                line_color="orange",
                annotation_text="Medium",
                row=pos[0], col=pos[1],
                annotation_position="bottom left"
            )
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
            
    .metric-card-co { border-left: 5px solid #ff4444; }
    .metric-card-no2 { border-left: 5px solid #ffaa00; }
    .metric-card-nox { border-left: 5px solid #2ca02c; }
    .metric-card-benzene { border-left: 5px solid #764ba2; }

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
    layout="centered",
    initial_sidebar_state="expanded"
)

def display_alerts(alerts):
    """Display alerts using custom CSS classes"""
    st.subheader(f"Generated Alerts ({len(alerts)})")
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
    st.markdown('<div class="main-header">Air Quality Forecasting & Alert System</div>', 
                unsafe_allow_html=True)
    
    st.sidebar.title("System Settings")
    st.sidebar.markdown("---")

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

    st.sidebar.success(f"{sum(len(v) for v in models.values())} model files loaded")
    st.sidebar.markdown(f"**Pollutants:** {', '.join([m.upper() for m in models.keys()])}")
    
    thresholds = load_alert_thresholds() 
    print(f"DEBUG: Value of 'thresholds' loaded in app.py: {thresholds}")
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
        - Add at least a few hours of data for meaningful forecasts.
        """)

    if uploaded_file is None:
        st.info("Upload raw dataset CSV to generate predictions.")
        return

    # Load and process data
    try:
        try:
            df_input = pd.read_csv(uploaded_file, sep=None, engine="python")
        except Exception:
            df_input = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return

    # --- NEW: Row Count Validation ---
    MIN_REQUIRED_ROWS = 5 # Set to 5 to satisfy rolling_mean_3 and lag features

    if len(df_input) < MIN_REQUIRED_ROWS:
        st.error(f"**Insufficient Data:** The uploaded file contains only {len(df_input)} rows.")
        st.warning(f"The model requires at least **{MIN_REQUIRED_ROWS} consecutive hours** of data to calculate trends, rolling averages, and lag features accurately. Please upload a larger dataset.")
        st.stop()

    st.success(f"Successfully loaded {len(df_input)} rows. Proceeding with analysis...")
    
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
    
    # # Prediction Section
    # st.subheader("Generate Predictions")
    
    col_pred_left, col_pred_center, col_pred_right = st.columns([1, 2, 1])
    
    with col_pred_center:
        predict_button = st.button(
            "Generate Forecasts",
            type="secondary",
            width='stretch'
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
                
                for t in TARGET_COLUMNS:
                    if t in X.columns:
                        X = X.drop(columns=[t])
                
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
                    fig = plot_pollutant_trends(preds_df, thresholds) 
                    st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("View Pollutant Distribution Analysis (Matplotlib)", expanded=False):
                    st.markdown("### Pollutant Distribution Analysis")
                    
                    fig_dist, axes = plt.subplots(2, 2, figsize=(15, 10))
                    axes = axes.flatten()
                    
                    pollutants_dist = TARGET_COLUMNS
                    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
                    
                    for idx, (pollutant, color) in enumerate(zip(pollutants_dist, colors)):
                        if pollutant not in preds_df.columns:
                             axes[idx].set_title(f'{pollutant.upper()} - Data Missing', fontsize=14)
                             axes[idx].set_xlim(0, 1) 
                             axes[idx].set_ylim(0, 1)
                             continue

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
                    st.pyplot(fig_dist)

                with st.expander("View Pollutant Correlation Matrix (Seaborn)", expanded=False):
                    st.markdown("Pollutant Correlation Matrix")
                    if len(preds_df) <= 1:
                        st.warning("⚠️ Correlation Matrix cannot be generated for a single row of data. Please upload a dataset with multiple time entries to see pollutant relationships.")
                    elif preds_df.nunique().max() <= 1:
                        st.info("ℹ️ Correlation cannot be calculated because the predicted values are constant (no variation) for this batch.")
                    else:
                        try:
                            fig_corr, ax = plt.subplots(figsize=(10, 8))
                            corr_matrix = preds_df.corr()
                            if corr_matrix.isnull().all().all():
                                st.error("Unable to calculate correlation: The data contains too many missing values.")
                            else:
                                sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                                            center=0, square=True, linewidths=1, ax=ax,
                                            cbar_kws={"shrink": 0.8})
                                ax.set_title('Correlation Between Predicted Pollutants', 
                                            fontsize=16, fontweight='bold')
                                st.pyplot(fig_corr)
                        except Exception as e:
                            st.error(f"Could not generate correlation matrix: {e}")
                
                st.markdown("---")
                st.subheader("Download Results")
                
                col1_dl, col2_dl = st.columns(2) 

                with col1_dl:
                    csv = preds_df.to_csv(index=False)
                    st.download_button(
                        label="Download Predictions CSV",
                        data=csv,
                        file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        width='stretch'
                    )
                
                with col2_dl:
                    if 'DateTime' in df_features.columns:
                        full_results = pd.concat([
                            df_features[['DateTime']].reset_index(drop=True),
                            preds_df.reset_index(drop=True)
                        ], axis=1)
                        csv_full = full_results.to_csv(index=False)
                        st.download_button(
                            label="Download Full Results with Timestamps",
                            data=csv_full,
                            file_name=f"full_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    else:
                        st.info("Timestamped results not available (DateTime column missing).")
                
                # --- END OF NEW VISUALIZATIONS AND DOWNLOAD SECTIONS ---

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")


if __name__ == "__main__":
    main()