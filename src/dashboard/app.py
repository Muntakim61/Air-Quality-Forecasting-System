import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml
import json

# Inline function definitions to avoid import issues
def clean_data(df):
    """Clean data"""
    df_clean = df.copy()
    
    if 'Date' in df_clean.columns and 'Time' in df_clean.columns:
        
        # --- FIX 1: Replace dots with colons in the Time column ---
        # The time format in the source data is 18.00.00 (dots), but Pandas expects 18:00:00 (colons)
        df_clean['Time'] = df_clean['Time'].astype(str).str.replace('.', ':', regex=False)
        
        datetime_series = df_clean['Date'].astype(str) + ' ' + df_clean['Time'].astype(str)
        
        # --- FIX 2: Specify the exact datetime format ---
        # The original dataset uses DD/MM/YYYY, and now the time uses colons.
        try:
            df_clean['DateTime'] = pd.to_datetime(
                datetime_series, 
                format='%d/%m/%Y %H:%M:%S', # Explicitly define the DD/MM/YYYY H:M:S format
                errors='coerce' # Set invalid dates/times to NaT
            )
        except Exception as e:
            # Fallback for unexpected format errors
            print(f"Failed to parse DateTime with explicit format. Error: {e}", file=sys.stderr)
            df_clean['DateTime'] = pd.to_datetime(datetime_series, errors='coerce')
        
        df_clean.drop(['Date', 'Time'], axis=1, inplace=True)

    df_clean.replace(-200, np.nan, inplace=True)
    
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].ffill().bfill()
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
    
    df_clean.drop_duplicates(inplace=True)
    
    target_cols = ['CO(GT)', 'NOx(GT)', 'NO2(GT)', 'C6H6(GT)']
    for col in target_cols:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 3 * IQR
            upper = Q3 + 3 * IQR
            df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
    
    return df_clean

def create_features(df):
    """Create features"""
    df_features = df.copy()
    
    if 'DateTime' in df_features.columns:
        df_features['DateTime'] = pd.to_datetime(df_features['DateTime'])
        df_features['hour'] = df_features['DateTime'].dt.hour
        df_features['day_of_week'] = df_features['DateTime'].dt.dayofweek
        df_features['month'] = df_features['DateTime'].dt.month
        df_features['day_of_year'] = df_features['DateTime'].dt.dayofyear
        df_features['is_weekend'] = df_features['day_of_week'].isin([5, 6]).astype(int)
        
        df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
        df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)
        df_features['dow_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
        df_features['dow_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)
        df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
        df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
    
    sensor_cols = [col for col in df_features.columns if 'PT08' in col]
    for col in sensor_cols:
        df_features[f'{col}_lag1'] = df_features[col].shift(1)
        df_features[f'{col}_lag2'] = df_features[col].shift(2)
        df_features[f'{col}_rolling_mean_3'] = df_features[col].rolling(window=3, min_periods=1).mean()
    
    if 'T' in df_features.columns and 'RH' in df_features.columns:
        df_features['T_RH_interaction'] = df_features['T'] * df_features['RH']
        df_features['T_squared'] = df_features['T'] ** 2
    
    df_features = df_features.bfill()
    
    rename_map = {
        'CO(GT)': 'co',
        'NOx(GT)': 'nox',
        'NO2(GT)': 'no2',
        'C6H6(GT)': 'benzene'
    }
    df_features.rename(columns=rename_map, inplace=True)
    
    return df_features



def load_alert_thresholds(): # No config_path argument needed in call
    """Load alert thresholds, returning defaults if load fails."""
    
    # 1. Calculate the absolute path relative to the app.py file
    script_dir = os.path.dirname(os.path.abspath(__file__)) 
    config_path = os.path.join(script_dir, '..', 'config', 'configs', 'alerts.yaml')
    
    # 2. Define the COMPLETE and mandatory default thresholds
    DEFAULT_THRESHOLDS = {
        'co':      {'low': 4.0, 'medium': 7.0, 'high': 10.0, 'unit': 'mg/m³', 'description': 'Carbon Monoxide'},
        'no2':     {'low': 100, 'medium': 150, 'high': 200, 'unit': 'µg/m³', 'description': 'Nitrogen Dioxide'},
        'nox':     {'low': 150, 'medium': 250, 'high': 350, 'unit': 'µg/m³', 'description': 'Nitrogen Oxides'},
        'benzene': {'low': 1.5, 'medium': 3.0, 'high': 5.0, 'unit': 'µg/m³', 'description': 'Benzene'}
    }
    
    try:
        # 3. Attempt to load using the absolute path
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if config and 'thresholds' in config:
             # Ensure the loaded config is itself a dictionary
             if isinstance(config['thresholds'], dict):
                 return config['thresholds']
             else:
                 print("!!! YAML LOADED BUT 'thresholds' IS NOT A DICTIONARY. FALLING BACK !!!", file=sys.stderr)
                 return DEFAULT_THRESHOLDS
        else:
             print("!!! YAML LOADED BUT MISSING 'thresholds' KEY. FALLING BACK !!!", file=sys.stderr)
             return DEFAULT_THRESHOLDS
             
    except Exception as e:
        # 4. Fallback is guaranteed here for File Not Found or YAML parsing errors.
        print(f"!!! FILE LOAD FAILED: {config_path}. FALLING BACK TO DEFAULTS. Error: {e} !!!", file=sys.stderr)
        return DEFAULT_THRESHOLDS
    

def evaluate_alerts(predictions_df, thresholds):
    """Evaluate alerts"""
    alerts = []
    
    for idx, row in predictions_df.iterrows():
        for pollutant, value in row.items():
            if pollutant in thresholds:
                threshold = thresholds[pollutant]
                
                if value >= threshold['high']:
                    severity = 'high'
                    message = f"🔴 CRITICAL: {pollutant.upper()} level {value:.2f} exceeds high threshold ({threshold['high']})"
                elif value >= threshold['medium']:
                    severity = 'medium'
                    message = f"🟠 WARNING: {pollutant.upper()} level {value:.2f} exceeds medium threshold ({threshold['medium']})"
                elif value >= threshold['low']:
                    severity = 'low'
                    message = f"🟡 ADVISORY: {pollutant.upper()} level {value:.2f} exceeds low threshold ({threshold['low']})"
                else:
                    continue
                
                alerts.append({
                    'timestamp': datetime.now().isoformat(),
                    'index': int(idx),
                    'pollutant': pollutant,
                    'value': float(value),
                    'severity': severity,
                    'message': message
                })
    
    return alerts

def save_alerts(alerts, output_dir='outputs/alerts'):
    """Save alerts"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_path = os.path.join(output_dir, f'alerts_{timestamp}.json')
    
    try:
        with open(file_path, 'w') as f:
            json.dump(alerts, f, indent=2)
        return file_path
    except:
        return None

# Page config
st.set_page_config(
    page_title="Air Quality Forecasting System",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    
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
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
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
    
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load trained models"""
    models = {}
    model_dir = 'outputs/models'
    target_pollutants = ['co', 'no2', 'nox', 'benzene']
    
    if not os.path.exists(model_dir):
        return models
    
    for pollutant in target_pollutants:
        model_path = os.path.join(model_dir, f'{pollutant}.joblib')
        if os.path.exists(model_path):
            try:
                models[pollutant] = joblib.load(model_path)
            except Exception as e:
                st.error(f"Error loading model for {pollutant}: {e}")
    
    return models

def prepare_features(df):
    """Prepare features from uploaded data matching training pipeline"""
    # Clean data
    df_clean = clean_data(df)
    
    # Create features
    df_features = create_features(df_clean)
    
    return df_features

def make_predictions(models, X):
    """Generate ensemble predictions"""
    predictions = {}
    
    for pollutant, model_dict in models.items():
        # Get predictions from each model in ensemble
        preds = []
        for model_name, model in model_dict.items():
            pred = model.predict(X)
            preds.append(pred)
        
        # Ensemble average
        predictions[pollutant] = np.mean(preds, axis=0)
    
    return pd.DataFrame(predictions)

def plot_pollutant_trends(predictions_df, thresholds):
    """Create interactive plotly charts for pollutants"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('CO (Carbon Monoxide)', 'NO2 (Nitrogen Dioxide)', 
                       'NOx (Nitrogen Oxides)', 'Benzene'),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    pollutants = ['co', 'no2', 'nox', 'benzene']
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
                row=pos[0], col=pos[1]
            )
            
            # Medium threshold
            fig.add_hline(
                y=threshold['medium'],
                line_dash="dash",
                line_color="orange",
                annotation_text="Medium",
                row=pos[0], col=pos[1]
            )
            
            # Low threshold
            fig.add_hline(
                y=threshold['low'],
                line_dash="dash",
                line_color="yellow",
                annotation_text="Low",
                row=pos[0], col=pos[1]
            )
    
    fig.update_layout(
        height=700,
        showlegend=False,
        title_text="Pollutant Forecasts with Alert Thresholds",
        title_font_size=20
    )
    
    fig.update_xaxes(title_text="Sample Index")
    fig.update_yaxes(title_text="Concentration")
    
    return fig

def main():
    # Header
    st.markdown('🌍 Air Quality Forecasting & Alert System', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ System Settings")
    st.sidebar.markdown("---")
    
    # Check if models exist
    models = load_models()
    
    if len(models) == 0:
        st.error("⚠️ **No trained models found!**")
        st.info("""
        ### To get started:
        1. Place your raw data in `data/raw/AirQualityUCI.csv`
        2. Run the training script:
        ```bash
        python src/models/train_ensemble.py
        ```
        3. Refresh this dashboard
        """)
        return
    
    st.sidebar.success(f"✅ {len(models)} models loaded successfully")
    st.sidebar.markdown(f"**Models:** {', '.join([m.upper() for m in models.keys()])}")
    
    # Load thresholds
    thresholds = load_alert_thresholds()
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Alert Thresholds")
    
    for pollutant, threshold in thresholds.items():
        with st.sidebar.expander(f"{pollutant.upper()}"):
            st.write(f"**Description:** {threshold.get('description', 'N/A')}")
            st.write(f"**Unit:** {threshold.get('unit', 'N/A')}")
            st.write(f"🟢 Low: {threshold['low']}")
            st.write(f"🟡 Medium: {threshold['medium']}")
            st.write(f"🔴 High: {threshold['high']}")
    
    # Main content
    st.markdown("---")
    
    # File upload section
    st.subheader("📂 Upload Air Quality Data")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload CSV file with air quality measurements",
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
    
    if uploaded_file is not None:
        try:
            # Load data
            df_input = pd.read_csv(uploaded_file, sep=';')
            
            st.success(f"✅ Successfully loaded {len(df_input)} rows")
            
            # Show data preview
            with st.expander("📊 Data Preview (First 10 Rows)", expanded=True):
                st.dataframe(df_input.head(10), use_container_width=True)
            
            # Data info
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Rows", len(df_input))
            col2.metric("Total Columns", len(df_input.columns))
            col3.metric("Memory Usage", f"{df_input.memory_usage(deep=True).sum() / 1024:.2f} KB")
            
            st.markdown("---")
            
            # Prediction section
            st.subheader("🔮 Generate Predictions")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                predict_button = st.button(
                    "🚀 Generate Forecasts",
                    type="primary",
                    use_container_width=True
                )
            
            if predict_button:
                with st.spinner("🔄 Processing data and generating predictions..."):
                    try:
                        # Prepare features
                        df_features = prepare_features(df_input)
                        
                        # Get feature columns (exclude targets and datetime)
                        target_cols = ['co', 'no2', 'nox', 'benzene']
                        exclude_cols = target_cols + ['DateTime']
                        feature_cols = [col for col in df_features.columns if col not in exclude_cols]
                        
                        X = df_features[feature_cols]
                        
                        # Make predictions
                        predictions = make_predictions(models, X)
                        
                        # Store in session state
                        st.session_state['predictions'] = predictions
                        st.session_state['df_features'] = df_features
                        
                        st.success("✅ Predictions generated successfully!")
                        
                    except Exception as e:
                        st.error(f"❌ Error during prediction: {str(e)}")
                        st.exception(e)
            
            # Display results if predictions exist
            if 'predictions' in st.session_state:
                predictions = st.session_state['predictions']
                df_features = st.session_state['df_features']
                
                st.markdown("---")
                st.subheader("📈 Forecast Results")
                
                # Summary statistics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("🔴 CO", f"{predictions['co'].mean():.2f}", 
                             delta=f"{predictions['co'].std():.2f} std")
                with col2:
                    st.metric("🟠 NO2", f"{predictions['no2'].mean():.2f}",
                             delta=f"{predictions['no2'].std():.2f} std")
                with col3:
                    st.metric("🟢 NOx", f"{predictions['nox'].mean():.2f}",
                             delta=f"{predictions['nox'].std():.2f} std")
                with col4:
                    st.metric("🟣 Benzene", f"{predictions['benzene'].mean():.2f}",
                             delta=f"{predictions['benzene'].std():.2f} std")
                
                # Predictions table
                st.markdown("### 📋 Detailed Predictions")
                
                # Combine with datetime if available
                if 'DateTime' in df_features.columns:
                    display_df = pd.concat([
                        df_features[['DateTime']].reset_index(drop=True),
                        predictions.reset_index(drop=True)
                    ], axis=1)
                else:
                    display_df = predictions.copy()
                    display_df.insert(0, 'Index', range(len(display_df)))
                
                st.dataframe(
                    display_df.head(50).style.format({
                        'co': '{:.3f}',
                        'no2': '{:.3f}',
                        'nox': '{:.3f}',
                        'benzene': '{:.3f}'
                    }),
                    use_container_width=True,
                    height=400
                )
                
                # Alert evaluation
                st.markdown("---")
                st.subheader("⚠️ Alert Evaluation")
                
                alerts = evaluate_alerts(predictions, thresholds)
                
                if alerts:
                    # Alert summary
                    alert_counts = {'high': 0, 'medium': 0, 'low': 0}
                    for alert in alerts:
                        alert_counts[alert['severity']] += 1
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"""
                        
                            {alert_counts['high']}
                            🔴 Critical Alerts
                        
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        
                            {alert_counts['medium']}
                            🟠 Warning Alerts
                        
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        
                            {alert_counts['low']}
                            🟡 Advisory Alerts
                        
                        """, unsafe_allow_html=True)
                    
                    st.markdown("### 📢 Alert Messages")
                    
                    # Filter alerts by severity
                    severity_filter = st.selectbox(
                        "Filter by severity:",
                        options=['All', 'High', 'Medium', 'Low']
                    )
                    
                    filtered_alerts = alerts
                    if severity_filter != 'All':
                        filtered_alerts = [a for a in alerts if a['severity'] == severity_filter.lower()]
                    
                    # Display alerts
                    for alert in filtered_alerts[:20]:  # Show first 20
                        severity_class = f"alert-{alert['severity']}"
                        st.markdown(
                            f'{alert["message"]}',
                            unsafe_allow_html=True
                        )
                    
                    if len(filtered_alerts) > 20:
                        st.info(f"Showing 20 of {len(filtered_alerts)} alerts")
                    
                    # Save alerts
                    if st.button("💾 Save Alerts to File"):
                        alert_file = save_alerts(alerts)
                        if alert_file:
                            st.success(f"✅ Alerts saved to: `{alert_file}`")
                
                else:
                    st.success("✅ **All Clear!** No alerts detected - all pollutant levels are within safe limits.")
                
                # Visualizations
                st.markdown("---")
                st.subheader("📊 Interactive Visualizations")
                
                # Plotly interactive chart
                fig = plot_pollutant_trends(predictions, thresholds)
                st.plotly_chart(fig, use_container_width=True)
                
                # Distribution plots
                st.markdown("### 📉 Pollutant Distribution Analysis")
                
                fig_dist, axes = plt.subplots(2, 2, figsize=(15, 10))
                axes = axes.flatten()
                
                pollutants = ['co', 'no2', 'nox', 'benzene']
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
                
                for idx, (pollutant, color) in enumerate(zip(pollutants, colors)):
                    # Histogram
                    axes[idx].hist(predictions[pollutant], bins=30, alpha=0.7, 
                                  color=color, edgecolor='black')
                    axes[idx].set_title(f'{pollutant.upper()} Distribution', 
                                       fontsize=14, fontweight='bold')
                    axes[idx].set_xlabel('Concentration')
                    axes[idx].set_ylabel('Frequency')
                    axes[idx].grid(True, alpha=0.3)
                    
                    # Add mean line
                    mean_val = predictions[pollutant].mean()
                    axes[idx].axvline(mean_val, color='red', linestyle='--', 
                                     linewidth=2, label=f'Mean: {mean_val:.2f}')
                    axes[idx].legend()
                
                plt.tight_layout()
                st.pyplot(fig_dist)
                
                # Correlation heatmap
                st.markdown("### 🔥 Pollutant Correlation Matrix")
                
                fig_corr, ax = plt.subplots(figsize=(10, 8))
                corr_matrix = predictions.corr()
                sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                           center=0, square=True, linewidths=1, ax=ax,
                           cbar_kws={"shrink": 0.8})
                ax.set_title('Correlation Between Predicted Pollutants', 
                            fontsize=16, fontweight='bold')
                st.pyplot(fig_corr)
                
                # Download section
                st.markdown("---")
                st.subheader("💾 Download Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Download predictions
                    csv = predictions.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions CSV",
                        data=csv,
                        file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    # Download full results with datetime
                    if 'DateTime' in df_features.columns:
                        full_results = pd.concat([
                            df_features[['DateTime']].reset_index(drop=True),
                            predictions.reset_index(drop=True)
                        ], axis=1)
                        csv_full = full_results.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Full Results with Timestamps",
                            data=csv_full,
                            file_name=f"full_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
        
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            st.exception(e)
    
    else:
        # Show welcome message when no file is uploaded
        st.info("""
        ### 👋 Welcome to the Air Quality Forecasting System!
        
        **How to use:**
        1. Upload a CSV file with air quality data
        2. Click 'Generate Forecasts' to predict pollutant levels
        3. View alerts, charts, and download results
        
        **System Features:**
        - 🤖 Ensemble ML models (RandomForest + XGBoost + LightGBM)
        - 📊 Interactive visualizations
        - ⚠️ Three-tier alert system (Low, Medium, High)
        - 💾 Export predictions and alerts
        
        Upload your data to get started!
        """)

if __name__ == '__main__':
    main()