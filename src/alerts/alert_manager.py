import yaml
import json
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# --- Configuration Constants ---
# Resolve the project root directory (assumes script is in a subdirectory like 'src/')
BASE_DIR = Path(__file__).resolve().parent.parent 
DEFAULT_YAML_PATH = BASE_DIR / "config" / "configs" / "alerts.yaml"

def load_alert_thresholds(config_path=None):
    """
    Loads pollutant threshold levels from a YAML configuration file.
    
    Args:
        config_path (str/Path, optional): Path to the YAML config. 
            Defaults to DEFAULT_YAML_PATH.

    Returns:
        dict: A dictionary mapping pollutant keys to their severity thresholds.
              Returns DEFAULT_THRESHOLDS if file is missing or invalid.
    """
    # Hardcoded fallback defaults if config file is unavailable
    DEFAULT_THRESHOLDS = {
        'co': {'low': 2.0, 'medium': 4.0, 'high': 9.0, 'unit': 'mg/m³', 'description': 'Carbon Monoxide'},
        'no2': {'low': 100, 'medium': 200, 'high': 400, 'unit': 'µg/m³', 'description': 'Nitrogen Dioxide'},
        'nox': {'low': 150, 'medium': 300, 'high': 600, 'unit': 'µg/m³', 'description': 'Nitrogen Oxides'},
        'benzene': {'low': 5.0, 'medium': 10.0, 'high': 20.0, 'unit': 'µg/m³', 'description': 'Benzene'}
    }

    if config_path is None:
        config_path = DEFAULT_YAML_PATH
    path_obj = Path(config_path)

    if path_obj.exists():
        try:
            with path_obj.open('r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                # Validation: ensure 'thresholds' key exists and is the correct format
                if isinstance(config, dict) and isinstance(config.get('thresholds'), dict):
                    return config['thresholds']
                print("Warning: 'thresholds' key missing or malformed in YAML.", file=sys.stderr)
        except Exception as e:
            print(f"Error reading alert config: {e}", file=sys.stderr)

    return DEFAULT_THRESHOLDS

def evaluate_alerts(predictions_df, thresholds):
    """
    Analyzes prediction data to identify and group air quality incidents.
    
    This function uses a rolling average to smooth out noise in the sensor data
    before comparing against low, medium, and high thresholds.

    Args:
        predictions_df (pd.DataFrame): DataFrame containing pollutant concentrations.
        thresholds (dict): Threshold definitions from load_alert_thresholds.

    Returns:
        list: A list of incident dictionaries containing duration, severity, and max values.
    """
    alerts = []
    window_size = 3  # Window for simple moving average to prevent alert flickering
    df = predictions_df.copy()

    for pollutant in thresholds:
        if pollutant not in df.columns:
            continue
            
        threshold = thresholds[pollutant]
        # Smooth data: min_periods=1 ensures we get values even at the start of the series
        smoothed_values = df[pollutant].rolling(window=window_size, min_periods=1).mean()
        
        current_incident = None
        
        for idx, value in smoothed_values.items():
            # Determine severity level based on smoothed average
            severity = None
            if value >= threshold['high']:
                severity = 'high'
            elif value >= threshold['medium']:
                severity = 'medium'
            elif value >= threshold['low']:
                severity = 'low'
            
            if severity:
                # Format visual indicators for human-readable messages
                prefix = "🔴 CRITICAL" if severity == 'high' else "🟠 WARNING" if severity == 'medium' else "🟡 ADVISORY"
                msg = f"{prefix}: {threshold['description']} levels are {severity} (Avg: {value:.2f} {threshold['unit']})"
                
                # Logic: If the severity remains the same as the previous point, extend the incident
                if current_incident and current_incident['severity'] == severity:
                    current_incident['end_index'] = str(idx)
                    # Track the actual raw peak (not the smoothed value)
                    current_incident['max_value'] = max(current_incident['max_value'], float(df.loc[idx, pollutant]))
                    current_incident['message'] = msg 
                else:
                    # Severity changed or new incident started: close existing and start new
                    if current_incident:
                        alerts.append(current_incident)
                    
                    current_incident = {
                        'timestamp': datetime.now().isoformat(),
                        'start_index': str(idx),
                        'end_index': str(idx),
                        'pollutant': pollutant,
                        'max_value': float(df.loc[idx, pollutant]),
                        'severity': severity,
                        'message': msg
                    }
            else:
                # Below 'low' threshold: close the current active incident
                if current_incident:
                    alerts.append(current_incident)
                    current_incident = None
        
        # Cleanup: append any incident that was active at the end of the dataframe
        if current_incident:
            alerts.append(current_incident)

    return alerts

def save_alerts(alerts, output_dir='outputs/alerts'):
    """
    Exports the list of detected incidents to a JSON file.

    Args:
        alerts (list): List of incident dictionaries.
        output_dir (str): Directory where the JSON file will be stored.

    Returns:
        str: The full path to the saved file, or None if no alerts were saved.
    """
    if not alerts:
        return None

    # Ensure the output directory exists (prevents FileNotFoundError)
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate unique filename using current timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_path = os.path.join(output_dir, f'alerts_{timestamp}.json')
    
    try:
        with open(file_path, 'w') as f:
            json.dump(alerts, f, indent=2)
        return file_path
    except Exception as e:
        print(f"Error saving alerts: {e}")
        return None