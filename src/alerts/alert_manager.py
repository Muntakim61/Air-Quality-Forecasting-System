import yaml
import json
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent 
DEFAULT_YAML_PATH = BASE_DIR / "config" / "configs" / "alerts.yaml"

def load_alert_thresholds(config_path=None):
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
                if isinstance(config, dict) and isinstance(config.get('thresholds'), dict):
                    return config['thresholds']
                print("Warning: 'thresholds' key missing or malformed in YAML.", file=sys.stderr)
        except Exception as e:
            print(f"Error reading alert config: {e}", file=sys.stderr)

    return DEFAULT_THRESHOLDS

def evaluate_alerts(predictions_df, thresholds):
    """
    Evaluates alerts using Incident-Based logic:
    1. Uses a rolling average to prevent 'flicker' alerts.
    2. Groups consecutive violations into a single 'Incident'.
    """
    alerts = []
    # Use a 3-hour window for operational stability
    window_size = 3 
    
    # Ensure we are working with a numeric copy
    df = predictions_df.copy()

    for pollutant in thresholds:
        if pollutant not in df.columns:
            continue
            
        threshold = thresholds[pollutant]
        
        # --- Step 1: Temporal Smoothing ---
        # We calculate a rolling mean so one single spike doesn't panic the user.
        # This represents "Sustained Exposure" which is what humans care about.
        smoothed_values = df[pollutant].rolling(window=window_size, min_periods=1).mean()
        
        current_incident = None
        
        for idx, value in smoothed_values.items():
            severity = None
            if value >= threshold['high']:
                severity = 'high'
            elif value >= threshold['medium']:
                severity = 'medium'
            elif value >= threshold['low']:
                severity = 'low'
            
            if severity:
                # If we are already in an incident for this pollutant, just update it
                if current_incident and current_incident['severity'] == severity:
                    current_incident['end_index'] = int(idx)
                    current_incident['max_value'] = max(current_incident['max_value'], float(df.loc[idx, pollutant]))
                else:
                    # If severity changed or new incident, close old and start new
                    if current_incident:
                        alerts.append(current_incident)
                    
                    prefix = "🔴 CRITICAL" if severity == 'high' else "🟠 WARNING" if severity == 'medium' else "🟡 ADVISORY"
                    current_incident = {
                        'timestamp': datetime.now().isoformat(),
                        'start_index': int(idx),
                        'end_index': int(idx),
                        'pollutant': pollutant,
                        'max_value': float(df.loc[idx, pollutant]),
                        'severity': severity,
                        'message': f"{prefix}: {threshold['description']} levels are sustained {severity} ({value:.2f} {threshold['unit']})"
                    }
            else:
                # Value dropped below Low, close the incident
                if current_incident:
                    alerts.append(current_incident)
                    current_incident = None
        
        # Catch any incident still open at the end of the data
        if current_incident:
            alerts.append(current_incident)

    return alerts

def save_alerts(alerts, output_dir='outputs/alerts'):
    """Save alerts to JSON file"""
    if not alerts:
        return None

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_path = os.path.join(output_dir, f'alerts_{timestamp}.json')
    
    try:
        with open(file_path, 'w') as f:
            json.dump(alerts, f, indent=2)
        return file_path
    except Exception as e:
        print(f"✗ Error saving alerts: {e}")
        return None