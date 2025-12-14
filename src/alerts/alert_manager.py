import yaml
import json
import os
from datetime import datetime

def load_alert_thresholds(config_path='src/config/configs/alerts.yaml'):
    """Load alert threshold configuration"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config['thresholds']
    except Exception as e:
        print(f"✗ Error loading alert config: {e}")
        # Return default thresholds based on WHO and EPA standards
        return {
            'co': {'low': 2.0, 'medium': 4.0, 'high': 9.0},
            'no2': {'low': 100, 'medium': 200, 'high': 400},
            'nox': {'low': 150, 'medium': 300, 'high': 600},
            'benzene': {'low': 5.0, 'medium': 10.0, 'high': 20.0}
        }

def evaluate_alerts(predictions_df, thresholds):
    """Evaluate alert levels based on predictions"""
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
    """Save alerts to JSON file"""
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