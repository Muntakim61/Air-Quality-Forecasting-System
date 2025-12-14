import pandas as pd
import os

def load_raw_data(file_path):
    """Load raw CSV data"""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        df = pd.read_csv(file_path)
        print(f"✓ Loaded {len(df)} rows from {file_path}")
        return df
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        raise

def save_processed_data(df, output_path):
    """Save processed data to CSV"""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"✓ Saved processed data to {output_path}")
    except Exception as e:
        print(f"✗ Error saving data: {e}")
        raise