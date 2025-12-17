import pandas as pd
import os

def load_raw_data(file_path, min_required_rows=6):
    """
    Load raw CSV data with a validation check for minimum data volume.
    
    Args:
        file_path (str): Path to the CSV file.
        min_required_rows (int): Minimum rows needed for feature engineering (default 6).
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine delimiter (handling both standard CSV and your test_data semicolon format)
        df = pd.read_csv(file_path, sep=None, engine='python')
        
        row_count = len(df)
        print(f"✓ Loaded {row_count} rows from {file_path}")

        # --- NEW VALIDATION LOGIC ---
        if row_count < min_required_rows:
            error_msg = (
                f"Insufficient data: Found {row_count} rows, but at least {min_required_rows} "
                "are required to generate accurate temporal features (lags and rolling averages)."
            )
            print(f"✗ {error_msg}")
            # Raising a ValueError stops the pipeline from moving forward
            raise ValueError(error_msg)
        # ----------------------------

        return df
    except Exception as e:
        print(f"✗ Error during data loading/validation: {e}")
        raise

def save_processed_data(df, output_path):
    """Save processed data to CSV"""
    try:
        if df is None or df.empty:
            print("⚠ Warning: Attempting to save an empty DataFrame.")
            return

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"✓ Saved processed data to {output_path}")
    except Exception as e:
        print(f"✗ Error saving data: {e}")
        raise