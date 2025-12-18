import pandas as pd
import os

def load_raw_data(file_path, min_required_rows=6):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        df = pd.read_csv(file_path, sep=None, engine='python')
        row_count = len(df)
        print(f"Loaded {row_count} rows from {file_path}")
        if row_count < min_required_rows:
            error_msg = (
                f"Insufficient data: Found {row_count} rows, but at least {min_required_rows} "
                "are required to generate accurate temporal features (lags and rolling averages)."
            )
            print(f"{error_msg}")
            raise ValueError(error_msg)
        return df
    except Exception as e:
        print(f"Error during data loading/validation: {e}")
        raise

def save_processed_data(df, output_path):
    try:
        if df is None or df.empty:
            print("Warning: Attempting to save an empty DataFrame.")
            return

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved processed data to {output_path}")
    except Exception as e:
        print(f"Error saving data: {e}")
        raise