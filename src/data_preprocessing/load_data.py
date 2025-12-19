import pandas as pd
import os

def load_raw_data(file_path, min_required_rows=6):
    """
    Loads raw CSV data and performs initial validation on the dataset size.

    The function utilizes Pandas' 'python' engine with 'sep=None' to 
    automatically detect delimiters (e.g., commas vs. semicolons), which is 
    essential for various European CSV formats like the UCI Air Quality set.

    Args:
        file_path (str): Path to the source CSV file.
        min_required_rows (int): Minimum record count needed for feature 
            engineering (e.g., to satisfy 3-hour rolling windows). Defaults to 6.

    Returns:
        pd.DataFrame: The loaded dataset.

    Raises:
        FileNotFoundError: If the file_path is invalid.
        ValueError: If the dataset does not meet the min_required_rows threshold.
        Exception: For other ingestion-related errors.
    """
    try:
        # --- Pre-flight Check ---
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Engine 'python' and sep=None allows for flexible delimiter detection
        df = pd.read_csv(file_path, sep=None, engine='python')
        
        row_count = len(df)
        print(f"Loaded {row_count} rows from {file_path}")

        # --- Data Integrity Validation ---
        # We enforce a minimum row count because time-series features like 
        # 3-hour lags and moving averages require a baseline history.
        if row_count < min_required_rows:
            error_msg = (
                f"Insufficient data: Found {row_count} rows, but at least {min_required_rows} "
                "are required to generate accurate temporal features (lags and rolling averages)."
            )
            print(f"{error_msg}")
            raise ValueError(error_msg)

        return df

    except Exception as e:
        # Log the error with context before re-raising to the caller
        print(f"Error during data loading/validation: {e}")
        raise

def save_processed_data(df, output_path):
    """
    Exports a DataFrame to a CSV file, ensuring the target directory exists.

    Args:
        df (pd.DataFrame): The cleaned/processed DataFrame to save.
        output_path (str): The full destination path including filename.

    Returns:
        None

    Raises:
        Exception: If directory creation fails or write permissions are denied.
    """
    try:
        # Check for empty or NoneType dataframes to prevent accidental file overwrites
        if df is None or df.empty:
            print("Warning: Attempting to save an empty DataFrame.")
            return

        # Ensure the directory tree exists (prevents FileNotFoundError on save)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save without the index column to keep the CSV clean for downstream scripts
        df.to_csv(output_path, index=False)
        print(f"Saved processed data to {output_path}")

    except Exception as e:
        print(f"Error saving data: {e}")
        raise