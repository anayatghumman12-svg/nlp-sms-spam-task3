import pandas as pd
from pathlib import Path
from src.settings import (
    RAW_DATA_PATH,
    RAW_CSV_ENCODING,
    RAW_LABEL_COLUMN,
    RAW_MESSAGE_COLUMN,
    LABEL_COLUMN,
    MESSAGE_COLUMN,
)

def load_raw_dataset(csv_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Loads the raw spam dataset from a CSV file and renames its columns.

    Args:
        csv_path (Path, optional): Path to the raw CSV dataset.
                                    Defaults to RAW_DATA_PATH.

    Returns:
        pd.DataFrame: DataFrame containing the label and message columns,
                      renamed to LABEL_COLUMN and MESSAGE_COLUMN.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Please place spam.csv in data/raw/ folder."
        )
    
    # Raw data load karein
    df = pd.read_csv(csv_path, encoding=RAW_CSV_ENCODING)
    
    # Specific columns extract aur rename karein
    df = df[[RAW_LABEL_COLUMN, RAW_MESSAGE_COLUMN]].copy()
    df.columns = [LABEL_COLUMN, MESSAGE_COLUMN]
    
    return df

def summarize_dataset(df: pd.DataFrame) -> dict:
    """
    Summarizes the dataset by computing total sample count and class distribution.

    Args:
        df (pd.DataFrame): Input DataFrame containing the label column.

    Returns:
        dict: A dictionary with "total_samples" (int) and "class_counts"
              (dict mapping each label to its count).
    """
    class_counts = df[LABEL_COLUMN].value_counts().to_dict()
    total_samples = len(df)
    
    return {
        "total_samples": total_samples,
        "class_counts": class_counts
    }