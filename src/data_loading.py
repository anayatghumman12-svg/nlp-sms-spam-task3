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
    class_counts = df[LABEL_COLUMN].value_counts().to_dict()
    total_samples = len(df)
    
    return {
        "total_samples": total_samples,
        "class_counts": class_counts
    }