import pandas as pd

from src.settings import (
    LABEL_COLUMN,
    MESSAGE_COLUMN,
    RAW_CSV_ENCODING,
    RAW_DATA_PATH,
    RAW_LABEL_COLUMN,
    RAW_MESSAGE_COLUMN,
)
def load_raw_dataset(csv_path=RAW_DATA_PATH) -> pd.DataFrame:
  
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset nahi mila: {csv_path}. Pehle spam.csv ko "
            f"data/raw/ folder mein daalo."
        )

    df = pd.read_csv(csv_path, encoding=RAW_CSV_ENCODING)
    df = df[[RAW_LABEL_COLUMN, RAW_MESSAGE_COLUMN]]
    df = df.rename(columns={
        RAW_LABEL_COLUMN: LABEL_COLUMN,
        RAW_MESSAGE_COLUMN: MESSAGE_COLUMN,
    })

    return df


def summarize_dataset(df: pd.DataFrame) -> dict:
    class_counts = df[LABEL_COLUMN].value_counts()
    class_percent = df[LABEL_COLUMN].value_counts(normalize=True) * 100
    char_lengths = df[MESSAGE_COLUMN].str.len()
    word_counts = df[MESSAGE_COLUMN].str.split().str.len()

    summary = {
        "total_rows": len(df),
        "class_counts": class_counts.to_dict(),
        "class_percent": class_percent.round(2).to_dict(),
        "avg_char_length": round(char_lengths.mean(), 2),
        "avg_word_count": round(word_counts.mean(), 2),
    }

    return summary