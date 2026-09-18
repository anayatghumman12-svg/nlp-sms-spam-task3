from pathlib import Path

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.data_loading import load_raw_dataset
from src.preprocessing import add_clean_message_column
from src.settings import (
    CLEAN_MESSAGE_COLUMN, FINAL_PIPELINE_PATH, LABEL_COLUMN,
    LABEL_MAP, MODELS_DIR, RANDOM_STATE, TEST_SIZE, TFIDF_MAX_FEATURES,
)


def train_final_pipeline() -> Pipeline:
    """
    Loads and preprocesses the dataset, then trains a calibrated Linear SVM
    pipeline (TF-IDF + calibrated classifier) for spam detection.

    Returns:
        Pipeline: The trained scikit-learn pipeline, fitted on the training split.
    """
    df = load_raw_dataset()
    df = add_clean_message_column(df)

    x = df[CLEAN_MESSAGE_COLUMN]
    y = df[LABEL_COLUMN].map(LABEL_MAP)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    base_svm = LinearSVC(class_weight="balanced", random_state=RANDOM_STATE)
    calibrated_svm = CalibratedClassifierCV(base_svm, method="sigmoid", cv=5)

    pipeline = Pipeline(steps=[
        ("tfidf", TfidfVectorizer(max_features=TFIDF_MAX_FEATURES)),
        ("classifier", calibrated_svm),
    ])

    pipeline.fit(x_train, y_train)

    test_accuracy = pipeline.score(x_test, y_test)
    print(f"Final pipeline test accuracy: {test_accuracy:.4f}")

    return pipeline


def save_pipeline(pipeline: Pipeline, path: Path = FINAL_PIPELINE_PATH) -> None:
    """
    Saves the trained pipeline to disk using joblib.

    Args:
        pipeline (Pipeline): The trained pipeline to save.
        path (Path, optional): Destination file path. Defaults to FINAL_PIPELINE_PATH.

    Returns:
        None
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"Pipeline saved to {path}")


def load_pipeline(path: Path = FINAL_PIPELINE_PATH) -> Pipeline:
    """
    Loads a previously trained pipeline from disk.

    Args:
        path (Path, optional): Path to the saved pipeline file. Defaults to FINAL_PIPELINE_PATH.

    Returns:
        Pipeline: The loaded scikit-learn pipeline.

    Raises:
        FileNotFoundError: If no pipeline file exists at the given path.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"trained pipeline not found any {path} pe. runs'python main.py' first ."
        )
    return joblib.load(path)


if __name__ == "__main__":
    trained_pipeline = train_final_pipeline()
    save_pipeline(trained_pipeline)