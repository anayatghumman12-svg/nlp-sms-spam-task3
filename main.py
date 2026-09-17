from typing import Any, Dict, List, Tuple

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.data_loading import load_raw_dataset, summarize_dataset
from src.features import EmbeddingVectorizer, build_bow_features, build_tfidf_features
from src.modeling import (
    find_misclassified,
    get_linear_svm,
    get_naive_bayes,
    plot_confusion_matrix,
    top_spam_words,
    train_and_evaluate,
)
from src.preprocessing import add_clean_message_column
from src.settings import (
    CLEAN_MESSAGE_COLUMN,
    LABEL_COLUMN,
    LABEL_MAP,
    MESSAGE_COLUMN,
    RESULTS_DIR,
)
from src.train_and_save import save_pipeline, train_final_pipeline


def main() -> None:
    print("Step 1: Loading dataset...")
    df = load_raw_dataset()
    summary = summarize_dataset(df)
    print(summary)

    print("\nStep 2: Cleaning messages...")
    df = add_clean_message_column(df)

    x = df[CLEAN_MESSAGE_COLUMN]
    y = df[LABEL_COLUMN].map(LABEL_MAP)
    msgs = df[MESSAGE_COLUMN]

    x_train, x_test, y_train, y_test, msg_train, msg_test = train_test_split(
        x, y, msgs, test_size=0.2, random_state=42, stratify=y
    )

    results: List[Dict[str, Any]] = []
    # Har result ke sath uska x_test bhi record karte hain, taake baad
    # mein confusion matrix ke liye sahi predictions nikal sakein
    results_with_test_features: List[Tuple[Dict[str, Any], Any]] = []

    print("\nStep 3 & 4: TF-IDF/BoW features aur classical models...")
    tfidf_vec, x_train_tfidf, x_test_tfidf = build_tfidf_features(x_train, x_test)
    bow_vec, x_train_bow, x_test_bow = build_bow_features(x_train, x_test)

    nb_result = train_and_evaluate(
        get_naive_bayes(), x_train_tfidf, y_train, x_test_tfidf, y_test,
        "TF-IDF + Naive Bayes"
    )
    results.append(nb_result)
    results_with_test_features.append((nb_result, x_test_tfidf))

    svm_result = train_and_evaluate(
        get_linear_svm(), x_train_tfidf, y_train, x_test_tfidf, y_test,
        "TF-IDF + Linear SVM"
    )
    results.append(svm_result)
    results_with_test_features.append((svm_result, x_test_tfidf))

    bow_result = train_and_evaluate(
        get_naive_bayes(), x_train_bow, y_train, x_test_bow, y_test,
        "BoW + Naive Bayes"
    )
    results.append(bow_result)
    results_with_test_features.append((bow_result, x_test_bow))

    print("\nStep 5: GloVe embeddings...")
    try:
        embedder =