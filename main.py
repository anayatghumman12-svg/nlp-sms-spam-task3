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

    print("Step 2: Cleaning messages...")
    df = add_clean_message_column(df)

    x = df[CLEAN_MESSAGE_COLUMN]
    y = df[LABEL_COLUMN].map(LABEL_MAP)
    msgs = df[MESSAGE_COLUMN]

    x_train, x_test, y_train, y_test, msg_train, msg_test = train_test_split(
        x, y, msgs, test_size=0.2, random_state=42, stratify=y
    )

    results: List[Dict[str, Any]] = []
    results_with_test_features: List[Tuple[Dict[str, Any], Any]] = []

    print("Step 3 and 4: TF-IDF and BoW features and classical models")
    tfidf_vec, x_train_tfidf, x_test_tfidf = build_tfidf_features(x_train, x_test)
    bow_vec, x_train_bow, x_test_bow = build_bow_features(x_train, x_test)

    nb_result = train_and_evaluate(get_naive_bayes(), x_train_tfidf, y_train, x_test_tfidf, y_test, "TF-IDF + Naive Bayes")
    results.append(nb_result)
    results_with_test_features.append((nb_result, x_test_tfidf))

    svm_result = train_and_evaluate(get_linear_svm(), x_train_tfidf, y_train, x_test_tfidf, y_test, "TF-IDF + Linear SVM")
    results.append(svm_result)
    results_with_test_features.append((svm_result, x_test_tfidf))

    bow_result = train_and_evaluate(get_naive_bayes(), x_train_bow, y_train, x_test_bow, y_test, "BoW + Naive Bayes")
    results.append(bow_result)
    results_with_test_features.append((bow_result, x_test_bow))

    print("Step 5: GloVe embeddings")
    try:
        embedder = EmbeddingVectorizer()
        x_train_emb = embedder.transform(list(x_train))
        x_test_emb = embedder.transform(list(x_test))
        emb_result = train_and_evaluate(LogisticRegression(max_iter=1000, class_weight="balanced"), x_train_emb, y_train, x_test_emb, y_test, "GloVe Embeddings + Logistic Regression")
        results.append(emb_result)
        results_with_test_features.append((emb_result, x_test_emb))
    except Exception as e:
        print("Step 5 skipped, error:", e)

    print("Step 6: Sentence transformer embeddings")
    try:
        from sentence_transformers import SentenceTransformer
        st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        x_train_st = st_model.encode(list(x_train))
        x_test_st = st_model.encode(list(x_test))
        st_result = train_and_evaluate(LogisticRegression(max_iter=1000, class_weight="balanced"), x_train_st, y_train, x_test_st, y_test, "Sentence-Transformer + Logistic Regression")
        results.append(st_result)
        results_with_test_features.append((st_result, x_test_st))
    except Exception as e:
        print("Step 6 skipped, error:", e)

    print("Computing top spam words and misclassified examples")
    best_tfidf_result = max(results[:2], key=lambda r: r["f1_spam"])
    top_words = top_spam_words(best_tfidf_result["model"], tfidf_vec, 15)
    print("Top 15 spam words:", top_words)

    y_pred_best_tfidf = best_tfidf_result["model"].predict(x_test_tfidf)
    mismatched = find_misclassified(msg_test, y_test, y_pred_best_tfidf, n=5)
    print("Misclassified examples:", mismatched)

    print("Computing confusion matrices for best and worst model")
    best_overall, best_x_test = max(results_with_test_features, key=lambda pair: pair[0]["f1_spam"])
    worst_overall, worst_x_test = min(results_with_test_features, key=lambda pair: pair[0]["f1_spam"])

    y_pred_best_overall = best_overall["model"].predict(best_x_test)
    y_pred_worst_overall = worst_overall["model"].predict(worst_x_test)

    plot_confusion_matrix(y_test, y_pred_best_overall, "Best Model: " + best_overall["approach"], RESULTS_DIR / "confusion_matrix_best_model.png")
    plot_confusion_matrix(y_test, y_pred_worst_overall, "Worst Model: " + worst_overall["approach"], RESULTS_DIR / "confusion_matrix_worst_model.png")

    print("Best model:", best_overall["approach"], best_overall["f1_spam"])
    print("Worst model:", worst_overall["approach"], worst_overall["f1_spam"])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    comparison_df = pd.DataFrame([{k: v for k, v in r.items() if k != "model"} for r in results])
    comparison_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    print(comparison_df)

    top_words_df = pd.DataFrame(top_words, columns=["word", "score"])
    top_words_df.to_csv(RESULTS_DIR / "top_spam_words.csv", index=False)

    mismatched_df = pd.DataFrame(mismatched)
    mismatched_df.to_csv(RESULTS_DIR / "misclassified_examples.csv", index=False)

    print("Training final deployed pipeline")
    final_pipeline = train_final_pipeline()
    save_pipeline(final_pipeline)

    print("Done")


if __name__ == "__main__":
    main()