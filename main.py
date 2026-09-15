import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.data_loading import load_raw_dataset, summarize_dataset
from src.features import EmbeddingVectorizer, build_bow_features, build_tfidf_features
from src.modeling import (
    find_misclassified,
    get_linear_svm,
    get_naive_bayes,
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


def main():
    # ---------- Step 1: Load & Inspect ----------
    print("Step 1: Loading dataset...")
    df = load_raw_dataset()
    summary = summarize_dataset(df)
    print(summary)

    # ---------- Step 2: Preprocessing ----------
    print("\nStep 2: Cleaning messages...")
    df = add_clean_message_column(df)
    x = df[CLEAN_MESSAGE_COLUMN]
    y = df[LABEL_COLUMN].map(LABEL_MAP)
    msgs = df[MESSAGE_COLUMN]

    x_train, x_test, y_train, y_test, msg_train, msg_test = train_test_split(
        x, y, msgs, test_size=0.2, random_state=42, stratify=y
    )

    results = []

    # ---------- Step 3 & 4: TF-IDF / BoW + classical models ----------
    print("\nStep 3 & 4: TF-IDF/BoW features aur classical models...")
    tfidf_vec, x_train_tfidf, x_test_tfidf = build_tfidf_features(x_train, x_test)
    bow_vec, x_train_bow, x_test_bow = build_bow_features(x_train, x_test)

    nb_result = train_and_evaluate(
        get_naive_bayes(), x_train_tfidf, y_train, x_test_tfidf, y_test,
        "TF-IDF + Naive Bayes"
    )
    results.append(nb_result)

    svm_result = train_and_evaluate(
        get_linear_svm(), x_train_tfidf, y_train, x_test_tfidf, y_test,
        "TF-IDF + Linear SVM"
    )
    results.append(svm_result)

    bow_result = train_and_evaluate(
        get_naive_bayes(), x_train_bow, y_train, x_test_bow, y_test,
        "BoW + Naive Bayes"
    )
    results.append(bow_result)
    print("\nStep 5: GloVe embeddings...")
    try:
        embedder = EmbeddingVectorizer()
        x_train_emb = embedder.transform(list(x_train))
        x_test_emb = embedder.transform(list(x_test))

        emb_result = train_and_evaluate(
            LogisticRegression(max_iter=1000, class_weight="balanced"),
            x_train_emb, y_train, x_test_emb, y_test,
            "GloVe Embeddings + Logistic Regression"
        )
        results.append(emb_result)
    except Exception as e:
        print(f"Step 5 skip ho gaya, error: {e}")
    print("\nStep 6: Sentence-transformer embeddings...")
    try:
        from sentence_transformers import SentenceTransformer

        st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        x_train_st = st_model.encode(list(x_train))
        x_test_st = st_model.encode(list(x_test))

        st_result = train_and_evaluate(
            LogisticRegression(max_iter=1000, class_weight="balanced"),
            x_train_st, y_train, x_test_st, y_test,
            "Sentence-Transformer + Logistic Regression"
        )
        results.append(st_result)
    except Exception as e:
        print(f"Step 6 skip ho gaya, error: {e}")
    print("\nTop spam words aur misclassified messages nikal rahe hain...")
    best_result = max(results[:2], key=lambda r: r["f1_spam"]) 
    top_words = top_spam_words(best_result["model"], tfidf_vec, 15)
    print("Top 15 spam words:", top_words)

    y_pred_best = best_result["model"].predict(x_test_tfidf)
    mismatched = find_misclassified(msg_test, y_test, y_pred_best, n=5)
    print("Misclassified examples:", mismatched)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    comparison_df = pd.DataFrame(
        [{k: v for k, v in r.items() if k != "model"} for r in results]
    )
    comparison_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    print(f"\nModel comparison table:\n{comparison_df}")

    top_words_df = pd.DataFrame(top_words, columns=["word", "score"])
    top_words_df.to_csv(RESULTS_DIR / "top_spam_words.csv", index=False)

    mismatched_df = pd.DataFrame(mismatched)
    mismatched_df.to_csv(RESULTS_DIR / "misclassified_examples.csv", index=False)
    print("\nFinal deployed pipeline (TF-IDF + Calibrated SVM) train kar rahe hain...")
    final_pipeline = train_final_pipeline()
    save_pipeline(final_pipeline)

    print("\nDone! results/ folder mein CSVs dekho, models/ mein saved pipeline hai.")


if __name__ == "__main__":
    main()