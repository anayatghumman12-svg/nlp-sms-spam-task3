from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.settings import HAM_LABEL, RANDOM_STATE, SPAM_LABEL


def get_naive_bayes() -> MultinomialNB:
    """
    Initializes and returns a Multinomial Naive Bayes classifier.

    Returns:
        MultinomialNB: A new instance of the Multinomial Naive Bayes classifier.
    """
    return MultinomialNB()


def get_linear_svm() -> LinearSVC:
    """
    Initializes and returns a Linear Support Vector Classifier with balanced class weights.

    Returns:
        LinearSVC: A new instance of the Linear SVM classifier configured
                   with balanced class weights and a fixed random state.
    """
    return LinearSVC(class_weight="balanced", random_state=RANDOM_STATE)


def train_and_evaluate(
    model: Any, x_train: Any, y_train: Any, x_test: Any, y_test: Any, approach_name: str
) -> Dict[str, Any]:
    """
    Trains a given model on training data and evaluates performance metrics on test data.

    Args:
        model (Any): The classifier/model to train.
        x_train (Any): Training feature data.
        y_train (Any): Training labels.
        x_test (Any): Test feature data.
        y_test (Any): Test labels.
        approach_name (str): Name identifying the approach/model being evaluated.

    Returns:
        Dict[str, Any]: A dictionary containing the approach name, accuracy,
                         precision, recall, f1-score (for the spam class),
                         and the trained model itself.
    """
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    report = classification_report(
        y_test, y_pred, target_names=[HAM_LABEL, SPAM_LABEL],
        output_dict=True, zero_division=0,
    )

    result = {
        "approach": approach_name,
        "accuracy": round(accuracy, 4),
        "precision_spam": round(report[SPAM_LABEL]["precision"], 4),
        "recall_spam": round(report[SPAM_LABEL]["recall"], 4),
        "f1_spam": round(report[SPAM_LABEL]["f1-score"], 4),
        "model": model,
    }

    return result


def top_spam_words(model: Any, vectorizer: Any, top_n: int = 15) -> List[Tuple[str, float]]:
    """
    Extracts the top N most influential words for spam classification.

    Args:
        model (Any): The trained classifier (must have `coef_` or `feature_log_prob_`).
        vectorizer (Any): The fitted vectorizer used to extract feature names.
        top_n (int, optional): Number of top words to return. Defaults to 15.

    Returns:
        List[Tuple[str, float]]: A list of (word, score) tuples sorted by
                                  descending influence/importance.
    """
    feature_names = np.array(vectorizer.get_feature_names_out())

    if hasattr(model, "coef_"):
        scores = model.coef_.ravel()
    else:
        scores = model.feature_log_prob_[1] - model.feature_log_prob_[0]

    top_indices = np.argsort(scores)[::-1][:top_n]

    return list(zip(feature_names[top_indices], scores[top_indices]))


def find_misclassified(messages: Any, y_true: Any, y_pred: Any, n: int = 5) -> List[Dict[str, Any]]:
    """
    Identifies misclassified text examples by comparing true and predicted labels.

    Args:
        messages (Any): Iterable of original text messages.
        y_true (Any): Iterable of true labels.
        y_pred (Any): Iterable of predicted labels.
        n (int, optional): Maximum number of misclassified examples to return. Defaults to 5.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each containing the message,
                               true label, and predicted label for a misclassified example.
    """
    messages_list = list(messages)
    y_true_list = list(y_true)
    y_pred_list = list(y_pred)

    mismatched = []
    for msg, true_label, pred_label in zip(messages_list, y_true_list, y_pred_list):
        if true_label != pred_label:
            mismatched.append({
                "message": msg,
                "true_label": true_label,
                "predicted_label": pred_label,
            })
        if len(mismatched) >= n:
            break

    return mismatched


def plot_confusion_matrix(y_true: Any, y_pred: Any, title: str, save_path: Path) -> None:
    """
    Generates and saves a confusion matrix plot image to the specified path.

    Args:
        y_true (Any): Iterable of true labels.
        y_pred (Any): Iterable of predicted labels.
        title (str): Title to display on the plot.
        save_path (Path): File path where the plot image will be saved.

    Returns:
        None
    """
    cm = confusion_matrix(y_true, y_pred)
    labels = [HAM_LABEL, SPAM_LABEL]

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color=color, fontsize=14, fontweight="bold")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)