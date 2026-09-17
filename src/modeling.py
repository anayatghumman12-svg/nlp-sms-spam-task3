from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.settings import HAM_LABEL, RANDOM_STATE, SPAM_LABEL


def get_naive_bayes() -> MultinomialNB:
    return MultinomialNB()


def get_linear_svm() -> LinearSVC:
    return LinearSVC(class_weight="balanced", random_state=RANDOM_STATE)


def train_and_evaluate(
    model: Any, x_train: Any, y_train: Any, x_test: Any, y_test: Any, approach_name: str
) -> Dict[str, Any]:
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
    feature_names = np.array(vectorizer.get_feature_names_out())

    if hasattr(model, "coef_"):
        scores = model.coef_.ravel()
    else:
        scores = model.feature_log_prob_[1] - model.feature_log_prob_[0]

    top_indices = np.argsort(scores)[::-1][:top_n]

    return list(zip(feature_names[top_indices], scores[top_indices]))


def find_misclassified(messages: Any, y_true: Any, y_pred: Any, n: int = 5) -> List[Dict[str, Any]]:
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