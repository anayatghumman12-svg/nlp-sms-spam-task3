import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.settings import HAM_LABEL, RANDOM_STATE, SPAM_LABEL


def get_naive_bayes():
    return MultinomialNB()


def get_linear_svm():
    return LinearSVC(class_weight="balanced", random_state=RANDOM_STATE)


def train_and_evaluate(model, x_train, y_train, x_test, y_test, approach_name):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    report = classification_report(
        y_test,
        y_pred,
        target_names=[HAM_LABEL, SPAM_LABEL],
        output_dict=True,
        zero_division=0,
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


def top_spam_words(model, vectorizer, top_n=15):
   
    feature_names = np.array(vectorizer.get_feature_names_out())

    if hasattr(model, "coef_"):
        scores = model.coef_.ravel()
    else:
        scores = model.feature_log_prob_[1] - model.feature_log_prob_[0]

    top_indices = np.argsort(scores)[::-1][:top_n]

    return list(zip(feature_names[top_indices], scores[top_indices]))


def find_misclassified(messages, y_true, y_pred, n=5):
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