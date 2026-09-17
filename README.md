# SMS Spam Classification — NLP Track

Complete pipeline: raw SMS text → cleaned text → TF-IDF/BoW/embedding features
→ trained classical & embedding-based models → best model deployed behind a
FastAPI web app with a confidence score.

## 1. Environment Setup

\`\`\`bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
\`\`\`

Download the dataset and place it at `data/raw/spam.csv` (SMS Spam Collection
Dataset, UCI/Kaggle).

## 2. Train the Model

\`\`\`bash
python main.py
\`\`\`

Outputs written to `results/`:
- `model_comparison.csv`
- `top_spam_words.csv`
- `misclassified_examples.csv`
- `confusion_matrix_best_model.png`
- `confusion_matrix_worst_model.png`

## 3. Run the FastAPI App

\`\`\`bash
uvicorn app:app --reload
\`\`\`

Open http://127.0.0.1:8000 in a browser.

## 4. Model Comparison

| Approach | Accuracy | Precision(spam) | Recall(spam) | F1(spam) |
|---|---|---|---|---|
| TF-IDF + Naive Bayes | 0.9704 | 0.9915 | 0.7852 | 0.8764 |
| **TF-IDF + Linear SVM** | **0.9874** | **0.9720** | **0.9329** | **0.9521** |
| BoW + Naive Bayes | 0.9821 | 0.9574 | 0.9060 | 0.9310 |
| GloVe Embeddings + Logistic Regression | 0.9067 | 0.5983 | 0.9195 | 0.7249 |
| Sentence-Transformer + Logistic Regression | 0.9776 | 0.8875 | 0.9530 | 0.9191 |

**Recommendation:** TF-IDF + Linear SVM is deployed — best F1-score on the
spam class, the metric that matters most on this imbalanced dataset.

## 5. Project Structure

\`\`\`
nlp-sms-spam-task3/
├── data/raw/spam.csv
├── src/
│   ├── settings.py
│   ├── data_loading.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── modeling.py
│   └── train_and_save.py
├── models/spam_pipeline.joblib
├── results/
├── templates/index.html
├── app.py
├── main.py
├── pyproject.toml
└── README.md
\`\`\`