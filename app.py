from typing import Tuple

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.preprocessing import clean_text
from src.train_and_save import load_pipeline

app = FastAPI(title="SMS Spam Classifier")
templates = Jinja2Templates(directory="templates")

pipeline = load_pipeline()


def predict_message(message: str) -> Tuple[str, float]:
    cleaned = clean_text(message)
    probabilities = pipeline.predict_proba([cleaned])[0]
    spam_probability = probabilities[1]

    if spam_probability >= 0.5:
        label = "spam"
        confidence = spam_probability
    else:
        label = "ham"
        confidence = 1 - spam_probability

    return label, round(confidence * 100, 2)


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html", {"result": None})


@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request, message: str = Form(...)) -> HTMLResponse:
    label, confidence = predict_message(message)
    result = {"label": label, "confidence": confidence}

    return templates.TemplateResponse(
        request, "index.html", {"result": result, "submitted_message": message},
    )