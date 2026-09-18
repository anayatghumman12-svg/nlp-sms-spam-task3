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
    """
    Predicts whether a given message is spam or ham using the trained pipeline.

    Args:
        message (str): The raw input message to classify.

    Returns:
        Tuple[str, float]: A tuple containing the predicted label ("spam" or "ham")
                            and the confidence percentage (0-100) for that prediction.
    """
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
    """
    Renders the home page with an empty result.

    Args:
        request (Request): The incoming HTTP request.

    Returns:
        HTMLResponse: The rendered home page template.
    """
    return templates.TemplateResponse(request, "index.html", {"result": None})


@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request, message: str = Form(...)) -> HTMLResponse:
    """
    Handles form submission, classifies the submitted message, and renders
    the result on the home page.

    Args:
        request (Request): The incoming HTTP request.
        message (str): The message submitted via the form.

    Returns:
        HTMLResponse: The rendered home page template with the prediction
                      result and the submitted message.
    """
    label, confidence = predict_message(message)
    result = {"label": label, "confidence": confidence}

    return templates.TemplateResponse(
        request, "index.html", {"result": result, "submitted_message": message},
    )