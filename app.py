
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.preprocessing import clean_text
from src.train_and_save import load_pipeline

app = FastAPI(title="SMS Spam Classifier")
templates = Jinja2Templates(directory="templates")

# Pipeline ko app start hote hi ek baar load kar lete hain,
# har request pe dobara load karne ki zaroorat nahi
pipeline = load_pipeline()


def predict_message(message: str):
    """
    Ek SMS message ko clean karta hai, phir model se predict karwata hai.

    Args:
        message: raw SMS text jo user ne type kiya

    Returns:
        (label, confidence_percentage) - jaise ("spam", 97.5)
    """
    cleaned = clean_text(message)

    # predict_proba() do numbers deta hai: [ham_probability, spam_probability]
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
def home(request: Request):
    """Homepage - khaali form dikhata hai."""
    return templates.TemplateResponse(request, "index.html", {"result": None})


@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request, message: str = Form(...)):
    """
    Form submit hone pe ye chalta hai - message ko predict karta hai
    aur result ke sath page dobara render karta hai.
    """
    label, confidence = predict_message(message)
    result = {"label": label, "confidence": confidence}

    return templates.TemplateResponse(
        request,
        "index.html",
        {"result": result, "submitted_message": message},
    )