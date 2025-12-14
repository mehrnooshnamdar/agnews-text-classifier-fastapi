from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.config import SETTINGS
from app.inference import TextClassifier


app = FastAPI(title="AG News Text Classifier", version="1.0.0")


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=5000)
    top_k: int = Field(2, ge=1, le=4)


class PredictResponse(BaseModel):
    label: str
    confidence: float
    top_k: list[dict]


_classifier: TextClassifier | None = None


@app.on_event("startup")
def load_model() -> None:
    global _classifier
    model_dir = Path(SETTINGS.model_dir)
    _classifier = TextClassifier(model_dir=model_dir)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    assert _classifier is not None, "Model not loaded"
    return _classifier.predict(req.text, top_k=req.top_k)
