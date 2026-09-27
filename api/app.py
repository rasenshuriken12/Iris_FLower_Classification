"""FastAPI inference server."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from api.schemas import IrisFeatures, PredictionResponse
from src.predict import predict

app = FastAPI(title="Iris Classification API", version="1.0.0")


@app.get("/")
def root():
    return {"message": "Iris Classification API", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict_iris(payload: IrisFeatures):
    features = [
        payload.sepal_length, payload.sepal_width,
        payload.petal_length, payload.petal_width,
    ]
    label, proba = predict(features)

    proba_dict = None
    if proba is not None:
        from sklearn.utils.multiclass import unique_labels
        proba_dict = {str(i): float(p) for i, p in enumerate(proba)}

    return PredictionResponse(species=str(label), probabilities=proba_dict)