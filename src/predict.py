"""Inference utilities."""
import numpy as np
import joblib
from src.config import MODELS_DIR


def load_artifacts(model_name: str = "svm"):
    """Load model and scaler from disk."""
    model = joblib.load(MODELS_DIR / f"{model_name}.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    return model, scaler


def predict(features, model_name: str = "svm"):
    """
    Predict species from a list of 4 feature values.

    Parameters
    ----------
    features : list | np.ndarray
        [sepal_length, sepal_width, petal_length, petal_width]

    Returns
    -------
    (predicted_label, probabilities_or_None)
    """
    model, scaler = load_artifacts(model_name)
    arr = np.asarray(features, dtype=float).reshape(1, -1)
    arr_scaled = scaler.transform(arr)

    pred = model.predict(arr_scaled)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(arr_scaled)[0]

    return pred, proba


def predict_batch(X, model_name: str = "svm"):
    """Predict for a 2D array of features."""
    model, scaler = load_artifacts(model_name)
    X_scaled = scaler.transform(X)
    return model.predict(X_scaled)