"""Train multiple models and persist them."""
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.config import (
    MODELS_DIR, LOGISTIC_PARAMS, KNN_PARAMS,
    SVM_PARAMS, DT_PARAMS, RF_PARAMS,
)


def get_models() -> dict:
    """Return a dict of untrained model instances."""
    return {
        "logistic": LogisticRegression(**LOGISTIC_PARAMS),
        "knn": KNeighborsClassifier(**KNN_PARAMS),
        "svm": SVC(**SVM_PARAMS),
        "decision_tree": DecisionTreeClassifier(**DT_PARAMS),
        "random_forest": RandomForestClassifier(**RF_PARAMS),
    }


def train_models(X_train, y_train, save: bool = True) -> dict:
    """Fit all models and optionally save to disk."""
    models = get_models()
    trained = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
        if save:
            path = MODELS_DIR / f"{name}.pkl"
            joblib.dump(model, path)
        print(f"✅ Trained: {name}")

    return trained