"""Evaluate trained models."""
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)

from src.config import REPORTS_DIR, FIGURES_DIR


def evaluate_model(model, X_test, y_test, model_name: str,
                   save_report: bool = True) -> dict:
    """Compute accuracy, classification report, and confusion matrix."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"\n📊 {model_name} — Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=model.classes_)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"cm_{model_name}.png", dpi=120)
    plt.close()

    metrics = {"model": model_name, "accuracy": acc, "report": report}

    if save_report:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(REPORTS_DIR / f"metrics_{model_name}.json", "w") as f:
            json.dump(metrics, f, indent=2)

    return metrics


def plot_feature_importance(model, feature_names, model_name="model"):
    """Bar plot of feature importances (tree-based models only)."""
    if not hasattr(model, "feature_importances_"):
        print(f"⚠️  {model_name} has no feature_importances_")
        return

    import pandas as pd
    imp = pd.Series(model.feature_importances_,
                    index=feature_names).sort_values()
    plt.figure(figsize=(7, 4))
    imp.plot(kind="barh", color="teal")
    plt.title(f"Feature Importance — {model_name}")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"feat_imp_{model_name}.png", dpi=120)
    plt.close()