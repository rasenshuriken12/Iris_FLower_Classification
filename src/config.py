"""Central configuration: paths, constants, hyperparameters."""
from pathlib import Path

# ---------- Paths ----------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Create dirs if missing
for _dir in [DATA_RAW, DATA_PROCESSED, MODELS_DIR, FIGURES_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ---------- Data ----------
TARGET_COL = "species"
FEATURE_COLS = ["sepal length (cm)", "sepal width (cm)",
                "petal length (cm)", "petal width (cm)"]

# ---------- Split ----------
RANDOM_STATE = 42
TEST_SIZE = 0.2

# ---------- Model Hyperparameters ----------
LOGISTIC_PARAMS = {"max_iter": 200, "random_state": RANDOM_STATE}
KNN_PARAMS = {"n_neighbors": 5}
SVM_PARAMS = {"C": 1.0, "kernel": "rbf", "gamma": "scale", "probability": True}
DT_PARAMS = {"random_state": RANDOM_STATE}
RF_PARAMS = {"n_estimators": 100, "random_state": RANDOM_STATE}

# ---------- Grid Search ----------
SVM_GRID = {
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", 0.01, 0.1, 1],
    "kernel": ["rbf", "linear"],
}