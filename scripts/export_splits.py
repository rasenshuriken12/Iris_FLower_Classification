"""One-off script to export train/test splits as CSV."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_loader import load_raw_data
from src.config import DATA_PROCESSED, TEST_SIZE, RANDOM_STATE, TARGET_COL

df = load_raw_data(save_csv=False)
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
X_train.to_csv(DATA_PROCESSED / "X_train.csv", index=False)
X_test.to_csv(DATA_PROCESSED / "X_test.csv", index=False)
y_train.to_csv(DATA_PROCESSED / "y_train.csv", index=False)
y_test.to_csv(DATA_PROCESSED / "y_test.csv", index=False)

print("✅ Exported 4 CSVs to data/processed/")