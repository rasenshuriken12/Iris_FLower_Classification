"""Preprocessing: split + scale."""
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (
    RANDOM_STATE, TEST_SIZE, TARGET_COL, MODELS_DIR
)


def split_and_scale(df: pd.DataFrame, save_scaler: bool = True):
    """
    Split into train/test and standardize features.

    Returns
    -------
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names
    """
    feature_names = [c for c in df.columns if c != TARGET_COL]
    X = df[feature_names]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if save_scaler:
        joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
        print(f"💾 Saved scaler -> {MODELS_DIR / 'scaler.pkl'}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names