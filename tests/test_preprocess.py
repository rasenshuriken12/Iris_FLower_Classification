import numpy as np
from src.data_loader import load_raw_data
from src.preprocess import split_and_scale


def test_split_sizes():
    df = load_raw_data(save_csv=False)
    X_train, X_test, y_train, y_test, _, _ = split_and_scale(df, save_scaler=False)
    assert len(X_train) == 120
    assert len(X_test) == 30
    assert len(y_train) == 120


def test_scaling_zero_mean():
    df = load_raw_data(save_csv=False)
    X_train, X_test, _, _, _, _ = split_and_scale(df, save_scaler=False)
    assert abs(X_train.mean()) < 1e-6
    assert abs(X_train.std() - 1) < 1e-6