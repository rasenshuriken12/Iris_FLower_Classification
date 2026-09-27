from src.data_loader import load_raw_data
from src.preprocess import split_and_scale
from src.train import train_models


def test_models_train():
    df = load_raw_data(save_csv=False)
    X_train, X_test, y_train, y_test, _, _ = split_and_scale(df, save_scaler=False)
    models = train_models(X_train, y_train, save=False)

    assert set(models.keys()) == {"logistic", "knn", "svm",
                                   "decision_tree", "random_forest"}
    for name, m in models.items():
        acc = m.score(X_test, y_test)
        assert acc > 0.85, f"{name} accuracy too low: {acc}"