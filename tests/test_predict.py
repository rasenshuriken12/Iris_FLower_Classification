from src.data_loader import load_raw_data
from src.preprocess import split_and_scale
from src.train import train_models
from src.predict import predict


def test_predict_returns_label():
    # Train on-the-fly so files exist
    df = load_raw_data(save_csv=False)
    X_train, _, y_train, _, _, _ = split_and_scale(df, save_scaler=True)
    train_models(X_train, y_train, save=True)

    label, proba = predict([5.1, 3.5, 1.4, 0.2], model_name="svm")
    assert label in ["setosa", "versicolor", "virginica"]
    assert proba is None or len(proba) == 3