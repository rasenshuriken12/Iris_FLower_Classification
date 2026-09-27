"""End-to-end training pipeline."""
import sys
from pathlib import Path

# Make 'src' importable when running as a script
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_raw_data
from src.preprocess import split_and_scale
from src.train import train_models
from src.evaluate import evaluate_model, plot_feature_importance


def main():
    print("=" * 60)
    print("🌸 Iris Flower Classification — Training Pipeline")
    print("=" * 60)

    # 1. Load
    df = load_raw_data(save_csv=True)

    # 2. Preprocess
    X_train, X_test, y_train, y_test, scaler, feature_names = split_and_scale(df)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # 3. Train
    models = train_models(X_train, y_train, save=True)

    # 4. Evaluate
    results = {}
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test, name)
        results[name] = metrics["accuracy"]

    # 5. Feature importance (best tree model)
    plot_feature_importance(models["random_forest"], feature_names,
                            model_name="random_forest")

    # 6. Summary
    print("\n" + "=" * 60)
    print("📈 Final Accuracies")
    print("=" * 60)
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        print(f"{name:15s}: {acc:.4f}")


if __name__ == "__main__":
    main()