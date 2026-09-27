"""CLI for single-sample inference."""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.predict import predict


def main():
    parser = argparse.ArgumentParser(description="Predict iris species.")
    parser.add_argument("--features", nargs=4, type=float, required=True,
                        metavar=("SEPAL_L", "SEPAL_W", "PETAL_L", "PETAL_W"),
                        help="4 feature values in cm")
    parser.add_argument("--model", default="svm", help="Model name")
    args = parser.parse_args()

    label, proba = predict(args.features, model_name=args.model)
    print(f"\n🌸 Predicted species: {label}")
    if proba is not None:
        for cls, p in zip(sorted(set([label])), proba):
            pass
        print(f"   Probabilities: {proba}")


if __name__ == "__main__":
    main()