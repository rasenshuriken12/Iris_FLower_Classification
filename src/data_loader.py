"""Load the Iris dataset."""
import pandas as pd
from sklearn.datasets import load_iris
from src.config import DATA_RAW


def load_raw_data(save_csv: bool = True) -> pd.DataFrame:
    """
    Load Iris dataset as a pandas DataFrame.

    Returns
    -------
    df : pd.DataFrame
        Columns: 4 features + 'species' (string labels)
    """
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()

    # Map numeric target -> species name
    df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
    df = df.drop(columns=["target"])

    if save_csv:
        out_path = DATA_RAW / "iris.csv"
        df.to_csv(out_path, index=False)
        print(f"💾 Saved raw data -> {out_path}")

    return df


if __name__ == "__main__":
    df = load_raw_data()
    print(df.head())
    print(f"Shape: {df.shape}")