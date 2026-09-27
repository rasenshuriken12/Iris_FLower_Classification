from src.data_loader import load_raw_data


def test_shape():
    df = load_raw_data(save_csv=False)
    assert df.shape == (150, 5)  # 4 features + species


def test_species_labels():
    df = load_raw_data(save_csv=False)
    assert set(df["species"].unique()) == {"setosa", "versicolor", "virginica"}


def test_no_nulls():
    df = load_raw_data(save_csv=False)
    assert df.isnull().sum().sum() == 0