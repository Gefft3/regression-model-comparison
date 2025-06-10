import pandas as pd
from src.data.download import download_raw
from src.data.preprocess import split


def test_download_raw_creates_csv(tmp_path):
    out_path = tmp_path / "raw"
    download_raw(str(out_path))

    csv_file = out_path / "california_housing.csv"
    assert csv_file.exists(), "CSV file not created"

    df = pd.read_csv(csv_file)
    assert not df.empty, "Downloaded CSV is empty"
    assert "MedHouseVal" in df.columns, "Expected column not found in CSV"


def test_split_creates_train_test_csvs(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"

    download_raw(str(raw_dir))
    split(str(processed_dir), str(raw_dir))

    expected_files = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]
    for fname in expected_files:
        fpath = processed_dir / fname
        assert fpath.exists(), f"{fname} not created"

        df = pd.read_csv(fpath)
        assert not df.empty, f"{fname} is empty"
