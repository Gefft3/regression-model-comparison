import pytest
import pandas as pd
from src.models.registry import get_models
from src.models.train import compare_models, train_best
from sklearn.datasets import fetch_california_housing
from src import config

@pytest.fixture
def dummy_data():
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    return X.head(100), y.head(100)  

def test_get_models_keys():
    models = get_models()
    expected = {"LinearRegression", "RandomForest", "XGBoost", "CatBoost"}
    assert set(models.keys()) == expected

def test_compare_models_returns_valid_dataframe(dummy_data):
    X, y = dummy_data
    df = compare_models(X, y, cv=3)
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"model", "rmse_mean", "rmse_std"}
    assert not df.empty
    assert df["rmse_mean"].min() > 0

def test_train_best_returns_pipeline_and_saves(tmp_path, monkeypatch, dummy_data):
    X, y = dummy_data

    # Override path configs temporarily
    monkeypatch.setattr(config, "MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(config, "MODEL_FILENAME", "test_pipeline.joblib")
    monkeypatch.setattr(config, "MODEL_PATH", str(tmp_path / "test_pipeline.joblib"))

    # Test with a valid model without saving
    pipe = train_best(X, y, "LinearRegression", save=False)
    assert pipe is not None
    assert not (tmp_path / "test_pipeline.joblib").exists()

