import pytest
from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor

from src.inference.inference import load_pipeline, predict


@pytest.fixture
def dummy_pipeline(tmp_path):
    pipe = Pipeline([("model", DummyRegressor(strategy="mean"))])
    pipe.fit([[0]], [1.23])  # treino mínimo para o DummyRegressor
    path = tmp_path / "dummy_model.joblib"
    dump(pipe, path)
    return path


def test_load_pipeline(dummy_pipeline):
    pipe = load_pipeline(str(dummy_pipeline))
    assert isinstance(pipe, Pipeline)


def test_predict_value(dummy_pipeline):
    pipe = load_pipeline(str(dummy_pipeline))
    features = {"f1": 0}
    result = predict(pipe, features)
    assert isinstance(result, float)
