import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from src.features.pipeline import numeric_pipeline

def test_numeric_pipeline_type():
    """
    Test if the numeric_pipeline function returns a scikit-learn Pipeline object.
    """
    pipeline = numeric_pipeline()
    assert isinstance(pipeline, Pipeline), "The returned object is not a Pipeline."

def test_numeric_pipeline_steps():
    """
    Test if the numeric_pipeline has the correct steps: imputer and scaler.
    """
    pipeline = numeric_pipeline()
    steps = [step[0] for step in pipeline.steps]
    assert steps == ["imputer", "scaler"], f"Pipeline steps are incorrect: {steps}"

def test_numeric_pipeline_imputer_strategy():
    """
    Test if the SimpleImputer in the pipeline uses the 'median' strategy.
    """
    pipeline = numeric_pipeline()
    imputer = pipeline.named_steps["imputer"]
    assert isinstance(imputer, SimpleImputer), "The imputer is not an instance of SimpleImputer."
    assert imputer.strategy == "median", f"Imputer strategy is not 'median': {imputer.strategy}"

def test_numeric_pipeline_scaler():
    """
    Test if the StandardScaler is correctly included in the pipeline.
    """
    pipeline = numeric_pipeline()
    scaler = pipeline.named_steps["scaler"]
    assert isinstance(scaler, StandardScaler), "The scaler is not an instance of StandardScaler."

def test_numeric_pipeline_functionality():
    """
    Test the functionality of the numeric_pipeline with sample data.
    """
    pipeline = numeric_pipeline()
    data = np.array([[1, 2, np.nan], [4, np.nan, 6], [7, 8, 9]])
    transformed_data = pipeline.fit_transform(data)
    assert transformed_data.shape == data.shape, "Transformed data shape is incorrect."
    assert not np.isnan(transformed_data).any(), "Transformed data contains NaN values."