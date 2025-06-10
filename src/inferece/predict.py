import pandas as pd
from src.models.train import train_best
# from src.models.registry import get_models
# from src.features.pipeline import numeric_pipeline

pipeline = None


def load_pipeline(best_name: str):
    global pipeline
    pipeline = train_best(
        pd.read_csv("data/processed/X_train.csv"),
        pd.read_csv("data/processed/y_train.csv").squeeze(),
        best_name,
    )


def predict_house_value(features: dict) -> float:
    df = pd.DataFrame([features])
    return float(pipeline.predict(df)[0])
