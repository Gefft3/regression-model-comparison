import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from .registry import get_models
from src.features.pipeline import numeric_pipeline

def compare_models(X, y, cv=10):
    results = []
    for name, model in get_models().items():
        pipe = Pipeline([("preproc", numeric_pipeline()), ("model", model)])
        scores = cross_val_score(pipe, X, y, scoring="neg_mean_squared_error", cv=cv)
        rmse = np.sqrt(-scores)
        results.append({"model": name, "rmse_mean": rmse.mean(), "rmse_std": rmse.std()})
    return pd.DataFrame(results).sort_values("rmse_mean")

def train_best(X, y, best_name):
    model = get_models()[best_name]
    pipe = Pipeline([("preproc", numeric_pipeline()), ("model", model)])
    return pipe.fit(X, y)

if __name__ == "__main__":
    df = pd.read_csv("data/processed/X_train.csv")
    y = pd.read_csv("data/processed/y_train.csv").squeeze()
    df_results = compare_models(df, y)
    best = df_results.iloc[0]["model"]
    _ = train_best(df, y, best)
