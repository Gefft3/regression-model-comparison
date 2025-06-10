import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from joblib import dump
from .registry import get_models
from src.features.pipeline import numeric_pipeline
from src.config import PROCESSED_DATA_PATH, MODEL_DIR, MODEL_PATH

os.makedirs(MODEL_DIR, exist_ok=True)


def compare_models(X, y, cv=10):
    """
    Compare models using cross-validation and calculate RMSE metrics.

    Parameters:
    - X: Features dataset.
    - y: Target dataset.
    - cv: Number of cross-validation folds.

    Returns:
    - DataFrame with model names, mean RMSE, and RMSE standard deviation.
    """
    results = []
    for name, model in get_models().items():
        pipe = Pipeline([("preproc", numeric_pipeline()), ("model", model)])
        # Use cross_val_score to calculate negative MSE scores 
        neg_mse_scores = cross_val_score(
            pipe, X, y, scoring="neg_mean_squared_error", cv=cv
        )
        rmse_scores = np.sqrt(-neg_mse_scores)  # Convert negative MSE to RMSE
        results.append(
            {
                "model": name,
                "rmse_mean": rmse_scores.mean(),
                "rmse_std": rmse_scores.std(),
            }
        )
    return pd.DataFrame(results).sort_values("rmse_mean")


def train_best(X, y, best_name):
    """
    Train the best model and save the pipeline.

    Parameters:
    - X: Features dataset.
    - y: Target dataset.
    - best_name: Name of the best model to train.

    Returns:
    - Trained pipeline.
    """
    model = get_models()[best_name]
    pipe = Pipeline([("preproc", numeric_pipeline()), ("model", model)])
    pipe.fit(X, y)
    dump(pipe, MODEL_PATH)
    return pipe


if __name__ == "__main__":
    # Load training data
    X = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "X_train.csv"))
    y = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "y_train.csv")).squeeze()

    # Compare models and select the best one
    df_results = compare_models(X, y)
    best = df_results.iloc[0]["model"]

    # Train and save the best model
    train_best(X, y, best)
    print(f"Trained and saved pipeline: {best} -> {MODEL_PATH}")
