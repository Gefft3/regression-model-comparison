import matplotlib.pyplot as plt
import pandas as pd
from src.config import MODEL_DIR, PLOTS_DIR
import numpy as np
from sklearn.metrics import mean_squared_error
import os

def rmse(y_true, y_pred, squared=False):
    """
    Calculate the Root Mean Squared Error (RMSE).

    Parameters:
    - y_true: array-like of shape (n_samples,) - Ground truth (correct) target values.
    - y_pred: array-like of shape (n_samples,) - Estimated target values.
    - squared: bool, default=False - If True, return Mean Squared Error (MSE) instead of RMSE.

    Returns:
    - float: RMSE or MSE value.
    """
    mse = mean_squared_error(y_true, y_pred)
    return mse if squared else np.sqrt(mse)


def plot_comparison():
    df = pd.read_csv(f"{MODEL_DIR}/model_comparison.csv")
    plt.barh(df["model"], df["rmse_mean"], xerr=df["rmse_std"])
    plt.xlabel("RMSE")
    plt.title("Model Comparison")
    plt.grid(axis="x")
    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_path = f"{PLOTS_DIR}/model_comparison.png" 
    plt.savefig(plot_path)  