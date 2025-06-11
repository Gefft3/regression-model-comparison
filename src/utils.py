import matplotlib.pyplot as plt
import pandas as pd
from src.config import RAW_DATA_PATH, MODEL_DIR, PLOTS_DIR
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


def plot_eda():
    df = pd.read_csv(f"{RAW_DATA_PATH}/california_housing.csv")
    
    os.makedirs(PLOTS_DIR, exist_ok=True)

    X = df.drop(columns=["MedHouseVal"])

    corr = df.corr()
    plt.figure(figsize=(10, 8))
    plt.imshow(corr, cmap="coolwarm", interpolation="none")
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/correlation_matrix.png")
    plt.close("all")

    for col in X.columns:
        plt.figure(figsize=(10, 6))
        plt.hist(X[col], bins=30)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.savefig(f"{PLOTS_DIR}/frequency_{col}.png")
        plt.close("all")

if __name__ == "__main__":
    plot_eda()
    plot_comparison()
    print("Plots generated successfully.")