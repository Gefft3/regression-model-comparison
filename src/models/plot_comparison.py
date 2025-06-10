import matplotlib.pyplot as plt
import pandas as pd
from src.config import PROCESSED_DATA_PATH
from src.models.train import compare_models


def main():
    X = pd.read_csv(f"{PROCESSED_DATA_PATH}/X_train.csv")
    y = pd.read_csv(f"{PROCESSED_DATA_PATH}/y_train.csv").squeeze()
    df = compare_models(X, y)
    plt.barh(df["model"], df["rmse_mean"], xerr=df["rmse_std"])
    plt.xlabel("RMSE")
    plt.title("Model Comparison")
    plt.grid(axis="x")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
