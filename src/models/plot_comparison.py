import matplotlib.pyplot as plt
import pandas as pd
from src.config import MODEL_DIR


def main():
    df = pd.read_csv(f"{MODEL_DIR}/model_comparison.csv")
    plt.barh(df["model"], df["rmse_mean"], xerr=df["rmse_std"])
    plt.xlabel("RMSE")
    plt.title("Model Comparison")
    plt.grid(axis="x")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
