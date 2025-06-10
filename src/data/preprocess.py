from sklearn.model_selection import train_test_split
import pandas as pd
import os


def split(out_path: str, raw_path: str, test_size=0.2, random_state=42):
    df = pd.read_csv(f"{raw_path}/california_housing.csv")
    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    os.makedirs(out_path, exist_ok=True)
    X_train.to_csv(f"{out_path}/X_train.csv", index=False)
    X_test.to_csv(f"{out_path}/X_test.csv", index=False)
    y_train.to_csv(f"{out_path}/y_train.csv", index=False)
    y_test.to_csv(f"{out_path}/y_test.csv", index=False)


if __name__ == "__main__":
    split("data/processed", "data/raw")
