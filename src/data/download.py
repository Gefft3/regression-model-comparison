from sklearn.datasets import fetch_california_housing
import os

def download_raw(out_path: str):
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    os.makedirs(out_path, exist_ok=True)
    df.to_csv(f"{out_path}/california_housing.csv", index=False)

if __name__ == "__main__":
    download_raw("data/raw")
