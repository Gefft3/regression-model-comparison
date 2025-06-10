from sklearn.datasets import fetch_california_housing
import os

def download_raw(out_path: str):
    """
    Download the California housing dataset and save it as a CSV file.
    Parameters:
        out_path (str): The directory where the dataset will be saved.
    Returns:
        None: The function saves the dataset as a CSV file in the specified output path.
    """
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    os.makedirs(out_path, exist_ok=True)
    df.to_csv(f"{out_path}/california_housing.csv", index=False)

if __name__ == "__main__":
    download_raw("data/raw")
