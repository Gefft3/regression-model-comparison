import sys
import json
import pandas as pd
from joblib import load
from src.config import MODEL_PATH


def load_pipeline(path=MODEL_PATH):
    """
    Load the trained model pipeline from the specified path.
    Parameters:
        path (str): Path to the model file. Defaults to MODEL_PATH from config.
    Returns:
        Pipeline: The loaded model pipeline.
    """
    return load(path)


def predict(pipeline, features: dict) -> float:
    """
    Predict the target value using the provided model pipeline and features.
    Parameters:
        pipeline (Pipeline): The trained model pipeline.
        features (dict): A dictionary of features to predict on.
    Returns:
        float: The predicted target value.
    """
    df = pd.DataFrame([features])
    return float(pipeline.predict(df)[0])


def main():
    if len(sys.argv) != 2:
        print('Usage: python -m src.inference.inference "<json_features>"')
        sys.exit(1)

    features = json.loads(sys.argv[1])
    pipeline = load_pipeline()
    print(predict(pipeline, features))


if __name__ == "__main__":
    main()
