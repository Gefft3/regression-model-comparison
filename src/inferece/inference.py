import sys
import json
import pandas as pd
from joblib import load

MODEL_PATH = "models/best_pipeline.joblib"


def load_pipeline(path=MODEL_PATH):
    return load(path)


def predict(pipeline, features: dict) -> float:
    df = pd.DataFrame([features])
    return float(pipeline.predict(df)[0])


def main():
    if len(sys.argv) != 2:
        print("Usage: python inference.py '<json_features>'")
        sys.exit(1)

    features = json.loads(sys.argv[1])
    pipeline = load_pipeline()
    prediction = predict(pipeline, features)
    print(prediction)


if __name__ == "__main__":
    main()