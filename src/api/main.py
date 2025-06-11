from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.inference.inference import load_pipeline, predict

app = FastAPI()
pipeline = load_pipeline()


class Features(BaseModel):
    data: dict


@app.post("/predict")
def get_prediction(features: Features):
    try:
        result = predict(pipeline, features.data)
        return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
