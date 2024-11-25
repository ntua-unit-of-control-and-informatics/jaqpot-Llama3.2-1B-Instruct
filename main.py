from typing import Union

from fastapi import FastAPI
from jaqpot_api_client.models.prediction_request import PredictionRequest
from jaqpot_api_client.models.prediction_response import PredictionResponse

from src.my_model_handler import model_infer

app = FastAPI()

@app.get("/")
def health_check():
    return {"status": "UP"}

@app.post("/infer")
def predict(req: PredictionRequest) -> PredictionResponse:
    return model_infer(req)

