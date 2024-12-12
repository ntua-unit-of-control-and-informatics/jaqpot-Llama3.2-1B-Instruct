from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from jaqpot_api_client.models.prediction_request import PredictionRequest
from jaqpot_api_client.models.prediction_response import PredictionResponse

from src.loggers.logger import logger
from src.loggers.log_middleware import LogMiddleware
from src.model import ModelService

from fastapi.responses import StreamingResponse

model_service: ModelService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    global model_service
    model_service = ModelService()
    yield

app = FastAPI(title="ML Model API", lifespan=lifespan)
app.add_middleware(LogMiddleware)


@app.post("/infer", response_model=PredictionResponse)
def infer(req: PredictionRequest) -> StreamingResponse:
    try:
        return model_service.infer(req)
    except Exception as e:
        logger.error("Prediction request for model failed " + str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    return {"status": "OK"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005, log_config=None)
