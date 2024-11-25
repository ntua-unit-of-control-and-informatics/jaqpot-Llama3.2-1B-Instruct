from jaqpot_api_client.models.prediction_request import PredictionRequest
from jaqpot_api_client.models.prediction_response import PredictionResponse


def model_infer(req: PredictionRequest) -> PredictionResponse:
    return PredictionResponse(predictions=[])

