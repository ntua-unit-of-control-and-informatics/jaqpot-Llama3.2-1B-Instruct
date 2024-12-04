from typing import Any

import joblib
import pandas as pd
from jaqpot_api_client.models.prediction_request import PredictionRequest
from jaqpot_api_client.models.prediction_response import PredictionResponse


class ModelService:
    def __init__(self):
        self.model = joblib.load('model.pkl')

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        # Convert input list to DataFrame
        input_data = pd.DataFrame(request.dataset.input)

        # Get feature columns (excluding jaqpotRowId)
        feature_cols = [col for col in input_data.columns if col != 'jaqpotRowId']

        # Make predictions for all rows at once
        predictions = self.model.predict(input_data[feature_cols])
        probabilities = self.model.predict_proba(input_data[feature_cols])

        # Get dependent feature keys
        dependent_feature_keys = [feature.key for feature in request.model.dependent_features]

        # Create prediction results
        prediction_results = []
        for i in range(len(input_data)):
            # Create dict with output features as keys
            prediction_dict = {
                feature_key: value
                for feature_key, value in zip(dependent_feature_keys, predictions[i].ravel().tolist())
            }

            # Add metadata
            prediction_dict["jaqpotMetadata"] = {
                "probabilities": probabilities[i].tolist(),
                "jaqpotRowId": input_data['jaqpotRowId'].iloc[i]
            }

            prediction_results.append(prediction_dict)

        return PredictionResponse(predictions=prediction_results)
