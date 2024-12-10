from typing import Any

import joblib
import pandas as pd
from jaqpot_api_client.models.prediction_request import PredictionRequest
from jaqpot_api_client.models.prediction_response import PredictionResponse


class ModelService:
    def __init__(self):
        self.model = joblib.load('model.pkl')
        self.tokenizer = joblib.load('tokenizer.pkl')

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        # Convert input list to DataFrame
        input_data = pd.DataFrame(request.dataset.input)

        # Get dependent feature keys
        dependent_feature_keys = [feature.key for feature in request.model.dependent_features]

        # Create prediction results
        prediction_results = []
        for i in range(len(input_data)):
            prompt = input_data.iloc[i]['prompt']

            # Encode the prompt and create attention mask
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {key: value.to('cpu') for key, value in inputs.items()}

            # Generate the output
            tokens = self.model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=50, )[0]
            output = self.tokenizer.decode(
                tokens,
                skip_special_tokens=True
            )

            prediction = {
                "output": output[len(prompt):]
            }

            # Create dict with output features as keys
            prediction_dict = {
                feature_key: prediction[feature_key]
                for feature_key in dependent_feature_keys
            }

            # Add metadata
            prediction_dict["jaqpotMetadata"] = {
                "jaqpotRowId": input_data['jaqpotRowId'].iloc[i]
            }

            prediction_results.append(prediction_dict)

        return PredictionResponse(predictions=prediction_results)
