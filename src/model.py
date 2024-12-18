import asyncio
from queue import Queue
from threading import Thread

import joblib
import pandas as pd
import torch
from fastapi.responses import StreamingResponse
from jaqpot_api_client.models.prediction_request import PredictionRequest

from src.streamer import CustomStreamer


class ModelService:
    def __init__(self):
        self.model = joblib.load('model.pkl')
        self.tokenizer = joblib.load('tokenizer.pkl')

    def infer(self, request: PredictionRequest) -> StreamingResponse:
        # Convert input list to DataFrame
        input_data = pd.DataFrame(request.dataset.input)

        last_index = input_data.index[-1]
        input_row = input_data.iloc[last_index]

        prompt = input_row['prompt']

        return StreamingResponse(self.response_generator(prompt), media_type='text/event-stream')

    # The generation process
    def start_generation(self, query: str, streamer):

        prompt = f"""{query}"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(device)
        # Update generation line:
        inputs = self.tokenizer([prompt], return_tensors="pt").to(device)
        generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=4096, temperature=0.1)
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # Generation initiator and response server

    async def response_generator(self, query: str):
        streamer_queue = Queue()
        streamer = CustomStreamer(streamer_queue, self.tokenizer, True)

        self.start_generation(query, streamer)

        while True:
            value = streamer_queue.get()
            if value is None:
                break
            yield value
            streamer_queue.task_done()
            await asyncio.sleep(0.1)
