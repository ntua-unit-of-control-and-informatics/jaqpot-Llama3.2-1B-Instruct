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

        current_prompt, previous_context = self.get_prompts(input_data)

        return StreamingResponse(self.response_generator(current_prompt, previous_context), media_type='text/event-stream')

    # The generation process
    def start_generation(self, current_prompt: str, previous_context: str, streamer):

        prompt = f"""{current_prompt}"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
        self.model = self.model.to(device)
        inputs = self.tokenizer([prompt], return_tensors="pt").to(device)
        generation_kwargs = dict(
            inputs,
            streamer=streamer,
            max_new_tokens=4096,
            eos_token_id=terminators,
            temperature=0.1
        )
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

    async def response_generator(self, current_prompt: str, previous_context: str) -> StreamingResponse:
        streamer_queue = Queue()
        streamer = CustomStreamer(streamer_queue, self.tokenizer, True)

        self.start_generation(current_prompt, previous_context, streamer)

        while True:
            value = streamer_queue.get()
            if value is None:
                break
            yield value.replace("<|eot_id|>", "")
            streamer_queue.task_done()
            await asyncio.sleep(0.1)

    def get_prompts(self, input_data):
        last_index = input_data.index[-1]

        # Get current prompt from the last row
        current_prompt = input_data.iloc[last_index]['prompt']

        # Get all previous prompts except the last one
        previous_prompts = input_data.iloc[:-1]['prompt']

        # Format previous prompts with numbering and blank lines
        context_parts = []
        for i, prompt in enumerate(previous_prompts, 1):
            context_parts.append(f"Prompt {i}:\n{prompt}")

        # Join all previous prompts with blank lines between them
        previous_context = "\n\n".join(context_parts)

        # If there's no previous context, use an empty string
        previous_context = previous_context if previous_context else ""

        return current_prompt, previous_context
