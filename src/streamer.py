# Contents of streamer.py file

# Importing the TextStreamer class from transformers
from transformers import TextStreamer


# Defining a custom streamer which inherits the Text Streamer
class CustomStreamer(TextStreamer):

    def __init__(self, queue, tokenizer, skip_prompt, **decode_kwargs) -> None:
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        # Queue taken as input to the class
        self._queue = queue
        self.stop_signal = None
        self.timeout = 1

    def on_finalized_text(self, text: str, stream_end: bool = False):
        # Instead of printing the text, we add the text into the queue
        self._queue.put(text)
        if stream_end:
            # Similarly we add the stop signal also in the queue to
            # break the stream
            self._queue.put(self.stop_signal)
