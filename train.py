import joblib
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "./llama"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

model.to("cpu")

# Save the model
joblib.dump(model, 'model.pkl')
joblib.dump(tokenizer, 'tokenizer.pkl')
