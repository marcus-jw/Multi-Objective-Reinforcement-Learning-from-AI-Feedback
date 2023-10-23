import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from trl.ppo import PPOTrainer
from trl.gpt2 import GPT2HeadWithValueModel, respond_to_batch
from tqdm import tqdm
import numpy as np
import pandas as pd
tqdm.pandas()

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2-medium"  # Can be "gpt2-small", "gpt2-medium", etc.
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
gpt2_model = GPT2HeadWithValueModel(model)

# Load your data: You should have your prompts, responses, and logprobs
# This is just an example format
data = [
    {"prompt": "According to simplicity,", "response": "The sky is blue.", "logprob": -0.5},
    # ... add more data points
]

# Convert prompts and responses to tensor format
prompts = tokenizer([item["prompt"] for item in data], return_tensors="pt", padding=True, truncation=True)
responses = tokenizer([item["response"] for item in data], return_tensors="pt", padding=True, truncation=True)

# Compute rewards: Here, we'll use the negative logprobs as rewards (higher is better)
rewards = torch.tensor([-item["logprob"] for item in data])

# Fine-tune model with PPO
ppo_trainer = PPOTrainer(gpt2_model, 
                         lr=2.5e-4, 
                         clip_epsilon=0.1, 
                         vf_coef=0.1, 
                         max_grad_norm=0.5)

ppo_epochs = 3  # Number of epochs to fine-tune
for epoch in range(ppo_epochs):
    new_responses = respond_to_batch(gpt2_model, prompts)
    loss = ppo_trainer.step(prompts, responses, rewards, new_responses)
    print(f"Epoch {epoch + 1}/{ppo_epochs}, Loss: {loss.item()}")

# Save fine-tuned model
gpt2_model.save_pretrained("./fine_tuned_gpt2")