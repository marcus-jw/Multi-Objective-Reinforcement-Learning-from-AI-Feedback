from LoRA_hotswapping_PM import PreferenceModelHotswapper
from MORL_scalarizer import MORLScalarizer
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from torch.optim import Adam
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
)

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, create_reference_model, set_seed
from trl.core import LengthSampler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset, TensorDataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


preference_models = PreferenceModelHotswapper('gpt2-medium', 'Preference Model LoRAs')
dataset = load_dataset("Anthropic/hh-rlhf", split="train")
preference_scores_chosen = []
preference_scores_rejected = []

input_chosen = [row['chosen'] for row in dataset]
input_rejected = [row['rejected'] for row in dataset]


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        return self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=512)  # TODO check max length


# Create datasets
chosen_dataset = TextDataset(input_chosen, preference_models.tokenizer)
rejected_dataset = TextDataset(input_rejected, preference_models.tokenizer)

# Create DataLoaders
batch_size = 16  
dataloader_chosen = DataLoader(chosen_dataset, batch_size=batch_size)
dataloader_rejected = DataLoader(rejected_dataset, batch_size=batch_size)

# Process batches
preference_scores_chosen = []
preference_scores_rejected = []
print("Processing chosen answers")
for batch in tqdm(dataloader_chosen):
    input_ids = batch['input_ids'].squeeze(1).to(device) 
    attention_mask = batch['attention_mask'].squeeze(1).to(device)  
    batch_scores = preference_models.compute_scores(input_ids, attention_mask)
    preference_scores_chosen.extend(batch_scores)
    
print("Processing rejected answers")
for batch in tqdm(dataloader_rejected):
    input_ids = batch['input_ids'].squeeze(1).to(device) 
    attention_mask = batch['attention_mask'].squeeze(1).to(device)  
    batch_scores = preference_models.compute_scores(input_ids, attention_mask)
    preference_scores_rejected.extend(batch_scores)


# Prepare the data
preference_scores = preference_scores_chosen + preference_scores_rejected
output = [1] * len(preference_scores_chosen) + [0] * len(preference_scores_rejected)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(preference_scores, output, test_size=0)

logreg = LogisticRegression()

# Fit the logistic regression model
logreg.fit(X_train, y_train)

# Save the coefficients to a text file
coefficients = logreg.coef_
with open("preference_weights.txt", "w") as file:
    preferences = ["conciseness", "ethical", "factual", "honesty", "legal", "racism", "relevance", "sexism", "sycophancy", "toxicity", "truthful", "usefulness", "violence", "x-risk"]
    for preference, coefficient in zip(preferences, coefficients):
        file.write(f"{preference}: {coefficient}\n")
# Calculate the accuracy of the model
y_pred = logreg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
