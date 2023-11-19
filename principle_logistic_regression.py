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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


preference_models = PreferenceModelHotswapper('gpt2-medium', '/Preference Models')
dataset = load_dataset("Anthropic/hh-rlhf", split="train")

preference_scores_chosen = []
preference_scores_rejected = []
for row in tqdm(dataloader):
    preference_scores_chosen.append(preference_models.compute_scores({
        "input_ids": torch.stack(row["chosen"]).to(device)
    }))
    preference_scores_rejected.append(preference_models.compute_scores({
        "input_ids": torch.stack(row["rejected"]).to(device)
    }))
    
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
