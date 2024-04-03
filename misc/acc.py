import torch
import json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


model_name = "gpt2-medium"
preference_models=[]
with open(f'data/datasets/{model_name}_train_PM_scores.jsonl') as file:
    train_data = [json.loads(line) for line in file]
with open(f'data/datasets/{model_name}_test_PM_scores.jsonl') as file:
    test_data = [json.loads(line) for line in file]

for principle in ["ethicality", "toxicity", "helpfulness", "sycophancy", "factuality"]:
    correct=0
    for data_point in train_data:
        if data_point[f"{principle}_score_chosen"] > data_point[f"{principle}_score_rejected"]:
            correct += 1
    print(f"Train accuracy for {principle}: {correct/len(train_data)}")
for principle in ["ethicality", "toxicity", "helpfulness", "sycophancy", "factuality"]:
    correct=0
    for data_point in test_data:
        if data_point[f"{principle}_score_chosen"] > data_point[f"{principle}_score_rejected"]:
            correct += 1
    print(f"Test accuracy for {principle}: {correct/len(test_data)}")

correct=0
for data_point in train_data:
    min_chosen = float("inf")
    min_rejected = float("inf")
    for principle in ["ethicality", "toxicity", "helpfulness", "sycophancy", "factuality"]:
        if data_point[f"{principle}_score_chosen"] < min_chosen:
            min_chosen = data_point[f"{principle}_score_chosen"]
        if data_point[f"{principle}_score_rejected"] < min_rejected:
            min_rejected = data_point[f"{principle}_score_rejected"]
    if min_chosen > min_rejected:
        correct += 1
print(f"Train accuracy for min: {correct/len(train_data)}")
correct=0
for data_point in test_data:
    min_chosen = float("inf")
    min_rejected = float("inf")
    for principle in ["ethicality", "toxicity", "helpfulness", "sycophancy", "factuality"]:
        if data_point[f"{principle}_score_chosen"] < min_chosen:
            min_chosen = data_point[f"{principle}_score_chosen"]
        if data_point[f"{principle}_score_rejected"] < min_rejected:
            min_rejected = data_point[f"{principle}_score_rejected"]
    if min_chosen > min_rejected:
        correct += 1
print(f"Test accuracy for min: {correct/len(test_data)}")

for principle in ["ethicality", "toxicity", "helpfulness", "sycophancy", "factuality"]:
    with open(f'data/datasets/hh_train_{principle}_feedback.jsonl',"r",encoding='utf-8') as file:
        train_data = [json.loads(line) for line in file]
    correct=0
    for data_point in train_data:
        if data_point["logits_A"] > data_point["logits_B"]:
            correct += 1
    print(f"Train feedback accuracy for {principle}: {correct/len(train_data)}")

for principle in ["ethicality", "toxicity", "helpfulness", "sycophancy", "factuality"]:
    with open(f'data/datasets/hh_test_{principle}_feedback.jsonl',"r",encoding='utf-8') as file:
        test_data = [json.loads(line) for line in file]
    correct=0
    for data_point in test_data:
        if data_point["logits_A"] > data_point["logits_B"]:
            correct += 1
    print(f"Test feedback accuracy for {principle}: {correct/len(test_data)}")

    