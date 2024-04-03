import torch
import json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import math
import numpy as np



model_name = "gpt2-medium"
preference_models=[]
if False:
    with open(f'data/datasets/{model_name}_train_PM_scores.jsonl') as file:
        train_data = [json.loads(line) for line in file]
    with open(f'data/datasets/{model_name}_test_PM_scores.jsonl') as file:
        test_data = [json.loads(line) for line in file]    
    X_train = []
    y_train=[]
    for datapoint in train_data:
        chosen_vector = []
        rejected_vector = []
        for principle in ["ethicality", "toxicity", "helpfulness", "sycophancy", "factuality"]:
            chosen_vector.append(datapoint[f"{principle}_score_chosen"]-datapoint[f"{principle}_score_rejected"])
            rejected_vector.append(datapoint[f"{principle}_score_rejected"]-datapoint[f"{principle}_score_chosen"])


        X_train.append(chosen_vector)
        y_train.append(1)
        X_train.append(rejected_vector)
        y_train.append(0)
    X_test = []
    y_test=[]
    for datapoint in test_data:
        chosen_vector = []
        rejected_vector = []
        for principle in ["ethicality", "toxicity", "helpfulness", "sycophancy", "factuality"]:
            chosen_vector.append(datapoint[f"{principle}_score_chosen"]-datapoint[f"{principle}_score_rejected"])
            rejected_vector.append(datapoint[f"{principle}_score_rejected"]-datapoint[f"{principle}_score_chosen"])
        X_test.append(chosen_vector)
        y_test.append(1)
        X_test.append(rejected_vector)
        y_test.append(0)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Test Accuracy:", accuracy)
    print("Coefficients:", model.coef_)
    y_pred = model.predict(X_train)
    accuracy = accuracy_score(y_train, y_pred)
    print("Train Accuracy:", accuracy)

with open(f'data/datasets/hh_train_helpfulness_feedback.jsonl',"r",encoding="utf-8") as file:
        train_data = [json.loads(line) for line in file]
for principle in ["ethicality", "toxicity", "helpfulness", "sycophancy", "factuality"]:
 
    with open(f'data/datasets/hh_train_{principle}_feedback.jsonl',"r",encoding="utf-8") as file:
        data = [json.loads(line) for line in file]
    for i in range(len(data)):
        train_data[i][f"{principle}_chosen"] = data[i]["logits_A"]
        train_data[i][f"{principle}_rejected"] = data[i]["logits_B"]

with open(f'data/datasets/hh_test_{principle}_feedback.jsonl',"r",encoding="utf-8") as file:
        test_data = [json.loads(line) for line in file]
for principle in ["ethicality", "toxicity", "helpfulness", "sycophancy", "factuality"]:
     
        with open(f'data/datasets/hh_test_{principle}_feedback.jsonl',"r",encoding="utf-8") as file:
            data = [json.loads(line) for line in file]
        for i in range(len(data)):
            test_data[i][f"{principle}_chosen"] = data[i]["logits_A"]
            test_data[i][f"{principle}_rejected"] = data[i]["logits_B"]
X_train = []
y_train=[]
for datapoint in train_data:
    chosen_vector = []
    rejected_vector = []
    for principle in ["ethicality", "toxicity", "helpfulness", "sycophancy", "factuality"]:
        chosen_vector.append(datapoint[f"{principle}_chosen"]-datapoint[f"{principle}_rejected"])
        rejected_vector.append(datapoint[f"{principle}_rejected"]-datapoint[f"{principle}_chosen"])
        X_train.append(chosen_vector)
        y_train.append(1)
        X_train.append(rejected_vector)
        y_train.append(0)
X_test = []
y_test=[]
for datapoint in test_data:
    chosen_vector = []
    rejected_vector = []
    for principle in ["ethicality", "toxicity", "helpfulness", "sycophancy", "factuality"]:
        #print(math.exp(datapoint[f"{principle}_chosen"]))
        chosen_vector.append(datapoint[f"{principle}_chosen"]-datapoint[f"{principle}_rejected"])
        rejected_vector.append(datapoint[f"{principle}_rejected"]-datapoint[f"{principle}_chosen"])
    X_test.append(chosen_vector)
    y_test.append(1)
    X_test.append(rejected_vector)
    y_test.append(0)

scaler = StandardScaler()

model = LogisticRegression(penalty=None,max_iter=1000,fit_intercept=True)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", test_accuracy)
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
y_pred = model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_pred)
print("Train Accuracy:", train_accuracy)


