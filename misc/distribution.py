import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# Read the JSON lines from the file
with open('data/datasets/gpt2-medium_test_PM_scores.jsonl', "r", encoding='utf-8') as file:
    data = [json.loads(line) for line in file]

principles = ["ethicality", "toxicity", "helpfulness", "sycophancy", "factuality"]
categories = ['Chosen', 'Rejected']
dists = {principle: {'Chosen': [], 'Rejected': []} for principle in principles}

for principle in principles:
    for datapoint in data:
        dists[principle]['Chosen'].append(datapoint[f"{principle}_score_chosen"])
        dists[principle]['Rejected'].append(datapoint[f"{principle}_score_rejected"])

plt.figure(figsize=(10, 8))

for index, principle in enumerate(principles):
    scores = dists[principle]['Chosen'] + dists[principle]['Rejected']
    categories_list = ['Chosen'] * len(dists[principle]['Chosen']) + ['Rejected'] * len(dists[principle]['Rejected'])
    sns.violinplot(x=categories_list, y=scores, cut=0, inner='point')

    plt.xticks([0, 1], [f"{principle} Chosen", f"{principle} Rejected"])
    plt.ylabel('Scores')
    plt.xlabel('Principles')

    if index < len(principles) - 1:
        plt.figure(figsize=(10, 8))

plt.tight_layout()
plt.show()
