import json
import numpy as np
from itertools import combinations
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns
# Load predictions from a file
def load_principle_predictions(in_filepath):
    if "extracted" in in_filepath:
        with open(in_filepath, 'r', encoding='utf-8') as in_file:
            return np.array([1 for line in in_file])
    else:
        with open(in_filepath, 'r', encoding='utf-8') as in_file:
            preds = [json.loads(line)["logits_A"] > json.loads(line)["logits_B"] for line in in_file]
        return np.array([1 if pred else 0 for pred in preds])

def calculate_and_store_agreement_proportions(filepaths):
    # Load predictions for all files
    predictions = {filepath: load_principle_predictions(filepath) for filepath in filepaths}
    filenames = list(predictions.keys())
    # Initialize an empty matrix for storing agreement proportions
    agreement_matrix = np.zeros((len(filenames), len(filenames)))
    
    for i, (file1, preds1) in enumerate(predictions.items()):
        for j, (file2, preds2) in enumerate(predictions.items()):
            if i == j:
                agreement_matrix[i, j] = 1.0  # Self-agreement is always 1
            elif i < j:  # To avoid redundant calculations
                agreement = np.mean(preds1 == preds2)
                agreement_matrix[i, j] = agreement
                agreement_matrix[j, i] = agreement  # Symmetric matrix
    
    # Mapping filepaths to a simpler name
    simplified_names = [f"{name.split('_')[-2]}" if "extracted" not in name else "true label" for i, name in enumerate(filenames)]
    return agreement_matrix, simplified_names

base = "data/datasets/hh_train_"
filepaths = [
    base + "ethicality_feedback.jsonl",
    base + "toxicity_feedback.jsonl",
    base + "helpfulness_feedback.jsonl",
    base + "sycophancy_feedback.jsonl",
    base + "factuality_feedback.jsonl",
    base + "relevance_feedback.jsonl",
    base + "bias_feedback.jsonl",
    base + "conciseness_feedback.jsonl",
    base + "context_feedback.jsonl",
    base + "detail_feedback.jsonl",
    base + "empathy_feedback.jsonl",
    base + "understandability_feedback.jsonl",
    "data/datasets/hh-rlhf-train-extracted.jsonl",
]


agreement_matrix, simplified_names = calculate_and_store_agreement_proportions(filepaths)

plt.figure(figsize=(10, 8))
sns.heatmap(agreement_matrix, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=simplified_names, yticklabels=simplified_names)
plt.title("Correlations between principle feedback")
plt.tight_layout()
plt.show()
