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

# Calculate differences between two sets of predictions
def calculate_differences(p1, p2):
    return np.sum(p1 != p2)

# Calculate Pearson correlation coefficient between two sets of predictions
def calculate_pearson_correlation(p1, p2):
    return np.corrcoef(p1, p2)[0, 1]

# Main function to load multiple files and output both differences and correlations
def output_differences_and_kappa_scores(filepaths):
    # Load predictions for all files
    predictions = {filepath: load_principle_predictions(filepath) for filepath in filepaths}
    
    # Output pairwise differences and Kappa coefficients
    for (file1, preds1), (file2, preds2) in combinations(predictions.items(), 2):
        diffs = calculate_differences(preds1, preds2)
        kappa = cohen_kappa_score(preds1, preds2)
        f1, f2 = file1, file2
        if "extracted" in file1:
            f1 = "base"
        else:
            f1 = file1.split("_")[-2]
        if "extracted" in file2:
            f2 = "base"
        else:
            f2 = file2.split("_")[-2]
        print(f"Between {f1} and {f2}:")
        print(f"  Differences: {diffs}")
        print(f"  Cohen's kappa: {kappa:.3f}")
        print() 

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
    
    # Mapping filepaths to a simpler name (optional, for visualization)         
    simplified_names = [f"{name.split('_')[-2]}" if "extracted" not in name else "Base" for i, name in enumerate(filenames)]
    return agreement_matrix, simplified_names

# Example usage
base = "data/datasets/hh_train_"
filepaths = [
    base + "ethicality_feedback.jsonl",
    base + "toxicity_feedback.jsonl",
    base + "helpfulness_feedback.jsonl",
    #"data/datasets/old_hh_train_" + "helpfulness_feedback.jsonl",
    base + "sycophancy_feedback.jsonl",
    base + "factuality_feedback.jsonl",
    "data/datasets/hh-rlhf-train-extracted.jsonl"
]


# Assuming you have run the modified code to get `agreement_matrix` and `simplified_names`
agreement_matrix, simplified_names = calculate_and_store_agreement_proportions(filepaths)

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(agreement_matrix, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=simplified_names, yticklabels=simplified_names)
plt.title("Agreement Proportions Between Datasets")
plt.show()
#output_differences_and_kappa_scores(filepaths)