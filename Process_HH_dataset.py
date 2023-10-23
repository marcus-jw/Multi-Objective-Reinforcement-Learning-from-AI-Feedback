from datasets import load_dataset, Dataset
import pandas as pd

# This function extracts the prompts used in the hh-rlhf dataset
def extract_prompts_from_split(split_name):
    dataset = load_dataset("Anthropic/hh-rlhf", split=split_name)
    prompts = []
    for row in dataset:
        prompts.append(extract_until_last_occurrence(row["chosen"]))
    
    df = pd.DataFrame({"prompt": prompts})
    df.to_json(f"Data/hh-rlhf-{split_name}-extracted.jsonl", orient='records', lines=True)

def extract_until_last_occurrence(text, substring="Assistant:"):
    last_index = text.rfind(substring)
    
    if last_index == -1:  # substring not found
        return ""
    
    end_index = last_index + len(substring)
    return text[:end_index]

# Extract prompts from train split
extract_prompts_from_split("train")

# Extract prompts from test split
extract_prompts_from_split("test")
    
    