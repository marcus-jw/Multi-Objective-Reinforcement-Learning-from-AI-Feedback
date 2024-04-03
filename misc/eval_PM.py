import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import json

# Set batch size and number of workers
batch_size = 64  # Adjust this based on your GPU/CPU memory availability
num_workers = 12  # Adjust this based on your CPU cores
train_test = "test" 
path = f'data/datasets/hh-rlhf-{train_test}-extracted.jsonl'
model_name = "gpt2-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"

if getattr(tokenizer, "pad_token", None) is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load data
with open(path) as file:
    data = [json.loads(line) for line in file]

class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item["prompt"] + item["chosen"], item["prompt"] + item["rejected"]

def collate_fn(batch):
    chosen_texts, rejected_texts = zip(*batch)
    chosen_batch = tokenizer(list(chosen_texts), padding=True, truncation=True, max_length=512, return_tensors="pt")
    rejected_batch = tokenizer(list(rejected_texts), padding=True, truncation=True, max_length=512, return_tensors="pt")
    return chosen_batch, rejected_batch

def main():
    dataset = TextDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    for principle in ["ethicality", "toxicity", "helpfulness", "sycophancy", "factuality"]:
        preference_model = AutoModelForSequenceClassification.from_pretrained(f"data/PMs/{model_name}_{principle}/final").to(device)
        preference_model.eval()
        print(f"Loaded model for {principle}")

        chosen_scores = []
        rejected_scores = []

        with torch.no_grad():
            for chosen_batch, rejected_batch in tqdm(dataloader):
                chosen_batch = {k: v.to(device) for k, v in chosen_batch.items()}
                rejected_batch = {k: v.to(device) for k, v in rejected_batch.items()}

                chosen_logits = preference_model(**chosen_batch).logits
                rejected_logits = preference_model(**rejected_batch).logits

                chosen_scores.extend(chosen_logits[:, 0].cpu().numpy())
                rejected_scores.extend(rejected_logits[:, 0].cpu().numpy())

        for i, (chosen_score, rejected_score) in enumerate(zip(chosen_scores, rejected_scores)):
            data[i][f"{principle}_score_chosen"] = chosen_score.item()
            data[i][f"{principle}_score_rejected"] = rejected_score.item()

    # Save the data with scores
    with open(f"data/datasets/{model_name}_{train_test}_PM_scores.jsonl", "w") as file:
        for line in data:
            file.write(json.dumps(line) + "\n")

if __name__ == "__main__":
    main()