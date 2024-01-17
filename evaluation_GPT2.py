from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import json
import numpy as np
from datasets import load_dataset
import torch
from tqdm import tqdm
import math
# Load the fine-tuned GPT-2 model
# model_path = 'Trained Models/GPT-2-XL-linear-no-syc' 
# model_name = 'gpt2-xl'
# config = GPT2Config.from_json_file(f"{model_path}/config.json")
# model = GPT2LMHeadModel.from_pretrained(model_path, config=config)
# tokenizer = GPT2Tokenizer.from_pretrained(model_name)

#load GPT-2-medium
model_name = 'gpt2-XL'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)


# Load the HH-RLHF dataset
dataset = []
with open('Data/hh-rlhf-test-extracted.jsonl', 'r', encoding='utf-8') as infile:
    for line in infile:
        input_dict = json.loads(line.strip())
        dataset.append(input_dict)


# Function to calculate the probability of a sequence
def sequence_probability(model, tokenizer, sequence):
    inputs = tokenizer.encode(sequence[:1024], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(inputs, labels=inputs)
    loss = outputs.loss
    return np.exp(-loss.item())

# Empirical Mutual Information calculation
def empirical_mutual_information(query, response, model, tokenizer):
    combined_sequence = query + " " + response
    p_a_given_q = sequence_probability(model, tokenizer, combined_sequence)
    p_a = sequence_probability(model, tokenizer, response)
    emi = math.log(p_a_given_q / p_a, 2)
    return emi

# Evaluate Responses and Compare
correct_count = 0
for row in tqdm(dataset):
    prompt = row['prompt']
    response_chosen = row['chosen']
    response_rejected = row['rejected']
    score_chosen = empirical_mutual_information(prompt,response_chosen, model, tokenizer)
    score_rejected = empirical_mutual_information(prompt,response_rejected, model, tokenizer)

    if score_chosen > score_rejected:
        correct_count += 1

# Compute Accuracy
accuracy = correct_count / len(dataset)
result_dict = {
    #"model": model_path.split("/")[1],
    "model": model_name,
    "accuracy": accuracy
}
# Write Accuracy to a Text File
with open('accuracy_results.jsonl', 'a') as file:
    result_json_str = json.dumps(result_dict)
    file.write(f"{result_json_str}\n")

print(f'Accuracy written to accuracy_results.txt')