import os
import random
import json
from tqdm import tqdm
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--response_path", type=str, default=None)
parser.add_argument("--dataset_path", type=str, default=None)
parser.add_argument("--save_path", type=str, default=None)
config = parser.parse_args()

with open(config.response_path, 'r', encoding='utf-8') as infile, open(config.dataset_path, 'r', encoding='utf-8') as data, open(config.save_path, 'w', encoding='utf-8') as outfile:
    dataset = data.readlines()
    dataset = [json.loads(data.strip()) for data in dataset]
    responses = infile.readlines()
    responses = [json.loads(response.strip()) for response in responses]

    # Sort the responses by id
    responses = sorted(responses, key=lambda x: x[2]["id"])


    for i in range(len(dataset)):

        dataset[i]["id"] = int(responses[i][2]["id"])
        choice = responses[i][1][0]["content"]["parts"][0]["text"].strip()

        if choice == "A":
            dataset[i]["logits_A"] = -0.5108256
            dataset[i]["logits_B"] = -0.9162907
        elif choice == "B":
            dataset[i]["logits_A"] = -0.9162907
            dataset[i]["logits_B"] = -0.5108256
    for i in range(len(dataset)):
        outfile.write(json.dumps(dataset[i], ensure_ascii=False) + '\n')