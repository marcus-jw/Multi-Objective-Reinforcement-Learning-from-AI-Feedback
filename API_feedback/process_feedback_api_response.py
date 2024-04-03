from openai import OpenAI
import os
import random
import json
from tqdm import tqdm
import time
import argparse
client = OpenAI()

parser = argparse.ArgumentParser()
parser.add_argument("--response_path", type=str, default=None)
parser.add_argument("--dataset_path", type=str, default=None)
parser.add_argument("--save_path", type=str, default=None)
config = parser.parse_args()


with open(config.response_path, 'r', encoding='utf-8') as infile, open(config.dataset_path,'r', encoding='utf-8') as data, open(config.save_path, 'w', encoding='utf-8') as outfile:
    dataset = data.readlines()
    
    dataset = [json.loads(data.strip()) for data in dataset]
    
    responses = infile.readlines()
    responses = [json.loads(response.strip()) for response in responses]
    # sort the responses by id
    responses = sorted(responses, key=lambda x: x[2]["id"])
    for i in range(len(dataset)):
        if "chosen" in dataset[i]:
            dataset[i]["responseA"] = dataset[i].pop("chosen")
            dataset[i]["responseB"] = dataset[i].pop("rejected")
        dataset[i]["id"] = int(responses[i][2]["id"])
        logprob_A = -10
        logprob_B = -10
        for token in responses[i][1]["choices"][0]["logprobs"]["content"][0]["top_logprobs"]:
            if token["token"] == "A":
                logprob_A = token["logprob"]
            if token["token"] == "B":
                logprob_B = token["logprob"]
        dataset[i]["logits_A"] = logprob_A
        dataset[i]["logits_B"] = logprob_B
    for i in range(len(dataset)):
        outfile.write(json.dumps(dataset[i], ensure_ascii=False) + '\n')





