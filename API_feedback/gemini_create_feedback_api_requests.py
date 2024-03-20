import os
import random
import json
from tqdm import tqdm
import time
import argparse
from vertexai import generative_models


parser = argparse.ArgumentParser()
parser.add_argument("--multi-objective", type=str, default="True")
parser.add_argument("--principle_folder", type=str, default=None)
parser.add_argument("--principle_name", type=str, default=None)
parser.add_argument("--feedback_model", type=str, default="gemini-1.0-pro")
parser.add_argument("--CoT", type=bool, default=False)
parser.add_argument("--few_shot_path", type=str, default=None)
parser.add_argument("--principle_path", type=str, default=None)
parser.add_argument("--dataset_path", type=str, default=None)
parser.add_argument("--save_path", type=str, default=None)
config = parser.parse_args()

def get_few_shot_examples(few_shot_path, principles, chain_of_thought=False):
    messages = []
    for example in open(few_shot_path, 'r'):
        example = json.loads(example.strip())
        principle = random.choice(principles)
        prompt = example["prompt"]
        options = example["options"]
        choice = example["choice"]
        CoT = example["CoT"]
        ending = "Please respond only with A or B. The answer is:\n\n"
        if chain_of_thought:
            conversation = prompt + principle + options
            choice = CoT + choice
        else:
            conversation = prompt + principle + options + ending
        messages.append({"role": "USER", "parts": [{"text": conversation}]})
        messages.append({"role": "MODEL", "parts": [{"text": choice}]})
    return messages

def get_principles(principle_path):
    principles = []
    for principle in open(principle_path, 'r'):
        principle = json.loads(principle.strip())
        principle = principle["principle"]
        principles.append(principle)
    return principles

def get_principles_from_folder(principle_folder_path):
    with open(os.path.join(principle_folder_path, config.principle_name + '.txt'), 'r') as infile:
        principles = infile.readlines()
    return principles

def prepare_request(conversation, responseA, responseB, principle, messages=[], metadata=None):
    """
    Asks the feedback model which response is better based on a given principle using logits.
    
    Args:
    - conversation (str): The conversation between the user and assistant which is to be rated.
    - responseA (str): The first response.
    - responseB (str): The second response.
    - principle (str): The principle to judge the responses.
    - messages (list): A list of messages to be prepended, used for few-shot examples.
    - metadata (dict): A dictionary containing metadata such as id (parallel api requests will change the order of the dataset).
    
    Returns:
    - request (dict): The request to be sent to the feedback model.
    """
    suffixes = ["\n\nHuman:", "\n\nHuman", "\nHuman:", "\nHuman", "\n\nhuman:", "\n\nhuman", "\nhuman", "\nhuman:", "Human", "human"]
    for suffix in suffixes:
        if responseA.endswith(suffix):
            responseA = responseA[:-len(suffix)]
            break
    for suffix in suffixes:
        if responseB.endswith(suffix):
            responseB = responseB[:-len(suffix)]
            break
    prompt = f"Consider the following conversation between a human and an assistant: \n" \
             f"Conversation: {conversation} \n '{principle}', " \
             f"Options: \n" \
             f"Option A. {responseA}\n" \
             f"Option B. {responseB}\n" \
             f"Please respond only with A or B. The answer is:\n\n"
    messages.append({"role": "USER", "parts": [{"text": prompt}]})
    
    request = {
        "contents": messages,
        "generation_config": {
            "temperature": 0.2,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 1
        },
        "metadata": metadata
    }
    return request

def process_dataset(input_filename, output_filename, model):
    if config.multi_objective == "True":
        print(type(config.multi_objective))
        principles = get_principles_from_folder(config.principle_folder)
    else:
        principles = get_principles(config.principle_path)
    if config.few_shot_path is not None:
        conversation = get_few_shot_examples(config.few_shot_path, principles)
    else:
        conversation = []
    with open(input_filename, 'r', encoding='utf-8') as infile, open(output_filename, 'w', encoding='utf-8') as outfile:
        lines = infile.readlines()
        for index, line in enumerate(tqdm(lines)):
            input_dict = json.loads(line.strip())
            question = input_dict["prompt"]
            responseA = input_dict["responseA"]
            responseB = input_dict["responseB"]
            principle = random.choice(principles)
            request = prepare_request(question, responseA, responseB, principle, messages=conversation.copy(), metadata={"id": index, "principle": principle})
            result_json_str = json.dumps(request)
            outfile.write(f"{result_json_str}\n")

process_dataset(config.dataset_path, config.save_path, config.feedback_model)