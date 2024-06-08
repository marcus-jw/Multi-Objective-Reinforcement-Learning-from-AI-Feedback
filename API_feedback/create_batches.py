from openai import OpenAI
import os
import random
import json
from tqdm import tqdm
import time
import argparse
import math
client = OpenAI()
train_test="train"
data_path=f"data/datasets/hh-rlhf-{train_test}-extracted"
principle="bias"
multi_objective=True 
principle_folder="principles" 
principle_name=f"{principle}" 
feedback_model="gpt-3.5-turbo-0125"
dataset_path=f"{data_path}.jsonl" 
save_path=f"data/api_requests/{principle}.jsonl"
principle_path=None
CoT=False
few_shot_path=None
batch_size=20000

def get_few_shot_examples(few_shot_path,principles,chain_of_thought=False):
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
        messages.append({"role": "user", "content": conversation})
        messages.append({"role": "assistant", "content": choice})
    return messages

def get_principles(principle_path):
    principles = []
    for principle in open(principle_path, 'r'):
        principle = json.loads(principle.strip())
        principle = principle["principle"]
        principles.append(principle)
    return principles
def get_principles_from_folder(principle_folder_path):
    with open(os.path.join(principle_folder_path, principle_name + '.txt'), 'r') as infile:
        principles = infile.readlines()
    return principles
        
def prepare_request(model,conversation, responseA, responseB,principle,id,messages=[]):
    """
    Asks the feedback model which response is better based on a given principle using logits.
    
    Args:
    - model (str): The model to use for feedback.
    - conversation (str): The conversation between the user and assistant which is to be rated.
    - responseA (str): The first response.
    - responseB (str): The second response.
    - principle (str): The principle to judge the responses.
    - id (int): The id of the request.
    - messages (list): A list of messages to be prepended, used for few-shot examples.

    Returns:
    - request (dict): The request to be sent to the feedback model.
    """
    suffixes = ["\n\nHuman:","\n\nHuman","\nHuman:","\nHuman","\n\nhuman:","\n\nhuman","\nhuman","\nhuman:","Human","human"]
    for suffix in suffixes:
        if responseA.endswith(suffix):
            responseA = responseA[:-len(suffix)]
            break  
    for suffix in suffixes:
        if responseB.endswith(suffix):
            responseB = responseB[:-len(suffix)]
            break  

    vars_dict = {"conversation": conversation, "responseA": responseA, "responseB": responseB, "principle": principle}
    with open("API_feedback/prompt.txt", "r") as file:
        prompt = file.read().format(**vars_dict)



    messages.append({"role": "user", "content": prompt})  
    
    request = {
        "custom_id": str(id),
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": messages,
            "max_tokens": 1,
            "logprobs": True,
            "top_logprobs": 5,
        },
    }
    return request



def process_dataset(input_filename, output_filename, model):
    if multi_objective:
        principles = get_principles_from_folder(principle_folder)
    else: 
        principles = get_principles(principle_path)
    if few_shot_path is not None:
        conversation = get_few_shot_examples(few_shot_path,principles)
    else:
        conversation = []
    requests = []
    with open(input_filename, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
        for index,line in enumerate(tqdm(lines)):
            input_dict = json.loads(line.strip())
            question = input_dict["prompt"]
            if "responseA" in input_dict:
                responseA = input_dict["responseA"]
                responseB = input_dict["responseB"]
            elif "chosen" in input_dict:
                responseA = input_dict["chosen"]
                responseB = input_dict["rejected"]
            principle = random.choice(principles)
            request = prepare_request(model,question, responseA, responseB, principle,messages=conversation.copy(), id=index)
            requests.append(request)

    for i in range(math.ceil(len(requests)/batch_size)):
        
        start = i*batch_size
        end = (i+1)*batch_size if (i+1)*batch_size < len(requests) else len(requests)
        requests_batch = requests[start:end]
        with open(output_filename.replace(".",f"_{i}.") , 'w', encoding='utf-8') as outfile:
            for request in requests_batch:
                result_json_str = json.dumps(request)
                outfile.write(f"{result_json_str}\n")
process_dataset(dataset_path,save_path,feedback_model)








