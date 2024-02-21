from openai import OpenAI
import os
import random
import json
client = OpenAI()

principle_path="Constitutional AI/anthropic_constitution.jsonl"
few_shot_path = "Constitutional AI/anthropic_few_shot_examples.jsonl"
feedback_model= "gpt-3.5-turbo"
def get_few_shot_principles(few_shot_path,principles,chain_of_thought=False):
    messages = []
    for example in open(few_shot_path, 'r'):
        example = json.loads(example.strip())
        principle = random.choice(principles)
        prompt = example["prompt"]
        options = example["options"]
        choice = example["choice"]
        CoT = example["CoT"]
        if chain_of_thought:
            choice = CoT + choice
        conversation = prompt + principle + options
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

def evaluate_responses(model,conversation, responseA, responseB, principle,messages=[]):
    """
    Asks the feedback model which response is better based on a given principle using logits.
    
    Args:
    - model (str): The model to use for feedback.
    - conversation (str): The conversation between the user and assistant which is to be rated.
    - responseA (str): The first response.
    - responseB (str): The second response.
    - principle (str): The principle to judge the responses.
    - messages (list): A list of messages to be prepended, used for few-shot examples.
    
    Returns:
    - logits_for_A the logits for response A
    - logits_for_B the logits for response B
    """
    
    prompt = f"Consider the following conversation between a human and an assistant: \n"\
            f"Conversation: {conversation} \n '{principle}', "\
            "Options: \n" \
             f"A. {responseA}\n" \
             f"B. {responseB}\n" \
             f"Please respond only with A or B. The answer is:\n\n"
    messages.append({"role": "user", "content": prompt})  
    response = client.chat.completions.create(
        messages=messages,
        model=model,
        max_tokens=1,
        logprobs=True,
        top_logprobs=5,
        n=1,
    )
    print(response)

    # Extracting the logits for the last tokens (which should correspond to "A" or "B")
    choices = response.choices[0]
    logprobs = choices['logprobs']['top_logprobs'][0]
    print(logprobs)
    logits_for_A = logprobs.get('A', None)
    logits_for_B = logprobs.get('B', None)

    return logits_for_A,logits_for_B


def process_dataset(input_filename, output_filename, model):
    principles = get_principles(principle_path)
    conversation = get_few_shot_principles(few_shot_path,principles)

    with open(input_filename, 'r', encoding='utf-8') as infile, open(output_filename, 'w', encoding='utf-8') as outfile:
        for line in infile:
            input_dict = json.loads(line.strip())
            question = input_dict["Prompt"]
            responseA = input_dict["ResponseA"]
            responseB = input_dict["ResponseB"]
            principle = random.choice(principles)
            logits_for_A,logits_for_B = evaluate_responses(model,question, responseA, responseB, principle,messages=conversation)
            result_dict = {
                "Prompt": question,
                "ResponseA": responseA,
                "ResponseB": responseB
            }
            result_dict["logits_A"] = logits_for_A
            result_dict["logits_B"] = logits_for_B
            
            result_json_str = json.dumps(result_dict)
            outfile.write(f"{result_json_str}\n")

#../Data/hh-rlhf-train-extracted.jsonl
     
process_dataset('Data/test.jsonl', 'Data/CAI-hh-rlhf-train-rated.jsonl',feedback_model)






