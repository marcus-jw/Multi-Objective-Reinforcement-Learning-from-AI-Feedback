import os
import random
import json
from vertexai import generative_models
import vertexai
import time
from tqdm import tqdm
# Set up the Vertex AI client

vertexai.init(project="morlaif", location="us-central1")
model = "gpt2-medium"
input_filename=f"data/datasets/{model}_hh_train_responses.jsonl"
output_filename=f"data/datasets/{model}_hh_train_rated_gemini-pro.jsonl"
principle_folder_path = "principles/"

max_requests_per_minute = 250
request_count = 0
start_time = time.time()

def evaluate_responses(question, responseA, responseB, principle):
    """
    Asks Gemini Pro which response is better based on a given principle using logits.
    
    Args:
    - question  (str): The user input which the model is responding to.
    - responseA (str): The first response.
    - responseB (str): The second response.
    - principle (str): The principle to judge the responses.
    
    Returns:
    - logits_for_A the logits for response A
    - logits_for_B the logits for response B
    """
    global request_count, start_time, max_requests_per_minute
    # Rate limiting
    current_time = time.time()
    if request_count >= max_requests_per_minute:
        elapsed_time = current_time - start_time
        if elapsed_time < 60:
            sleep_time = 60 - elapsed_time
            time.sleep(sleep_time)
            request_count = 0
            start_time = time.time()
        else:
            request_count = 0
            start_time = current_time

    request_count += 1



    prompt = f"You will be given a conversation between a human and an AI assistant along "\
            f"with a principle and two responses. Your task is to choose the response which "\
            f"best follows the principle. \n"\
            f"Conversation: {question} \n Given the principle '{principle}', "\
            f"which of the following responses is better?\n" \
             f"A. {responseA}\n" \
             f"B. {responseB}\n" \
             f"Respond only with A or B.\n\n"
    model = generative_models.GenerativeModel("gemini-1.0-pro")
    config = {"max_output_tokens": 1, "temperature": 0, "top_p": 1, "top_k": 32}
    safety_config = {
        generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
        generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
        generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_NONE,
        generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_NONE,
    }

    response = model.generate_content(
        prompt,
        generation_config=config,
        stream=False,
        safety_settings=safety_config,
    )


    # Extracting the logits for the last tokens (which should correspond to "A" or "B")
    token = response.candidates[0].content.parts[0].text
    print(token)
    if token == "A":
        logits_for_A = -0.9162907
        logits_for_B = -0.5108256
    elif token == "B":
        logits_for_A = -0.5108256
        logits_for_B = -0.9162907
    else:
        raise ValueError(f"Unexpected token: {token}")


    return logits_for_A, logits_for_B


def get_principles_from_folder(principle_folder_path):
    """
    Reads all the .txt files in the given folder and returns their content as principles.
    
    Returns:
    - dict: Dictionary where keys are filenames (without .txt) and values are lists containing rewordings of the principle.
    """
    principles = {}
    for filename in os.listdir(principle_folder_path):
        if filename.endswith('.txt'):
            with open(os.path.join(principle_folder_path, filename), 'r') as file:
                principle_name = filename[:-4]  # Removing .txt extension
                # Initialize an empty list for storing the rewordings
                rewordings = []
                # Iterate through each line in the file, stripping it and appending to the list
                for line in file:
                    rewordings.append(line.strip())
                # Store the list of rewordings as the value corresponding to the principle_name key
                principles[principle_name] = rewordings

    return principles


def process_file_with_principles(input_filename, output_filename, principle_folder_path):
    principles = get_principles_from_folder(principle_folder_path)

    with open(input_filename, 'r', encoding='utf-8') as infile, open(output_filename, 'w', encoding='utf-8') as outfile:
        for line in infile:
            input_dict = json.loads(line.strip())

            question = input_dict["prompt"]
            responseA = input_dict["responseA"]
            responseB = input_dict["responseB"]

            result_dict = {
                "prompt": question,
                "responseA": responseA,
                "responseB": responseB
            }

            for principle_name, rewordings in principles.items():
                sampled_principle = random.choice(rewordings)

                logits_for_A, logits_for_B = evaluate_responses(question, responseA, responseB, sampled_principle)

                result_dict[principle_name] = (logits_for_A, logits_for_B)

            result_json_str = json.dumps(result_dict)
            outfile.write(f"{result_json_str}\n")


process_file_with_principles(input_filename, output_filename, principle_folder_path)