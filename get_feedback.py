import openai
import os

# Read the API key from a file outside repo
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'openai_key.txt'), 'r') as key_file:
    openai.api_key = key_file.read().strip()

principle_folder_path="Principles/"

def evaluate_responses(question, responseA, responseB, principle):
    """
    Asks GPT-3.5 which response is better based on a given principle using logits.
    
    Args:
    - question  (str): The user input which the model is responding to.
    - responseA (str): The first response.
    - responseB (str): The second response.
    - principle (str): The principle to judge the responses.
    
    Returns:
    - logits_for_A the logits for response A
    - logits_for_B the logits for response B
    """
    
    prompt = f"Given the principle '{principle}', which of the following responses is better?\n\n" \
             f"A. {responseA}\n" \
             f"B. {responseB}\n" \
             f"Respond only with A or B.\n\n" #TODO check newlines #TODO add question

    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=1,
        logprobs=5,
        n=1,
    )
    
    # Extracting the logits for the last tokens (which should correspond to "A" or "B")
    choices = response.choices[0]
    logprobs = choices['logprobs']['top_logprobs'][0]
    print(logprobs)
    logits_for_A = logprobs.get('A', None)
    logits_for_B = logprobs.get('B', None)

    return logits_for_A,logits_for_B

def get_principles_from_folder():
    """
    Reads all the .txt files in the given folder and returns their content as principles.

    
    Returns:
    - dict: Dictionary where keys are filenames (without .txt) and values are the content of the files.
    """
    principles = {}
    for filename in os.listdir(principle_folder_path):
        if filename.endswith('.txt'):
            with open(os.path.join(principle_folder_path, filename), 'r') as file:
                principle_name = filename[:-4]  # Removing .txt extension
                principles[principle_name] = file.read().strip()
    return principles


def process_file_with_principles(input_filename, output_filename):
    """
    Reads the input file and writes the results to the output file.
    
    Args:
    - input_filename (str): The name of the input file.
    - output_filename (str): The name of the output file.
    """
    
    principles = get_principles_from_folder()

    with open(input_filename, 'r') as infile, open(output_filename, 'w') as outfile:
        
        for line in infile:
            line = line.strip()
            question, responseA, responseB = line.split('|')
            
            result_dict = {}
            result_dict["question"]=question
            result_dict["responseA"]=responseA
            result_dict["responseB"]=responseB
            for principle_name, principle in principles.items():
                logits_for_A, logits_for_B = evaluate_responses(question,responseA, responseB, principle)
                result_dict[principle_name] = (logits_for_A, logits_for_B)
            
            # Saving the results in a dictionary format
            outfile.write(f"{responseA},{responseB},{str(result_dict)}\n")


# 'input.txt' contains the list of (prompt, responseA, responseB) and we want the results in 'output.txt'
process_file_with_principles('input_test.txt', 'output_test.txt')



