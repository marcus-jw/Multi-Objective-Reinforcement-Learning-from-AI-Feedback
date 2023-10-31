import openai
import os
import random
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
    
    prompt = f"You will be given a conversation between a human and an AI assistant along "\
            "with a principle and two responses. Your task is to choose the response which "\
            "best follows the principle. \n"\
            "Conversation: {question} \n Given the principle '{principle}', "\
            "which of the following responses is better?\n" \
             f"A. {responseA}\n" \
             f"B. {responseB}\n" \
             f"Respond only with A or B.\n\n"

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


def process_file_with_principles(input_filename, output_filename):
    """
    - Reads each line from the input file and extracts a question and two responses.
    - For each question-response pair, evaluates the responses using for each principle with a randomly sampled wording
    - Writes the evaluations to an output file in dictionary format.
    
    Args:
    - input_filename (str): The name of the input file containing questions and two responses separated by '|' for each line.
    - output_filename (str): The name of the output file where evaluation results will be saved.

   
    """
    
    # Fetch the dictionary of principles where each key is a principle name
    # and each value is a list of rewordings of that principle.
    principles = get_principles_from_folder()

    # Open the input file for reading and the output file for writing.
    with open(input_filename, 'r') as infile, open(output_filename, 'w') as outfile:
        
        # Loop through each line in the input file.
        for line in infile:
            # Remove leading and trailing whitespaces from the line.
            line = line.strip()
            
            # Split the line by the delimiter '|' to extract the question and two responses.
            question, responseA, responseB = line.split('|')
            
            # Initialize a dictionary to hold the question, responses, and evaluation results.
            result_dict = {}
            result_dict["question"] = question
            result_dict["responseA"] = responseA
            result_dict["responseB"] = responseB
            
            # Loop through each principle and its rewordings.
            for principle_name, rewordings in principles.items():
                # Randomly select one rewording for the principle from the list.
                sampled_principle = random.choice(rewordings)
                
                # Evaluate the responses based on the randomly selected rewording.
                logits_for_A, logits_for_B = evaluate_responses(question, responseA, responseB, sampled_principle)
                
                # Save the evaluation results for the principle in the results dictionary.
                result_dict[principle_name] = (logits_for_A, logits_for_B)
            
            # Write the results dictionary to the output file.
            outfile.write(f"{result_dict}\n")

# Sample usage: 'input_test.txt' contains questions and response pairs,
# and 'output_test.txt' will store the evaluation results.
process_file_with_principles('input_test.txt', 'output_test.txt')


# 'input.txt' contains the list of (prompt, responseA, responseB) and we want the results in 'output.txt'
process_file_with_principles('input_test.txt', 'output_test.txt')



