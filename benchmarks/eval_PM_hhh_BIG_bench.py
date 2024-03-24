import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset
from tqdm import tqdm
from peft import PeftModel, PeftConfig    

model_path = "data/PM_LoRAs/gemma-2b_CAI/final"
test_file = "benchmarks/hhh_alignment.json"
lora="google/gemma-2b"

def load_preference_model(model_path):
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=1,local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    return model, tokenizer
def load_preference_model_LoRA(model_path):
    model = AutoModelForSequenceClassification.from_pretrained(lora, num_labels=1)
    tokenizer = AutoTokenizer.from_pretrained(lora)
    config = PeftConfig.from_pretrained(model_path)
    model = PeftModel(model, config)
    return model, tokenizer
def score_response(model, tokenizer, input_text, response_text):
    input_ids = tokenizer.encode(input_text + response_text, return_tensors="pt", truncation=True, padding=True)
    output = model(input_ids)[0]
    return output.item()

def evaluate_preference_model(model_path, test_file):
    if lora:
        model, tokenizer = load_preference_model_LoRA(model_path)
    else:
        model, tokenizer = load_preference_model(model_path)

    with open(test_file, 'r') as f:
        test_data = json.load(f)
    #print(test_data)

    correct_predictions = 0
    total_examples = len(test_data)
    print(total_examples)
    for example in tqdm(test_data):
        #print(example)
        input_text = example['input']
        scores = example['target_scores']

        responses = list(scores.keys())
        scores_list = list(scores.values())
        if len(responses) != 2:
            raise ValueError("Each example must have exactly 2 response options.")

        score0 = score_response(model, tokenizer, input_text, responses[0])
        score1 = score_response(model, tokenizer, input_text, responses[1])

        if (score0 > score1 and scores_list[0] > scores_list[1]) or (score1 > score0 and scores_list[1] > scores_list[0]):
            correct_predictions += 1

    accuracy = correct_predictions / total_examples
    print(f"Accuracy: {accuracy:.4f}")

evaluate_preference_model(model_path, test_file)