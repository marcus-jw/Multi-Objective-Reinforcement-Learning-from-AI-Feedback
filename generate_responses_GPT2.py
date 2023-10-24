import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from itertools import islice

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
model = GPT2LMHeadModel.from_pretrained("gpt2-medium").to(device)

# Set the pad token
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

dataset = load_dataset('json', data_files='../Data/hh-rlhf-train-extracted.jsonl')

BATCH_SIZE = 16  # You can adjust this value based on your GPU's memory

def generate_responses_batch(prompts, num_responses=2):
    input_ids = tokenizer(prompts, return_tensors='pt', add_special_tokens=True, padding='max_length', max_length=500, truncation=True).input_ids.to(device)
    attention_mask = input_ids.ne(tokenizer.pad_token_id).int().to(device)
    
    output = model.generate(input_ids=input_ids, 
                            attention_mask=attention_mask,
                            max_length=700, 
                            num_return_sequences=num_responses, 
                            no_repeat_ngram_size=2, 
                            temperature=0.9,
                            do_sample=True)
    
    # Reshape the output to group the responses by prompt
    output = output.reshape(BATCH_SIZE, num_responses, -1)
    responses = [[tokenizer.decode(o, skip_special_tokens=True) for o in batch] for batch in output]
    
    return responses

dataloader = DataLoader(dataset["train"], batch_size=BATCH_SIZE, shuffle=False)

with open('responses.txt', 'w', encoding='utf-8' ) as file:
    for batch in tqdm(islice(dataloader, 1000), desc="Processing dataset"): #TODO remove 1000
        prompts = batch["prompt"]
        batch_responses = generate_responses_batch(prompts)
        
        for prompt, responses in zip(prompts, batch_responses):
            file.write(f"Prompt: {prompt}\n")
            for idx, response in enumerate(responses, 1):
                file.write(f"Response {idx}: {response}\n")
            file.write("\n")