import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
model = GPT2LMHeadModel.from_pretrained("gpt2-medium").to(device)

# Set the pad token
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

with open('prompts.txt', 'r') as file:
    prompts = file.readlines()
    
    
    
def generate_responses(prompt, num_responses=2):
    input_ids = tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=True, padding='max_length', max_length=100, truncation=True).to(device)
    attention_mask = input_ids.ne(tokenizer.pad_token_id).int().to(device)
    responses = []
    
    output = model.generate(input_ids=input_ids, 
                            attention_mask=attention_mask,
                            max_length=200, 
                            num_return_sequences=num_responses, 
                            no_repeat_ngram_size=2, 
                            temperature=0.9,
                            do_sample=True)
    
    responses = [tokenizer.decode(o, skip_special_tokens=True) for o in output]
    
    return responses


all_responses = {}
for prompt in prompts:
    all_responses[prompt] = generate_responses(prompt)


with open('responses.txt', 'w') as file:
    for prompt, responses in all_responses.items():
        file.write(f"Prompt: {prompt}\n")
        for idx, response in enumerate(responses, 1):
            file.write(f"Response {idx}: {response}\n")
        file.write("\n")