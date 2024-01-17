import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed 
import time
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
    model = GPT2LMHeadModel.from_pretrained("gpt2-large").to(device)
    for param in model.parameters(): #freeze the model
        param.requires_grad = False
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    tokenizer.padding_side = 'left'
    dataset = load_dataset('json', data_files='Data/hh-rlhf-train-extracted.jsonl')
    BATCH_SIZE = 10

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

    def write_responses_to_file(batch_responses, prompts):
        output_list = []
        for prompt, responses in zip(prompts, batch_responses):
            reponseA = responses[0][len(prompt):-1]
            reponseB = responses[1][len(prompt):-1]
            output_dict = {"Prompt": prompt, "ResponseA": reponseA, "ResponseB": reponseB}
            output_list.append(json.dumps(output_dict, ensure_ascii=False) + '\n')
        return output_list

    dataloader = DataLoader(dataset["train"], batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    future_to_batch = {}
    with ThreadPoolExecutor(max_workers=2) as executor, open('hh_train_responses.jsonl', 'w', encoding='utf-8') as file:
            for batch in tqdm(dataloader, desc="Processing dataset"):
                prompts = batch["prompt"]
                # Submit the generation task
                future = executor.submit(generate_responses_batch, prompts)
                future_to_batch[future] = batch

                # Check if any generation task is completed and write to file
                for future in as_completed(future_to_batch):  
                    batch_responses = future.result()
                    output_lines = write_responses_to_file(batch_responses, future_to_batch[future]["prompt"])
                    file.writelines(output_lines)
                    del future_to_batch[future]

if __name__ == "__main__":
    main()
