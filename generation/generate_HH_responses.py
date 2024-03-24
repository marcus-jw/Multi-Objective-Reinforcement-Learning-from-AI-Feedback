import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from accelerate import Accelerator
#import time
from huggingface_hub import login

def main():
    parser = argparse.ArgumentParser()

    # Add custom arguments
    parser.add_argument("--model_name", type=str, default="gpt2-medium")
    parser.add_argument("--num_proc", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--start_at", type=int, default=0)
    config = parser.parse_args()
    accelerator = Accelerator()  

    # Log in to the Hugging Face API for gemma and llama
    with open("hf_api.txt", "r") as hf:
        token = hf.read()
        login(token=token)
    print(config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True,padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(config.model_name)
    if "gpt" in config.model_name:
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id
        tokenizer.pad_token_id = tokenizer.eos_token_id
    dataset = load_dataset('json', data_files=config.dataset_path)
    #use only the subset of the dataset after the start_at index
    dataset["train"]=dataset["train"].select(range(config.start_at, len(dataset["train"])))
    dataset = dataset.with_format("torch")
    
    dataloader = DataLoader(dataset["train"], batch_size=config.batch_size, shuffle=False, num_workers=4)
    model, tokenizer, dataloader = accelerator.prepare(model, tokenizer, dataloader) 
    model.eval()
    class Stop(StoppingCriteria):
        def __init__(self, prompt):
            self.target_sequences = ["Human:","\n\n Human:", " Human:","\n\nHuman:","\nHuman:"]
            self.prompt=prompt

        def __call__(self, input_ids, scores, **kwargs):
            # Get the generated text as a string
            generated_text = tokenizer.decode(input_ids[0])
            
            generated_text = generated_text.replace("".join(self.prompt),'')
            # Check if the target sequence appears in the generated text
            for seq in self.target_sequences:
                if seq in generated_text:
                    return True  # Stop generation
            
            return False  # Continue generation
        def __len__(self):
            return 1

        def __iter__(self):
            yield self

    with open(config.output_path, "w",encoding='utf-8') as file:
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Processing dataset"):
                encoded = tokenizer(batch["prompt"], return_tensors='pt', add_special_tokens=True, padding='max_length', max_length=500, truncation=True)
                input_ids = encoded['input_ids'].to(accelerator.device)
                attention_mask = encoded['attention_mask'].to(accelerator.device)
                #print("before: ",time.time())
                outputs=[]
                max_length = 0
                for _ in range(2):
                    output = model.generate(input_ids=input_ids, 
                                    attention_mask=attention_mask,
                                    max_new_tokens=200,
                                    min_new_tokens=2, 
                                    num_return_sequences=1, 
                                    #no_repeat_ngram_size=2, 
                                    temperature=0.7,
                                    do_sample=True,
                                    stopping_criteria=Stop(batch["prompt"]))
                    outputs.append(output)
                    max_length = max(max_length, output.size(-1))
                padded_outputs = []
                for output in outputs:
                    padding = torch.ones((output.size(0), max_length - output.size(-1)), dtype=output.dtype, device=output.device) * model.config.pad_token_id
                    padded_output = torch.cat([output, padding], dim=-1)
                    padded_outputs.append(padded_output)

                outputs = torch.stack(padded_outputs, dim=1)
                
                responses = [[tokenizer.decode(o, skip_special_tokens=True) for o in batch] for batch in outputs]
                for response,prompt in zip(responses,batch["prompt"]):
                    reponseA = response[0][len(prompt):-1]
                    reponseB = response[1][len(prompt):-1]
                    output_dict = {"prompt": prompt, "responseA": reponseA, "responseB": reponseB}
                    file.writelines(json.dumps(output_dict, ensure_ascii=False) + '\n')
        return responses


if __name__ == "__main__":
    main()
