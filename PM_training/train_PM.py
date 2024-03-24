from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import torch
torch.backends.cuda.matmul.allow_tf32 = True
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser,TrainingArguments
from accelerate import Accelerator
from trl import RewardConfig, RewardTrainer
from peft import LoraConfig, TaskType
import os
from huggingface_hub import login
with open("hf_api.txt", "r") as hf:
        token = hf.read()
        login(token=token)
accelerator = Accelerator()              
if __name__ == "__main__":
    parser = HfArgumentParser(RewardConfig)
    
    # Add custom arguments
    parser.add_argument("--model_name", type=str, default="gpt2-medium")
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--num_proc", type=int, default=4)
    parser.add_argument("--principle", type=str, default=None)
    parser.add_argument("--LoRA", type=str, default="False")
    parser.add_argument("--LoRA_r", type=int, default=None)
    parser.add_argument("--LoRA_alpha", type=int, default=None)
    parser.add_argument("--LoRA_dropout", type=float, default=None)
    # Parse the dictionary into RewardConfig
    reward_config,config = parser.parse_args_into_dataclasses()
    reward_config.gradient_checkpointing_kwargs={"use_reentrant":False}
    if config.principle:
        train_dataset = load_dataset('json', data_files=f'data/datasets/{config.dataset_dir}_hh_train_feedback.jsonl')
        test_dataset = load_dataset('json', data_files=f'data/datasets/{config.dataset_dir}_hh_test_feedback.jsonl')
    else:
        train_dataset = load_dataset('json', data_files=f'{config.dataset_dir}_hh_train_feedback.jsonl')
        test_dataset = load_dataset('json', data_files=f'{config.dataset_dir}_hh_test_feedback.jsonl') 
    # If LoRA is true, create a LoraConfig object

    if config.LoRA=="True" and "gemma" in config.model_name:
        peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=config.LoRA_r,
        lora_alpha=config.LoRA_alpha,
        lora_dropout=config.LoRA_dropout,
        use_rslora=True,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]
        )
    elif config.LoRA=="True":
        peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=config.LoRA_r,
        lora_alpha=config.LoRA_alpha,
        lora_dropout=config.LoRA_dropout,
        use_rslora=True,
        )
    else:
        peft_config = None

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name, num_labels=1)
    model.config.use_cache = False
    if "gpt2" in config.model_name:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.eos_token_id


    def create_preprocess_func(tokenizer,max_length):
        def preprocess_func(examples):
            inputs_chosen, inputs_rejected = [], []
            attention_masks_chosen, attention_masks_rejected = [], []

            for i in range(len(examples['prompt'])):
                question, answerA, answerB = examples['prompt'][i], examples['responseA'][i], examples['responseB'][i]
                logitsA, logitsB = examples['logits_A'][i], examples['logits_B'][i]

                if logitsA > logitsB:
                    tokenized_chosen = tokenizer(question + answerA, truncation=True, padding='max_length', max_length=max_length)
                    tokenized_rejected = tokenizer(question + answerB, truncation=True, padding='max_length', max_length=max_length)
                else:
                    tokenized_chosen = tokenizer(question + answerB, truncation=True, padding='max_length', max_length=max_length)
                    tokenized_rejected = tokenizer(question + answerA, truncation=True, padding='max_length', max_length=max_length)

                inputs_chosen.append(tokenized_chosen['input_ids'])
                attention_masks_chosen.append(tokenized_chosen['attention_mask'])
                inputs_rejected.append(tokenized_rejected['input_ids'])
                attention_masks_rejected.append(tokenized_rejected['attention_mask'])

            return {
                'input_ids_chosen': inputs_chosen,
                'attention_mask_chosen': attention_masks_chosen,
                'input_ids_rejected': inputs_rejected,
                'attention_mask_rejected': attention_masks_rejected,
            }
        return preprocess_func
    
    # preprocess the dataset
    train_dataset = train_dataset.map(
            create_preprocess_func(tokenizer,reward_config.max_length),
            batched=True,
            num_proc=config.num_proc,
        )
    test_dataset = test_dataset.map(
        create_preprocess_func(tokenizer,reward_config.max_length),
        batched=True,
        num_proc=config.num_proc,
    )

    trainer = accelerator.prepare(RewardTrainer(
            model=model,
            tokenizer=tokenizer,
            args=reward_config,
            train_dataset=train_dataset['train'],
            eval_dataset=test_dataset,
            peft_config=peft_config
        ))
    trainer.train()
    trainer.save_model(reward_config.output_dir + "/final" )



