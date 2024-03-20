from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser,TrainingArguments,pipeline
from accelerate import Accelerator
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from peft import LoraConfig, TaskType


dataset = load_dataset('json', data_files='Data/hh-rlhf-train-extracted.jsonl')

model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
tokenizer = AutoTokenizer.from_pretrained(config.model_name)

if "gpt2" in config.model_name:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.eos_token_id

def tokenize(sample):
    sample["input_ids"] = tokenizer.encode(sample["prompt"])
    return sample
dataset = dataset.map(tokenize, batched=False)
accelerator = Accelerator()

ppo_trainer = PPOTrainer(
    model=model,
    config=config,
    dataset=dataset,
    tokenizer=tokenizer,
)