import json
import math
import os
import sys
from itertools import islice
from typing import Optional
import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import login
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, HfArgumentParser
import argparse
from peft import PeftConfig, PeftModel
from transformers.modeling_utils import load_sharded_checkpoint
from safetensors.torch import save_model
from trlx.trainer import register_trainer
from trlx.trainer.accelerate_ppo_trainer import AcceleratePPOTrainer
from MORL_scalarizer import MORLScalarizer
from LoRA_hotswapping_PM import PreferenceModelHotswapper
import trlx
from trlx.data.default_configs import (
    ModelConfig,
    OptimizerConfig,
    PPOConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)
@register_trainer("CustomAcceleratePPOTrainer")
class CustomAcceleratePPOTrainer(AcceleratePPOTrainer):
    # Override the save methods to save the model in a different way
    def save(self, directory: Optional[str] = None, **kwargs):
        pass
    def save_pretrained(self, directory: Optional[str] = None, **kwargs):
        super().save_pretrained(self.config.train.checkpoint_dir, **kwargs)
       
with open("hf_api.txt", "r") as hf:
        token = hf.read()
        login(token=token)
        
config = TRLConfig.load_yaml("PPO_training/default_PPO_config.yaml")
parser = argparse.ArgumentParser()
parser.add_argument('--PM_path', type=str, default=None)
parser.add_argument('--training_set_path', type=str, default=None)
parser.add_argument('--test_set_path', type=str, default=None)
parser.add_argument('--reward_batch_size', type=int, default=48)
parser.add_argument('--MORL', type=str, default=None)
parser.add_argument('--LoRA', type=str, default=None)
parser.add_argument('--LoRA_r', type=int, default=8)
parser.add_argument('--LoRA_alpha', type=int, default=16)
parser.add_argument('--LoRA_dropout', type=float, default=0.1)
parser.add_argument('--PMs', type=str, default=None)
parser.add_argument('--scalarizer', type=str, default="linear")
parser.add_argument('--weight_file', type=str, default=None)


args, trl_args = parser.parse_known_args()
args.PMs = args.PMs.split(",")
arg_dict={}
for i in range(0, len(trl_args), 2):
    arg_dict[trl_args[i].lstrip("-")] = trl_args[i+1]
for key, value in arg_dict.items():
    try:
        value = json.loads(value)
    except json.JSONDecodeError:
        pass
    config_section, config_key = key.split('.', 1)
    setattr(getattr(config, config_section), config_key, value)


def create_reward_fn(): 
    preference_tokenizer = AutoTokenizer.from_pretrained(args.PM_path, use_fast=True)
    if getattr(preference_tokenizer, "pad_token", None) is None:
        preference_tokenizer.pad_token = preference_tokenizer.eos_token
    preference_tokenizer.truncation_side = "left"
    if args.MORL:
        if "LoRA" in args.PM_path:
            peft_config = PeftConfig.from_pretrained(args.PM_path + "_" + args.PMs[0])
            multi_PM = PreferenceModelHotswapper(args.PM_path, args.adapter_folder,args.PMs, peft_config)
        scalarizer = MORLScalarizer(args.scalarizer, args.weight_file)
    else:
        if "LoRA" in args.PM_path:
            peft_config = PeftConfig.from_pretrained(args.PM_path)
            preference_model = AutoModelForSequenceClassification.from_pretrained(peft_config.base_model_name_or_path,num_labels=1)
            preference_model = PeftModel(preference_model, peft_config)
        else:
            preference_model = AutoModelForSequenceClassification.from_pretrained(args.PM_path)

    preference_model.eval()
    preference_model.requires_grad_(False)
    PM_device = torch.cuda.device_count() - 1
    preference_model = preference_model.half().to(PM_device)
    delta_reward = False

    def get_reward(samples):
        input = preference_tokenizer(
            samples,
            padding=True,
            truncation=True,
            max_length=preference_tokenizer.max_len_single_sentence,
            return_tensors="pt",
        ).to(PM_device)

        mbs = args.reward_batch_size
        out = []
        for i in range(math.ceil(len(samples) / mbs)):
            batch_ixs = slice(i * mbs, (i + 1) * mbs)
            input_ids = input.input_ids[batch_ixs]
            attention_mask = input.attention_mask[batch_ixs]
            if args.MORL:
                rewards = multi_PM.compute_scores(input_ids, attention_mask)
                rewards = scalarizer.scalarize(rewards)
            else:
                rewards = preference_model(input_ids=input_ids, attention_mask=attention_mask).logits
            out.extend(rewards)
        return torch.hstack(out)

    def reward_fn(samples, prompts, **kwargs):
        samples = [s + preference_tokenizer.eos_token for s in samples]
        rewards = get_reward(samples)

        if not delta_reward:
            return rewards

        original_samples = [p + o + preference_tokenizer.eos_token for p, o in zip(prompts, original_output)]
        original_rewards = get_reward(original_samples)
        return rewards - original_rewards

    return reward_fn


def main():

    training_dataset = load_dataset("json", data_files=args.training_set_path)
    test_dataset = load_dataset("json", data_files=args.test_set_path)
    prompts = [{"prompt": x["prompt"]} for x in training_dataset["train"]]
    eval_prompts = [{"prompt": x["prompt"]} for x in islice(test_dataset["train"], 300)]
    reward_fn = create_reward_fn()
    config.train.trainer= "CustomAcceleratePPOTrainer"
    trlx.train(
        prompts=prompts,
        eval_prompts=eval_prompts,
        reward_fn=reward_fn,
        config=config,
        stop_sequences=["Human:", "human:", "Assistant:", "assistant:"],

    )


if __name__ == "__main__":
    main()