from LoRA_hotswapping_PM import PreferenceModelHotswapper
from MORL_scalarizer import MORLScalarizer
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from torch.optim import Adam
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    RobertaForSequenceClassification,
    RobertaTokenizer,
)

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, create_reference_model, set_seed
from trl.core import LengthSampler


tqdm.pandas()
MORL_objective = "max-min" # The name of the Multi-Objective RL scalarization function to use. 
# See the MORL_scalarization_funcs file for the available options

shared_layers = 0 # number of layers to share between trained model and target model. I.e. number of layers not to train to save memory


@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="gpt-2-medium")
    log_with: Optional[str] = field(default=None)
    learning_rate: Optional[float] = field(default=(1.47e-5) * 2)
    mini_batch_size: Optional[int] = field(default=4)
    batch_size: Optional[int] = field(default=16)
    gradient_accumulation_steps: Optional[int] = field(default=1)
    model_save_path: Optional[str] = field(default=f"/Trained Models/gpt-2-medium_{MORL_objective}")

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

config = PPOConfig(
    model_name=script_args.model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    ppo_epochs=100,
    mini_batch_size=script_args.mini_batch_size,
    batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
)



def build_dataloader(config, dataset):
    """
    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(sample):
        prompt = sample["prompt"]
        sample["input_ids"] = tokenizer.encode(prompt)
        return sample

    dataset = dataset.map(tokenize, batched=False)
    dataset.set_format(type="torch")


    return dataset

train_dataset = load_dataset('json', data_files='/Data/hh-rlhf-train-extracted.jsonl')

train_dataloader= build_dataloader(config, train_dataset)


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


# set seed before initializing value head for deterministic eval
set_seed(config.seed)

model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=torch.bfloat16)
model = AutoModelForCausalLMWithValueHead.from_pretrained(model)

# We create the reference model and can optionally share layers with our target model.
ref_model = create_reference_model(model, num_shared_layers=shared_layers)


optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)


tokenizer = AutoTokenizer.from_pretrained(config.model_name)
tokenizer.pad_token = tokenizer.eos_token

# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    dataset=train_dataset,
    data_collator=collator,
    optimizer=optimizer,
)

# We then build the reward pipeline, using the MORL scalarisation function chosen at the start

preference_models = PreferenceModelHotswapper('gpt2-medium', '/Preference Models')
scalarizer = MORLScalarizer(func=MORL_objective)


generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}
output_min_length = 20
output_max_length = 500
output_length_sampler = LengthSampler(output_min_length, output_max_length)

model_save_path = script_args.model_save_path

for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    query_tensors = batch["input_ids"]

    # Get response from the policy model
    response_tensors = []
    for query in query_tensors:
        gen_len = output_length_sampler()
        generation_kwargs["max_new_tokens"] = gen_len
        response = ppo_trainer.generate(query, **generation_kwargs)
        response_tensors.append(response.squeeze()[-gen_len:])
    batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

    # Compute preference scores using PreferenceModelHotswapper
    preference_scores = preference_models.compute_scores({
        "input_ids": torch.stack(response_tensors).to(ppo_trainer.accelerator.device)
    })

    # Scalarize the multi-objective scores
    scalarized_rewards = []
    for _, scores in preference_scores.items():
        rewards_dict = {k: v.item() for k, v in zip(preference_models.adapter_names, scores.squeeze().tolist())}
        scalarized_reward = scalarizer.scalarize(rewards_dict)
        scalarized_rewards.append(scalarized_reward)

    rewards = torch.tensor(scalarized_rewards).to(ppo_trainer.accelerator.device)

    # Run PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)

    # Save model every 100 epochs
    if epoch % 100 == 0:
        if ppo_trainer.accelerator.is_main_process:
            ppo_trainer.save_pretrained(model_save_path)