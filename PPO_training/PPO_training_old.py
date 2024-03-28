import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, HfArgumentParser, AutoModel
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
from datasets import load_dataset
from peft import PeftModel, TaskType, LoraConfig, PeftConfig
from accelerate import Accelerator
from tqdm import tqdm
from huggingface_hub import login
import time
import math

accelerator = Accelerator()
def main():
    with open("hf_api.txt", "r") as hf:
        token = hf.read()
        login(token=token)
    
    parser = HfArgumentParser(PPOConfig)
    # Add custom arguments
    parser.add_argument("--base_model", type=str, default="gpt2-medium")
    parser.add_argument("--PM_path", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--num_proc", type=int, default=4)
    parser.add_argument("--LoRA", type=str, default="False")
    parser.add_argument("--LoRA_r", type=int, default=None)
    parser.add_argument("--LoRA_alpha", type=int, default=None)
    parser.add_argument("--LoRA_dropout", type=float, default=None)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--output_min_length", type=int, default=8)
    parser.add_argument("--output_max_length", type=int, default=64)
    parser.add_argument("--save_interval", type=float, default=0.2)
    ppo_config,config = parser.parse_args_into_dataclasses()
    # Load the trained preference model
    if "LoRA" in config.PM_path:
        peft_config = PeftConfig.from_pretrained(config.PM_path)
        preference_model = AutoModelForSequenceClassification.from_pretrained(peft_config.base_model_name_or_path,num_labels=1)
        preference_model =accelerator.prepare(PeftModel(preference_model, peft_config))
    else:
        preference_model = accelerator.prepare(AutoModelForSequenceClassification.from_pretrained(config.PM_path))
    preference_model.eval()
    # Load the tokenizer for the preference model
    preference_tokenizer = accelerator.prepare(AutoTokenizer.from_pretrained(config.PM_path, use_fast=True))
    
    
    # Load the base model to be fine-tuned with RLAIF
    peft_config = None
    if config.LoRA=="True":
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=config.LoRA_r,
            lora_alpha=config.LoRA_alpha,
            lora_dropout=config.LoRA_dropout,
            )
        if "gemma" in config.base_model:
            peft_config.target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]
    base_model= AutoModelForCausalLMWithValueHead.from_pretrained(config.base_model, peft_config=peft_config).to(accelerator.device)
    base_tokenizer = AutoTokenizer.from_pretrained(config.base_model, use_fast=True,padding_side='left')
    
    
    if getattr(preference_tokenizer, "pad_token", None) is None:
        preference_tokenizer.pad_token = preference_tokenizer.eos_token
    if getattr(base_tokenizer, "pad_token", None) is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
        base_model.config.pad_token_id = base_tokenizer.eos_token_id

    def preprocess_func(example):
        ex = base_tokenizer(example["prompt"], truncation=True, max_length=config.max_length, padding="max_length", return_tensors="pt")
        return {"prompt": example["prompt"],
                "input_ids": ex["input_ids"],
                "attention_mask": ex["attention_mask"]}
    
    dataset = load_dataset('json', data_files=config.dataset_path)

    # preprocess the dataset
    dataset = dataset.map(
            preprocess_func,
            batched=False,
            num_proc=config.num_proc,
            )
    output_length_sampler = LengthSampler(
        config.output_min_length,
        config.output_max_length,
    )

    def reward_fn(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    prompt_text: list[str],
    response_text: list[str],
    ) -> list[torch.FloatTensor]:
        """Compute the reward for a given response to a prompt.

        Args:
            model (AutoModel): Huggingface model.
            tokenizer (AutoTokenizer): Huggingface tokenizer.
            prompt_text (list[str]): List of strings representing the prompt.
            response_text (list[str]): List of strings representing the response.
            device (str, optional): Device to run the model on. Defaults to 'cpu'.

        Returns:
            list[float]: A list of floats representing the reward.

        """
        if not prompt_text or not response_text:
            raise ValueError("Either prompt_text or response_text is empty.")
    
        with torch.no_grad():
            encoding = tokenizer(
                prompt_text,
                response_text,
                truncation=True,
                max_length=512,
                padding='max_length',
                return_tensors='pt',
            )
            encoding = {k: v.to(accelerator.device) for k, v in encoding.items()}
            logits = model(**encoding).logits
            scores = logits.flatten().tolist()

            return scores
        
    def collator(data):
        batched_data = {}
        # Handling tensor data
        for key in ['input_ids', 'attention_mask']:
            batched_data[key] = [torch.tensor(d[key]).squeeze() for d in data if key in d]
        # Handling non-tensor data
        for key in ['prompt']:
            batched_data[key] = [d[key] for d in data if key in d]
        return batched_data
    ppo_trainer = accelerator.prepare(PPOTrainer(
        model=base_model,
        tokenizer=base_tokenizer,
        dataset=dataset["train"],
        config=ppo_config,
        data_collator=collator,
    ))
    generation_kwargs = {
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    'pad_token_id': base_tokenizer.pad_token_id,
    'eos_token_id': base_tokenizer.eos_token_id,
    }

   
    current_batch = 0
    total_batches = len(ppo_trainer.dataloader)
    save_batch = math.floor(config.save_interval * total_batches)
    for batch in tqdm(ppo_trainer.dataloader):

        current_batch += 1
        prompt_tensors = batch['input_ids']
        response_tensors = ppo_trainer.generate(
            prompt_tensors,
            return_prompt=False,
            **generation_kwargs,
        )
        batch['response'] = base_tokenizer.batch_decode(
            response_tensors,
            skip_special_tokens=True,
        )
        # Compute reward score.
        scores = reward_fn(
            model=preference_model,
            tokenizer=preference_tokenizer,
            prompt_text=batch['prompt'],
            response_text=batch['response'],
        )
        ppo_trainer.accelerator.print(batch['prompt'],batch['response'])
        rewards = [torch.tensor(score) for score in scores]
        # Run the PPO step.
        stats = ppo_trainer.step(prompt_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)
        if current_batch > save_batch:
            ppo_trainer.save_pretrained(config.save_path+"/checkpoint_"+str(current_batch))
            save_batch += math.floor(config.save_interval * total_batches)
           

    ppo_trainer.save_pretrained(config.save_path+"/final")

if __name__ == "__main__":
    main()
