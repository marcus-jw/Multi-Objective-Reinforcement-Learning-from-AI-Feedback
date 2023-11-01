from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import evaluate
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.utils import PaddingStrategy

principle="violence" #this determines which principle to use for the preference model. 
#This needs to have the same name as the corresponding feedback dataset in Data/ and the principle file in Principles/

num_proc = 4  # CPU processors

# Define and parse arguments.
@dataclass
class ScriptArguments:
    local_rank: Optional[int] = field(default=-1)
    resume_from_checkpoint: Optional[bool] = field(default=False)
    deepspeed: Optional[str] = field(default=None)
    per_device_train_batch_size: Optional[int] = field(default=2)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=1)
    learning_rate: Optional[float] = field(default=2e-5)
    weight_decay: Optional[float] = field(default=0.001)
    model_name: Optional[str] = field(default="gpt2-medium")
    tokenizer_name: Optional[str] = field(default=None)
    bf16: Optional[bool] = field(default=False)
    num_train_epochs: Optional[int] = field(default=1)
    #train_subset: Optional[int] = field(default=100000)
    #eval_subset: Optional[int] = field(default=50000)
    gradient_checkpointing: Optional[bool] = field(default=False)
    optim: Optional[str] = field(default="adamw_hf")
    lr_scheduler_type: Optional[str] = field(default="linear")
    max_length: Optional[int] = field(default=512)
    eval_first_step: Optional[bool] = field(default=False)

@dataclass
class RewardDataCollator:
    # Secial data collator for batching the data in our format.
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features_A = []
        features_B = []
        logits_A = []
        logits_B = []  
        for feature in features:
            features_A.append(
                {
                    "input_ids": feature["input_ids_A"],
                    "attention_mask": feature["attention_mask_A"],
                }
            )
            features_B.append(
                {
                    "input_ids": feature["input_ids_B"],
                    "attention_mask": feature["attention_mask_B"],
                }
            )
            logits_A.append(feature["logits_A"])
            logits_B.append(feature["logits_B"])

        batch_A = self.tokenizer.pad(
            features_A,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_B = self.tokenizer.pad(
            features_B,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        batch = {
            "input_ids_A": batch_A["input_ids"],
            "attention_mask_A": batch_A["attention_mask"],
            "input_ids_B": batch_B["input_ids"],
            "attention_mask_B": batch_B["attention_mask"],
            "logits_A": torch.tensor(logits_A),
            "logits_B": torch.tensor(logits_B),
            "return_loss": True,
        }
        return batch
    
def preprocess_data(data, tokenizer):
        formated = {
            "input_ids_A": [],
            "attention_mask_A": [],
            "input_ids_B": [],
            "attention_mask_B": [],
            "logits_A": [],  # logits for AnswerA
            "logits_B": []   # logits for AnswerB
        }
        for prompt, answerA, answerB, logitsA, logitsB in zip(data["Prompt"], data["ResponseA"], data["ResponseB"], data[principle][0], data[principle][1]):
            tokenized_A = tokenizer("Prompt: " + prompt + "\n\nResponse: " + answerA, truncation=True)
            tokenized_B = tokenizer("Prompt: " + prompt + "\n\nResponse: " + answerB, truncation=True)

            formated["input_ids_A"].append(tokenized_A["input_ids"])
            formated["attention_mask_A"].append(tokenized_A["attention_mask"])
            formated["logits_A"].append(logitsA)
            formated["input_ids_B"].append(tokenized_B["input_ids"])
            formated["attention_mask_B"].append(tokenized_B["attention_mask"])
            formated["logits_B"].append(logitsB)
        return formated
def load_and_preprocess_datasets(script_args, tokenizer):
    train_dataset = load_dataset('json', data_files=f'Data/testing-s-rated.jsonl')['train']
    original_columns = train_dataset.column_names
    train_dataset = train_dataset.map(
        lambda data: preprocess_data(data, tokenizer),
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns
    )
    return train_dataset

def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        # Determine which response (A or B) has a higher predicted reward.
        # If reward for A > reward for B, predict A (0), otherwise predict B (1).
        predicted_labels = (predictions[:, 1] > predictions[:, 0]).astype(int)

        # Determine the true preferred response based on provided logits.
        # If logits_A > logits_B, true label is A (0), otherwise true label is B (1).
        true_labels = (labels[:, 1] > labels[:, 0]).astype(int)

        return accuracy.compute(predictions=predicted_labels, references=true_labels)

class RewardTrainer(Trainer):
        # Compute the pairwise logloss:
        def compute_loss(self, model, inputs, return_outputs=False):
            logits_A = inputs["logits_A"]
            logits_B = inputs["logits_B"]

            # Compute soft labels using the sigmoid function
            soft_labels = torch.sigmoid(logits_B - logits_A)

            rewards_A = model(input_ids=inputs["input_ids_A"], attention_mask=inputs["attention_mask_A"])[0].squeeze()
            rewards_B = model(input_ids=inputs["input_ids_B"], attention_mask=inputs["attention_mask_B"])[0].squeeze()

            # Compute the model's soft predictions using the sigmoid function
            model_soft_predictions = torch.sigmoid(rewards_B - rewards_A)

            # Compute the binary cross-entropy loss between the model's soft predictions and the soft labels
            loss_fct = nn.BCELoss()
            loss = loss_fct(model_soft_predictions, soft_labels)

            if return_outputs:
                return loss, {"rewards_A": rewards_A, "rewards_B": rewards_B}
            return loss


def configure_training(script_args):
    model_name_split = script_args.model_name.split("/")[-1]
    output_name = f"Models/PM_{script_args.model_name}_{principle}_LoRA"
    
    valid_args = set(TrainingArguments.__dataclass_fields__.keys())
    filtered_args = {k: v for k, v in vars(script_args).items() if k in valid_args}

    training_args = TrainingArguments(
        output_dir=output_name,
        **filtered_args,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        remove_unused_columns=False,
        label_names=[]
    )
    return training_args

def initialize_model_and_tokenizer(script_args):
    # Load the value-head model and tokenizer
    tokenizer_name = script_args.tokenizer_name if script_args.tokenizer_name is not None else script_args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_auth_token=True)
    
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(
        script_args.model_name, num_labels=1, torch_dtype=torch.float
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.use_cache = not script_args.gradient_checkpointing
    return model, tokenizer

def train_model(model, tokenizer, training_args,script_args):
    # Initialize the Trainer and train the model
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=load_and_preprocess_datasets(script_args,tokenizer),
        compute_metrics=compute_metrics,
        data_collator=RewardDataCollator(tokenizer=tokenizer, max_length=script_args.max_length)
    )
    
        
    trainer.train(script_args.resume_from_checkpoint)

def main():
    # Argument Parsing
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # Training Configuration
    training_args = configure_training(script_args)

    # Model Initialization
    model, tokenizer = initialize_model_and_tokenizer(script_args)

    # Dataset Loading
    load_and_preprocess_datasets(script_args, tokenizer)

    # Model Training
    train_model(model, tokenizer, training_args,script_args)
    print("last checkpoint")
    model.save_pretrained(training_args.output_dir + "_last_checkpoint")

if __name__ == '__main__':
    main()
