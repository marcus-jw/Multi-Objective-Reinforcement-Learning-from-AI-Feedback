from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
import os
from peft import LoraConfig, TaskType, get_peft_model
import torch

class PreferenceModelHotswapper:
    """
   A class that handles loading and swapping of adapters so that many preference models can be used without increasing memory consumption much.
   
   Methods:
   --------
   compute_scores(inputs):
       Computes scores for the given inputs using each of the loaded adapters.
   
   Parameters:
   -----------
   model_name : str
       The name of the base model to be used.
   adapter_folder : str
       The path to the folder containing adapter models.
   """
    def __init__(self, model_path,principles,peft_config):
        model_name = model_path.split("/")[-1]
        self.device = torch.cuda.device_count() - 1
        self.tokenizer = AutoTokenizer.from_pretrained(model_path + "_" + principles[0] + "/final", use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token 
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1, torch_dtype=torch.float)
        self.model.config.pad_token_id = self.tokenizer.eos_token_id
        self.model = get_peft_model(self.model, peft_config).to(self.device)

        self.adapter_names = []
        
        for principle in principles:
            adapter_path = model_path  +  "_" + principle + "/final"
            if not os.path.exists(adapter_path):
                print(f"Adapter path {adapter_path} does not exist.")
            else:
                try:
                    self.model.load_adapter(adapter_path, adapter_name = principle,is_trainable = False,torch_device = self.device)
                    print(f"Loaded adapter {principle} successfully.")
                    self.adapter_names.append(principle)
                except Exception as e:
                    print(f"Failed to load adapter {principle}: {str(e)}")

            
    def compute_scores(self, input_ids, attention_mask):
        scores = {}
        for adapter_name in self.adapter_names:
            # Set the current adapter as active
            self.model.set_adapter(adapter_name)
            self.model.eval()

            # Perform the model inference in a batch
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

            # Extract the logits as scores
            adapter_scores = outputs.logits.squeeze()

            scores[adapter_name]=(adapter_scores.cpu().numpy()) 
        return scores