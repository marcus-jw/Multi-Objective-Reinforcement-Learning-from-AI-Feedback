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
    def __init__(self, model_name, adapter_folder,principles,lora_config):
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token 
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1, torch_dtype=torch.float)
        self.model.config.pad_token_id = self.tokenizer.eos_token_id
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=True,
            r=lora_config.r,
            lora_alpha=lora_config.alpha,
            lora_dropout=lora_config.dropout,
        )
        self.model = get_peft_model(self.model, peft_config).to(self.device)

        self.adapter_names = []
        
        for principle in principles:
            adapter_path = adapter_folder + model_name +  "_" + principle
            self.model.load_adapter(adapter_path, principle)
            self.adapter_names.append(principle)
            
    def compute_scores(self, input_ids, attention_mask):
        scores = []
        for adapter_name in self.adapter_names:
            # Set the current adapter as active
            self.model.set_adapter(adapter_name)

            # Perform the model inference in a batch
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

            # Extract the logits as scores
            adapter_scores = outputs.logits.squeeze()

            scores.append(adapter_scores.cpu().numpy()) 

        return scores