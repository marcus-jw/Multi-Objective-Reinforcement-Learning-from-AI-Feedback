from transformers import GPT2Tokenizer, GPT2Model,GPT2LMHeadModel 
import os
from peft import LoraConfig, TaskType, get_peft_model
import torch

class PreferenceModelHotswapper:
    """
   A class that handles loading and swapping of adapters many preference models can effecively be used without increasing memory consumption much.
   
   Methods:
   --------
   compute_scores(inputs):
       Computes scores for the given inputs using each of the loaded adapters.
   
   Parameters:
   -----------
   base_model_name : str
       The name of the base model to be used.
   adapter_folder : str
       The path to the folder containing adapter models.
   """
    def __init__(self, base_model_name, adapter_folder):
        self.tokenizer = GPT2Tokenizer.from_pretrained(base_model_name)
        self.model = GPT2LMHeadModel.from_pretrained(base_model_name)
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )
    
        self.model = get_peft_model(self.model, peft_config)


        self.adapter_names = []
        
        # Load all adapters from the given folder
        for adapter_name in os.listdir(adapter_folder):

            adapter_path = os.path.join(adapter_folder, adapter_name)
            # Check if the adapter name already exists and skip or handle appropriately
            self.model.load_adapter(adapter_path,adapter_name)
            self.adapter_names.append(adapter_name)
            
    def compute_scores(self, inputs):
        scores = {}
        tok=self.tokenizer(inputs)
        # Compute score for each adapter
        for adapter_name in self.adapter_names:
            # Set the current adapter as active
            self.model.set_adapter(adapter_name) #TODO measure performance of this and see if there is a better way of doing it
            # Get preference model's score
            
            score = self.model(input_ids=tok['input_ids'],attention_mask=tok['attention_mask'])
            scores[adapter_name] = score
            
        return scores