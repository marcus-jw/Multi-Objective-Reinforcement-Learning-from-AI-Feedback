from transformers import GPT2Tokenizer, GPT2Model
import os

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
        self.model = GPT2Model.from_pretrained(base_model_name)
        self.adapter_names = []
        
        # Load all adapters from the given folder
        for adapter_name in os.listdir(adapter_folder):
            adapter_path = os.path.join(adapter_folder, adapter_name)
            self.model.load_adapter(adapter_path)
            self.adapter_names.append(adapter_name)
            
    def compute_scores(self, inputs):
        scores = {}
        
        # Compute score for each adapter
        for adapter_name in self.adapter_names:
            # Set the current adapter as active
            self.model.set_active_adapters(adapter_name) #TODO measure performance of this and see if there is a better way of doing it
            
            # Get preference model's score
            score = self.model(**inputs)
            scores[adapter_name] = score
            
        return scores