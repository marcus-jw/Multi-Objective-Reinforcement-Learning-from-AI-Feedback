import torch
# This file contains different Multi-Objective RL scalarization functions which will be tested
class MORLScalarizer:
    """
    Applies a scalarization method to a multi-objective reward set. 
    Before scalarization, rewards are transformed based on pre-defined scaling and offset values.
    """
    
    def __init__(self, func):
        """
        Initialize the scalarizer with a specific scalarization function.

        Parameters:
            func (str): The scalarization function to apply. Valid values are "max_min", "soft_max_min", and "equal_weight".
        """
        func_dict = {
            "max_min": self.max_min,
            "soft_max_min": self.soft_max_min,
            "equal_weight": self.equal_weight
        }
        self.func = func_dict[func]
        
        
    
    scaling_dict = {
            "conciseness":    1,
            "ethical":        1,
            "factual":        1,
            "honesty":        1,
            "legal":          1,
            "racism":         1,
            "relevance":      1,
            "sexism":         1,
            "sycophancy":     1,
            "toxicity":       1,
            "truthful":       1,
            "usefulness":     1,
            "violence":       1,
            "x_risk":         1
    }
    offset_dict = {
            "conciseness":    0,
            "ethical":        0,
            "factual":        0,
            "honesty":        0,
            "legal":          0,
            "racism":         0,
            "relevance":      0,
            "sexism":         0,
            "sycophancy":     0,
            "toxicity":       0,
            "truthful":       0,
            "usefulness":     0,
            "violence":       0,
            "x_risk":         0
        }
    
    
    def apply_scaling_and_offset(self,rewards):
        """
        Applies scaling and offset to the given rewards based on the scaling_dict and offset_dict.
        
        Parameters:
            rewards (dict): Dictionary of rewards for all objectives.
            
        Returns:
            dict: Dictionary of transformed rewards for all objectives.
        """
        transformed_rewards = {}
        for key, value in rewards.items():
            scaling = self.scaling_dict.get(key, 1)
            offset = self.offset_dict.get(key, 0)
            transformed_rewards[key] = value * scaling + offset
        return transformed_rewards
    
    def scalarize(self,rewards):
        """
        Applies a scalarization method to a multi-objective reward set. 
        Before scalarization, rewards are transformed based on pre-defined scaling and offset values.
        
        Parameters:
            rewards (dict): Dictionary of rewards for all objectives.
        
        Returns:
            float: Scalarized reward value.
        """
        transformed_rewards = self.apply_scaling_and_offset(rewards)
        
        
        return self.func(transformed_rewards)
    
    def max_min(self,rewards):
        """
        Max-min scalarization.
        
        Parameters:
            rewards (dict): Dictionary of rewards for all objectives.
            
        Returns:
            float: Scalarized reward.
        """
        r = torch.tensor(list(rewards.values()), dtype=torch.float32)
        return torch.min(r).item()
    
    
    def soft_max_min(self,rewards):
        """
        Soft max-min scalarization.
        
        Parameters:
            rewards (dict): Dictionary of rewards for all objectives.
            
        Returns:
            float: Scalarized reward.
        """
        r = torch.tensor(list(rewards.values()), dtype=torch.float32)
        return -torch.log(torch.sum(torch.exp(-r))).item()
    
    def equal_weight(self,rewards):
        """
        Weighted average scalarization.
        
        Parameters:
            rewards (dict): Dictionary of rewards for all objectives.
            weights (dict): Dictionary of weights for all objectives.
            
        Returns:
            float: Scalarized reward.
        """
        r = torch.tensor(list(rewards.values()), dtype=torch.float32)
        return torch.sum(r).item()
