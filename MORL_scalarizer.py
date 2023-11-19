import torch
# This file contains different Multi-Objective RL scalarization functions which will be tested
class MORLScalarizer:
    """
    Applies a scalarization method to a multi-objective reward set. 
    Before scalarization, rewards are transformed based on pre-defined scaling and offset values.
    """
    
    def __init__(self, func,weight_file):
        """
        Initialize the scalarizer with a specific scalarization function.

        Parameters:
            func (str): The scalarization function to apply. Valid values are "max_min", "soft_max_min", and "equal_weight".
        """
        func_dict = {
            "max_min": self.max_min,
            "soft_max_min": self.soft_max_min,
            "max_avg": self.max_avg
        }
        self.func = func_dict[func]
        
        # Read the preference file and store the contents as a dictionary
        self.preference_weights = {}
        with open(weight_file, "r") as file:
            for line in file:
                key, value = line.strip().split(":")
                self.preference_weights[key] = int(value)
    

    
    def apply_weighting(self, rewards):
        """
        Applies weighting to the given rewards based on the provided weights.

        Parameters:
            rewards (dict): Dictionary of rewards for all objectives.
            weights (dict): Dictionary of weights for all objectives.

        Returns:
            dict: Dictionary of weighted rewards for all objectives.
        """
        weighted_rewards = {}
        for key, value in rewards.items():
            weight = self.preference_weights[key]
            weighted_rewards[key] = value * weight
        return weighted_rewards


    def scalarize(self,rewards):
        """
        Applies a scalarization method to a multi-objective reward set. 
        Before scalarization, rewards are transformed based on pre-defined weights.
        
        Parameters:
            rewards (dict): Dictionary of rewards for all objectives.
        
        Returns:
            float: Scalarized reward value.
        """
        transformed_rewards = self.apply_weighting(rewards)
        
        
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
    
    def max_avg(self,rewards):
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
