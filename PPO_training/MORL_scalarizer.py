import torch
import json
import numpy as np
# This file contains different Multi-Objective RL scalarization functions which will be tested
class MORLScalarizer:
    """
    Applies a scalarization method to a multi-objective reward set. 
    Before scalarization, rewards are transformed based on pre-defined scaling and offset values.
    """
    
    def __init__(self, func,weight_file,uncertainty_weight=0.5,soft_max_min_temperature=1.0):
        """
        Initialize the scalarizer with a specific scalarization function.

        Parameters:
            func (str): The scalarization function to apply.
        """

        func_dict = {
            "max_min": self.max_min,
            "minimax": self.max_min,
            "worst_case": self.max_min,
            "soft_max_min": self.soft_max_min,
            "soft_minimax": self.soft_max_min,
            "max_avg": self.linear,
            "linear": self.linear,
            "zero_syco": self.zero_syco,
            "uncertainty_weighted": self.uncertainty_weighted,
            "lower_third": self.lower_third,
            "max_median": self.max_median,
            "bernoulli_nash": self.bernoulli_nash,
        }
        self.func = func_dict[func]
        self.uncertainty_weight = uncertainty_weight
        self.temperature = soft_max_min_temperature
        
        # Read the preference file and store the weights
        self.preference_weights = json.load(open(weight_file))
    

    
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
        numpy_array = np.array(list(rewards.values()))
        r = torch.tensor(numpy_array, dtype=torch.float32)
        min_values = torch.min(r, dim=0)[0] 
        return min_values
    
    
    def soft_max_min(self,rewards):
        """
        Soft max-min scalarization.
        
        Parameters:
            rewards (dict): Dictionary of rewards for all objectives.
            
        Returns:
            float: Scalarized reward.
        """
        numpy_array = np.array(list(rewards.values()))
        r = torch.tensor(numpy_array, dtype=torch.float32)
        
        # Apply softmin function
        exp_neg_r = torch.exp(-r / self.temperature)
        softmin_values = exp_neg_r / torch.sum(exp_neg_r)
        
        # Compute the dot product between softmin values and original rewards
        weighted_sum = torch.dot(softmin_values, r)
        
        return weighted_sum.item()
    
    def linear(self,rewards):
        """
        Weighted average scalarization.
        
        Parameters:
            rewards (dict): Dictionary of rewards for all objectives.
            weights (dict): Dictionary of weights for all objectives.
            
        Returns:
            float: Scalarized reward.
        """
        numpy_array = np.array(list(rewards.values()))
        r = torch.tensor(numpy_array, dtype=torch.float32)
        #print("lin",r)
        return r.sum(dim=0)
    def uncertainty_weighted(self,rewards):
        """
        Calculate the Uncertainty-Weighted Optimization (UWO) reward.

        Args:
            rewards (dict): Dictionary of rewards for all objectives

        Returns:
            float: The UWO reward.
        """
        numpy_array = np.array(list(rewards.values()))
        r = torch.tensor(numpy_array, dtype=torch.float32)
        k = len(r)
        mean_reward = np.mean(r)
        variance = np.sum((r - mean_reward) ** 2) / k
        uwo_reward = mean_reward - self.uncertainty_weight * variance
        return uwo_reward
        
    def zero_syco(self,rewards):
        """
        Zero-sycophancy scalarization.
        Sets the sychophancy PM weight to zero.
        Parameters:
            rewards (dict): Dictionary of rewards for all objectives.
            
        Returns:
            float: Scalarized reward.
        """
        rewards["sycophancy"] = 0
        r = torch.tensor(list(rewards.values()), dtype=torch.float32)
        return torch.sum(r).item()
    
    def lower_third(self,rewards):
        """
        Lower third scalarization.
        Returns the average of the lower third of the rewards.
        Parameters:
            rewards (dict): Dictionary of rewards for all objectives.
            
        Returns:
            float: Scalarized reward.
        """
        numpy_array = np.array(list(rewards.values()))
        r = torch.tensor(numpy_array, dtype=torch.float32)
        sorted_r = torch.sort(r).values
        lower_third = sorted_r[:len(sorted_r)//3]
        return lower_third.mean().item()
    def max_median(self,rewards):
        """
        Max-median scalarization.
        Returns the median of the rewards.
        Parameters:
            rewards (dict): Dictionary of rewards for all objectives.
        Returns:
            float: Scalarized reward.
        """
        numpy_array = np.array(list(rewards.values()))
        r = torch.tensor(numpy_array, dtype=torch.float32)
        return torch.median(r).item()
    def bernoulli_nash(self,rewards):
        """
        Bernoulli-Nash scalarization.
        Returns the n-th root of the product of the rewards.
        Parameters:
            rewards (dict): Dictionary of rewards for all objectives.
        Returns:
            float: Scalarized reward.
        """
        numpy_array = np.array(list(rewards.values()))
        r = torch.tensor(numpy_array, dtype=torch.float32)
        return torch.prod(r).item()
    
