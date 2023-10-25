import torch
# This class contains different Multi-Objective RL scalarization functions which will be tested

def scalarize_MORL(func,rewards):
    func_dict = {
        "max_min":max_min,
        "soft_max_min":soft_max_min,
        "equal_weight":equal_weight
    }
    
    return func_dict[func](rewards)

def max_min(rewards):
    """
    Max-min scalarization.
    
    Parameters:
        rewards (dict): Dictionary of rewards for all objectives.
        
    Returns:
        float: Scalarized reward.
    """
    r = torch.tensor(list(rewards.values()), dtype=torch.float32)
    return torch.min(r).item()


def soft_max_min(rewards):
    """
    Soft max-min scalarization.
    
    Parameters:
        rewards (dict): Dictionary of rewards for all objectives.
        
    Returns:
        float: Scalarized reward.
    """
    r = torch.tensor(list(rewards.values()), dtype=torch.float32)
    return -torch.log(torch.sum(torch.exp(-r))).item()

def equal_weight(rewards):
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
