import torch
import torch.nn.functional as F

def base_scale_act(scale_enc, act_type="relu"):
    """
    Apply basic activation function to scale parameters.

    Args:
        scale_enc: Tensor containing encoded scale parameters
        act_type: Activation type ("linear", "inv_log", "exp", "relu")

    Returns:
        Activated scale parameters
    """
    if act_type == "linear":
        return scale_enc
    elif act_type == "inv_log":
        return inverse_log_transform(scale_enc)
    elif act_type == "exp":
        return torch.exp(scale_enc)
    elif act_type == "relu":
        return F.relu(scale_enc)
    else:
        raise ValueError(f"Unknown act_type: {act_type}")
    

def inverse_log_transform(y):
    """
    Apply inverse log transform: sign(y) * (exp(|y|) - 1)

    Args:
        y: Input tensor

    Returns:
        Transformed tensor
    """
    return torch.sign(y) * (torch.expm1(torch.abs(y)))
