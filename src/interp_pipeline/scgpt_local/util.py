from typing import Dict, List, Mapping, Optional, Tuple, Union
import torch


def load_pretrained(
    model: torch.nn.Module,
    pretrained_params: Mapping[str, torch.Tensor],
    prefix: Optional[List[str]] = None,
    verbose: bool = False,
) -> torch.nn.Module:
    """
    Load pretrained weights to the model.

    Args:
        model (torch.nn.Module): The model to load weights to.
        pretrained_params (Mapping[str, torch.Tensor]): The pretrained parameters.
        strict (bool): Whether to strictly enforce that the keys in :attr:`pretrained_params`
            match the keys returned by this module's :meth:`Module.state_dict`. Default to False.
        prefix (List[str]): The list of prefix strings to match with the keys in
            :attr:`pretrained_params`. The matched keys will be loaded. Default to None.

    Returns:
        torch.nn.Module: The model with pretrained weights.
    """
    pretrained_params = {
        k.replace("Wqkv.", "in_proj_"): v for k, v in pretrained_params.items()
    }
    
    if prefix is not None and len(prefix) > 0:
        if isinstance(prefix, str):
            prefix = [prefix]
        pretrained_params = {
            k: v
            for k, v in pretrained_params.items()
            if any(k.startswith(p) for p in prefix)
        }

    model_dict = model.state_dict()
    
    pretrained_params = {
        k: v
        for k, v in pretrained_params.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }

    if verbose:
        keys1 = pretrained_params.keys()
        keys2 = model_dict.keys()

        key_intersection = set(keys1).intersection(set(keys2))
        key1_only = set(keys1) - set(keys2)
        key2_only = set(keys2) - set(keys1)

        if (len(key2_only) == 0) and (len(key1_only) == 0):
            print("All keys matched.")
        else:
            print(f"Keys in both models: {len(key_intersection)}")
            if len(key1_only):
                print(f"Keys only in pretrained: {len(key1_only)}")
                print("Keys only in pretrained_params:", key1_only)
            if len(key2_only):
                print(f"Keys only in model: {len(key2_only)}")
                print("Keys only in model:", key2_only)

    model_dict.update(pretrained_params)
    model.load_state_dict(model_dict)

    return model

def to_device(data, device):
    """
    Recursively move tensors in a nested dictionary to CPU.
    """
    if isinstance(data, dict):
        # If it's a dictionary, apply recursively to each value
        return {key: to_device(value, device) for key, value in data.items()}
    elif isinstance(data, list) or isinstance(data, tuple):
        # If it's a list, apply recursively to each element
        return [to_device(item, device) for item in data]
    elif isinstance(data, torch.Tensor):
        # If it's a tensor, move it to CPU
        return data.to(device)
    else:
        # If it's neither, return it as is
        return data