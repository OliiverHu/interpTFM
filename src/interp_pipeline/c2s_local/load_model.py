"""
Model loading utilities for C2S-Scale.
"""

import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from nnsight import LanguageModel

from .tokenizer import C2STokenizer


@dataclass
class C2SConfig:
    """Configuration for C2S model."""
    model_name_or_path: str
    cache_dir: Optional[str] = None
    device: str = "cuda"
    max_genes: Optional[int] = None  # Max genes per cell sentence


def load_c2s_model(
    model_name_or_path: str,
    cache_dir: Optional[str] = None,
    device: Union[str, torch.device] = "cuda",
) -> Tuple[PreTrainedModel, C2STokenizer]:
    """
    Load a C2S-Scale model and its tokenizer.

    For data preprocessing (AnnData → cell sentences), construct a C2SProcessor
    separately. For task-specific prompts, use C2SPromptFormatter.

    Args:
        model_name_or_path: HuggingFace model ID or local path.
            Common options:
            - "vandijklab/C2S-Pythia-410m-cell-type-conditioned-cell-generation"
            - "vandijklab/C2S-Scale-1B"
            - Local path to saved model
        cache_dir: Directory to cache downloaded models
        device: Device to load model on

    Returns:
        Tuple of (model, tokenizer)
    """
    device = torch.device(device) if isinstance(device, str) else device

    # Set up cache directory
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "c2s_models")
    os.makedirs(cache_dir, exist_ok=True)

    # Load HuggingFace tokenizer
    hf_tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        padding_side='left',
        cache_dir=cache_dir,
    )
    if hf_tokenizer.pad_token is None:
        hf_tokenizer.pad_token = hf_tokenizer.eos_token

    # Load HuggingFace model
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    # Wrap tokenizer
    tokenizer = C2STokenizer(
        hf_tokenizer=hf_tokenizer,
        device=device,
    )

    return LanguageModel(model), tokenizer


# def get_layer_names(model: PreTrainedModel) -> list:
#     """
#     Get the list of transformer layer names for a C2S model.

#     C2S models are based on GPT-NeoX (Pythia), so layers are at:
#     model.gpt_neox.layers[i]

#     Args:
#         model: The loaded C2S model

#     Returns:
#         List of layer name strings
#     """
#     # Check if it's a GPT-NeoX model
#     if hasattr(model, 'gpt_neox'):
#         n_layers = len(model.gpt_neox.layers)
#         return [f"gpt_neox.layers.{i}" for i in range(n_layers)]
#     # Check if it's wrapped in a CausalLM wrapper
#     elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
#         n_layers = len(model.model.layers)
#         return [f"model.layers.{i}" for i in range(n_layers)]
#     else:
#         raise ValueError(
#             f"Unknown model architecture. Expected GPT-NeoX based model, "
#             f"got {type(model)}"
#         )


def get_model_hidden_size(model: PreTrainedModel) -> int:
    """Get the hidden dimension of the model."""
    if hasattr(model, 'config'):
        if hasattr(model.config, 'hidden_size'):
            return model.config.hidden_size
        elif hasattr(model.config, 'n_embd'):
            return model.config.n_embd
    raise ValueError("Could not determine hidden size from model config")
