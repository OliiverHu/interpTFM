"""
Tokenizer for C2S-Scale models.
Wraps a HuggingFace tokenizer to handle cell sentence string encoding/decoding.

For AnnData preprocessing and cell sentence generation, see C2SProcessor.
For task-specific prompt formatting, see C2SPromptFormatter.
"""

from typing import Dict, List, Optional, Union

import torch
from transformers import AutoTokenizer

from .util import to_device


class C2STokenizer:
    """
    Pure text tokenizer wrapper for C2S models.

    Accepts pre-computed cell sentence strings (or any text) and encodes them
    to tensors suitable for the model. Does not handle AnnData or vocabulary.

    Typical usage::

        # processor converts AnnData → cell sentence strings
        sentences = processor.transform(adata_batch)
        # optionally wrap in a task prompt
        prompts = formatter.format(sentences, organism="Homo sapiens")
        # tokenize
        encoded = tokenizer(prompts, max_length=512)
    """

    def __init__(
        self,
        hf_tokenizer: AutoTokenizer,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        Args:
            hf_tokenizer: HuggingFace tokenizer from the C2S model.
            device:       Device to place output tensors on.
        """
        self.hf_tokenizer = hf_tokenizer
        self.device = device if device is not None else "cpu"

        if self.hf_tokenizer.pad_token is None:
            self.hf_tokenizer.pad_token = self.hf_tokenizer.eos_token

    def __call__(
        self,
        prompt_list
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize a list of cell sentence (or prompt) strings.

        Args:
            prompt_list:  List of strings to tokenize.

        Returns:
            Dict with input_ids and attention_mask tensors on self.device.
        """
        encoded = self.hf_tokenizer(
            prompt_list,
            padding=True,
            return_tensors="pt",
        )
        return to_device(dict(encoded), self.device)

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> List[str]:
        """
        Decode token IDs back to text.

        Args:
            token_ids:            [B, T] tensor of token IDs.
            skip_special_tokens:  Whether to skip special tokens.

        Returns:
            List of decoded strings.
        """
        return self.hf_tokenizer.batch_decode(token_ids, skip_special_tokens=skip_special_tokens)

    def decode_and_remove_prompt(
        self,
        token_ids: torch.Tensor,
        prompts: List[str],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """
        Decode token IDs and strip the input prompt from each output.
        Useful when the model generates text that includes the prompt prefix.

        Args:
            token_ids:           [B, T] tensor of token IDs (e.g. from model.generate()).
            prompts:             List of input prompt strings to remove.
            skip_special_tokens: Whether to skip special tokens.

        Returns:
            List of decoded strings with prompts removed.
        """
        decoded = self.hf_tokenizer.batch_decode(token_ids, skip_special_tokens=skip_special_tokens)
        results = []
        for pred, prompt in zip(decoded, prompts):
            pred = pred.replace(prompt, "", 1)
            pred = pred.replace("<|endoftext|>", "")
            results.append(pred.strip())
        return results

    @property
    def pad_token_id(self) -> int:
        return self.hf_tokenizer.pad_token_id

    @property
    def eos_token_id(self) -> int:
        return self.hf_tokenizer.eos_token_id