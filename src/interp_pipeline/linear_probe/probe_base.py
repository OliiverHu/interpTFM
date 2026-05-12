from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class LinearProbeSpec:
    """Hyperparameter config for training one linear probe."""
    n_concepts: int
    hidden_size: int
    epochs: int = 20
    batch_size: int = 8192
    lr: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.99)
    weight_decay: float = 0.1
    test_fraction: float = 0.2
    seed: int = 42


class LinearProbe:
    """
    A single weight matrix [hidden_size, n_concepts] mapping token activations
    to concept membership logits.  No bias — matches legacy design.
    """

    def __init__(self, hidden_size: int, n_concepts: int, device: str = "cpu", seed: int = 42):
        self.hidden_size = hidden_size
        self.n_concepts = n_concepts
        self.device = device
        torch.manual_seed(seed)
        self.weight = nn.Parameter(
            torch.randn(hidden_size, n_concepts, device=device) / (hidden_size ** 0.5)
        )

    def project(self, acts: torch.Tensor) -> torch.Tensor:
        """acts: [N, H] → logits: [N, n_concepts]"""
        return acts @ self.weight

    def parameters(self) -> List[nn.Parameter]:
        return [self.weight]

    def to(self, device: str) -> "LinearProbe":
        self.weight = nn.Parameter(self.weight.data.to(device))
        self.device = device
        return self

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu") -> "LinearProbe":
        ckpt = torch.load(path, map_location=device)
        probe = cls(ckpt["hidden_size"], ckpt["n_concepts"], device=device)
        probe.weight = nn.Parameter(ckpt["weight"].to(device))
        return probe


def save_probe(probe: LinearProbe, layer: str, output_dir: str) -> str:
    """
    Save probe checkpoint to output_dir/probe_{layer}.pt.
    Returns the path written.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"probe_{layer}.pt")
    torch.save(
        {
            "weight": probe.weight.detach().cpu(),
            "hidden_size": probe.hidden_size,
            "n_concepts": probe.n_concepts,
            "layer": layer,
        },
        path,
    )
    return path


def load_probe(path: str, device: str = "cpu") -> LinearProbe:
    """Load a probe from a checkpoint file."""
    return LinearProbe.from_checkpoint(path, device=device)
