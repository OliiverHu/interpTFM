from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SAESpec:
    n_latents: int = 4096
    l1: float = 1e-3
    lr: float = 1e-4
    steps: int = 50_000
    warmup_steps: int = 1_000
    resample_steps: int = 2_000
    seed: int = 0

@dataclass
class SAEResult:
    layer: str
    model_path: str
    summary: Dict[str, Any]


class AutoEncoder(nn.Module):
    """
    Minimal SAE: x -> enc -> relu -> dec
    Saved checkpoints from trainer.py are compatible with this class.
    """
    def __init__(self, d_in: int, n_latents: int):
        super().__init__()
        self.encoder = nn.Linear(d_in, n_latents, bias=False)
        self.decoder = nn.Linear(n_latents, d_in, bias=False)

    def forward(self, x: torch.Tensor):
        z = F.relu(self.encoder(x))
        x_hat = self.decoder(z)
        return x_hat, z

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.encoder(x))