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
    One-layer sparse autoencoder with:
      - learned activation-space bias
      - encoder bias
      - unit-norm decoder columns at init

    This restores the older SAE geometry that is much less prone to the
    dense-all-on behavior we observed with the minimal biasless SAE.
    """

    def __init__(self, d_in: int, n_latents: int):
        super().__init__()
        self.activation_dim = d_in
        self.dict_size = n_latents

        self.bias = nn.Parameter(torch.zeros(d_in))
        self.encoder = nn.Linear(d_in, n_latents, bias=True)
        self.decoder = nn.Linear(n_latents, d_in, bias=False)

        with torch.no_grad():
            dec_weight = torch.randn_like(self.decoder.weight)
            dec_weight = dec_weight / dec_weight.norm(dim=0, keepdim=True).clamp_min(1e-12)
            self.decoder.weight.copy_(dec_weight)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.encoder(x - self.bias))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z) + self.bias

    def forward(self, x: torch.Tensor, output_features: bool = False):
        z = self.encode(x)
        x_hat = self.decode(z)
        if output_features:
            return x_hat, z
        return x_hat, z

    @classmethod
    def from_pretrained(cls, path: str, device: str | None = None) -> "AutoEncoder":
        obj = torch.load(path, map_location=device or "cpu")
        if not isinstance(obj, dict):
            raise TypeError(f"Unsupported checkpoint format at {path}: {type(obj)}")

        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            state = obj["state_dict"]
            d_in = int(obj.get("d_in", state["encoder.weight"].shape[1]))
            n_latents = int(obj.get("n_latents", state["encoder.weight"].shape[0]))
        else:
            state = obj
            d_in = int(state["encoder.weight"].shape[1])
            n_latents = int(state["encoder.weight"].shape[0])

        ae = cls(d_in=d_in, n_latents=n_latents)
        ae.load_state_dict(state, strict=True)
        if device is not None:
            ae.to(device)
        return ae
