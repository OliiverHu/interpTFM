from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict

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
