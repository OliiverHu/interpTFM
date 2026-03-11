from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class PreprocessAConfig:
    q_low: float = 0.05
    q_high: float = 0.95


@dataclass(frozen=True)
class PreprocessBConfig:
    clip: float = 6.0
    eps: float = 1e-8


def preprocess_variant_a(X: np.ndarray, cfg: PreprocessAConfig) -> np.ndarray:
    """
    Notebook-ish: clip each dim to [q_low,q_high] and subtract median.
    """
    X = X.astype(np.float32)
    lo = np.quantile(X, cfg.q_low, axis=0)
    hi = np.quantile(X, cfg.q_high, axis=0)
    Xm = np.clip(X, lo, hi)
    med = np.median(Xm, axis=0)
    return (Xm - med).astype(np.float32)


def preprocess_variant_b(X: np.ndarray, cfg: PreprocessBConfig) -> np.ndarray:
    """
    Robust z-score with MAD, then clip.
    """
    X = X.astype(np.float32)
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0)
    mad = np.maximum(mad, cfg.eps)
    Z = (X - med) / (1.4826 * mad)
    if cfg.clip is not None:
        Z = np.clip(Z, -float(cfg.clip), float(cfg.clip))
    return Z.astype(np.float32)