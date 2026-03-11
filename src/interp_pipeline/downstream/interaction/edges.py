from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class EdgeConfig:
    radius: float
    weight_mode: str = "exp"   # "exp" | "inverse" | "binary"
    sigma: float = 60.0        # used for exp
    eps: float = 1e-8


def build_edges_radius(coords: np.ndarray, radius: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Undirected edges within radius. Returns (i, j, dist) with i<j.
    """
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=float(radius), output_type="ndarray")
    if pairs.size == 0:
        return (
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.float32),
        )
    i = pairs[:, 0].astype(np.int32)
    j = pairs[:, 1].astype(np.int32)
    d = np.linalg.norm(coords[i] - coords[j], axis=1).astype(np.float32)
    return i, j, d


def distance_weights(d: np.ndarray, cfg: EdgeConfig) -> np.ndarray:
    d = d.astype(np.float32)
    if cfg.weight_mode == "binary":
        return np.ones_like(d, dtype=np.float32)
    if cfg.weight_mode == "inverse":
        return (1.0 / (d + float(cfg.eps))).astype(np.float32)
    if cfg.weight_mode == "exp":
        sig2 = float(cfg.sigma) ** 2 + float(cfg.eps)
        return np.exp(-(d * d) / (2.0 * sig2)).astype(np.float32)
    raise ValueError(f"Unknown weight_mode: {cfg.weight_mode}")


def tile_ids(coords: np.ndarray, tile_size: float) -> np.ndarray:
    x = coords[:, 0]
    y = coords[:, 1]
    tx = np.floor((x - x.min()) / float(tile_size)).astype(np.int32)
    ty = np.floor((y - y.min()) / float(tile_size)).astype(np.int32)
    return (tx.astype(np.int64) << 32) ^ ty.astype(np.int64)