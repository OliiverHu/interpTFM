from __future__ import annotations

from typing import List, Tuple
import numpy as np

from .neighbors import NeighborConfig, radius_neighbors


def microenv_embedding(
    X: np.ndarray,
    coords: np.ndarray,
    cfg: NeighborConfig,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Build microenvironment embedding m_i = weighted avg of neighbor features.
    X: (N,D) dense float32
    coords: (N,2)
    Returns:
      m: (N,D)
      neigh: list of neighbor idx arrays
    """
    neigh = radius_neighbors(coords, cfg.radius)
    N, D = X.shape
    m = np.zeros((N, D), dtype=np.float32)

    if cfg.kernel == "uniform":
        for i in range(N):
            idx = neigh[i]
            if idx.size == 0:
                continue
            m[i] = X[idx].mean(axis=0)
        return m, neigh

    if cfg.kernel == "gaussian":
        sigma = float(cfg.radius) * float(cfg.sigma_frac)
        sigma2 = sigma * sigma + 1e-12
        for i in range(N):
            idx = neigh[i]
            if idx.size == 0:
                continue
            dx = coords[idx, 0] - coords[i, 0]
            dy = coords[idx, 1] - coords[i, 1]
            w = np.exp(-(dx * dx + dy * dy) / (2.0 * sigma2)).astype(np.float32)
            ws = float(w.sum())
            if ws <= 0:
                continue
            m[i] = (X[idx] * w[:, None]).sum(axis=0) / ws
        return m, neigh

    raise ValueError(f"Unknown kernel: {cfg.kernel}")