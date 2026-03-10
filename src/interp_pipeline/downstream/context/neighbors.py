from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class NeighborConfig:
    radius: float
    # for gaussian kernel
    kernel: str = "uniform"  # "uniform" | "gaussian"
    sigma_frac: float = 0.25  # sigma = radius * sigma_frac


def radius_neighbors(coords: np.ndarray, radius: float) -> List[np.ndarray]:
    """
    Returns neighbor indices for each cell within radius (including self).
    coords: (N,2)
    """
    tree = cKDTree(coords)
    neigh = tree.query_ball_point(coords, r=float(radius))
    return [np.asarray(x, dtype=np.int32) for x in neigh]


def tile_ids(coords: np.ndarray, tile_size: float) -> np.ndarray:
    """
    Assign each point to a grid tile for label-shuffle null.
    """
    x = coords[:, 0]
    y = coords[:, 1]
    tx = np.floor((x - x.min()) / float(tile_size)).astype(np.int32)
    ty = np.floor((y - y.min()) / float(tile_size)).astype(np.int32)
    # combine to single id
    return (tx.astype(np.int64) << 32) ^ ty.astype(np.int64)