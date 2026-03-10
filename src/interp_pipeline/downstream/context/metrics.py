from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score

from .neighbors import tile_ids


def mean_neighbor_purity(labels: np.ndarray, neigh: List[np.ndarray]) -> float:
    """
    For each cell: fraction of neighbors with same label, then average.
    """
    N = labels.shape[0]
    acc = 0.0
    cnt = 0
    for i in range(N):
        idx = neigh[i]
        if idx.size <= 1:
            continue
        same = (labels[idx] == labels[i]).mean()
        acc += float(same)
        cnt += 1
    return acc / max(1, cnt)


def tile_shuffle_null_purity(
    labels: np.ndarray,
    coords: np.ndarray,
    neigh: List[np.ndarray],
    *,
    tile_size: float,
    n_null: int,
    seed: int,
) -> Tuple[float, float]:
    """
    Shuffle labels within tiles to preserve coarse spatial distribution.
    Returns mean/std purity under null.
    """
    rng = np.random.default_rng(seed)
    tids = tile_ids(coords, tile_size=float(tile_size))
    uniq = np.unique(tids)

    null_vals = []
    for _ in range(int(n_null)):
        shuf = labels.copy()
        for t in uniq:
            mask = np.where(tids == t)[0]
            if mask.size <= 1:
                continue
            perm = rng.permutation(mask.size)
            shuf[mask] = shuf[mask][perm]
        null_vals.append(mean_neighbor_purity(shuf, neigh))
    return float(np.mean(null_vals)), float(np.std(null_vals) + 1e-12)


def separability_cosine(Z: np.ndarray, labels: np.ndarray) -> float:
    """
    Mean cosine distance between cluster centroids (upper triangle).
    """
    labs = np.unique(labels)
    if labs.size <= 1:
        return 0.0
    cents = []
    for k in labs:
        idx = np.where(labels == k)[0]
        if idx.size == 0:
            continue
        cents.append(Z[idx].mean(axis=0))
    C = np.stack(cents, axis=0).astype(np.float32)
    # cosine distance = pdist(metric="cosine")
    d = pdist(C, metric="cosine")
    return float(d.mean()) if d.size else 0.0


def min_cluster_size(labels: np.ndarray) -> int:
    vals, cnts = np.unique(labels, return_counts=True)
    return int(cnts.min()) if cnts.size else 0


def stability_ari_over_seeds(label_list: List[np.ndarray]) -> Tuple[float, float]:
    """
    Compare labels[0] to others, return mean/std ARI.
    """
    if len(label_list) <= 1:
        return 1.0, 0.0
    base = label_list[0]
    aris = []
    for L in label_list[1:]:
        aris.append(float(adjusted_rand_score(base, L)))
    return float(np.mean(aris)), float(np.std(aris))


def fragmentation_score_from_neigh(labels: np.ndarray, neigh: List[np.ndarray]) -> float:
    """
    Notebook-style fragmentation proxy:
    For each label, count number of connected components using neighbor adjacency (undirected),
    normalized by total nodes.
    """
    N = labels.shape[0]
    # build adjacency lists from neigh (symmetric enough for radius graph)
    adj = neigh

    score = 0.0
    for k in np.unique(labels):
        nodes = np.where(labels == k)[0]
        if nodes.size == 0:
            continue
        node_set = set(nodes.tolist())
        seen = set()
        comps = 0
        for n in nodes:
            if n in seen:
                continue
            comps += 1
            stack = [int(n)]
            seen.add(int(n))
            while stack:
                u = stack.pop()
                for v in adj[u]:
                    vv = int(v)
                    if vv in node_set and vv not in seen:
                        seen.add(vv)
                        stack.append(vv)
        score += comps
    return float(score / max(1, N))