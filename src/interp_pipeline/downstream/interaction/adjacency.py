from __future__ import annotations

import numpy as np
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

from .grouping import pair_ids, permute_within_tiles


def bh_fdr(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    ok = np.isfinite(p)
    q = np.full_like(p, np.nan, dtype=float)
    if ok.sum():
        _, qvals, _, _ = multipletests(p[ok], alpha=float(alpha), method="fdr_bh")
        q[ok] = qvals
    return q


def adjacency_counts(
    i: np.ndarray,
    j: np.ndarray,
    w: np.ndarray,
    group_codes: np.ndarray,
    G: int,
    *,
    use_weight: bool = True,
) -> np.ndarray:
    """
    Observed adjacency exposure per pair p (symmetric unordered).
    Returns obs[p] where obs is either sum(w) or edge_count.
    """
    P = int(G) * int(G)
    pid = pair_ids(group_codes[i], group_codes[j], G, ordered=False)
    if use_weight:
        obs = np.bincount(pid, weights=w.astype(np.float64), minlength=P).astype(np.float64)
    else:
        obs = np.bincount(pid, minlength=P).astype(np.float64)
    return obs


def adjacency_null(
    i: np.ndarray,
    j: np.ndarray,
    w: np.ndarray,
    group_codes: np.ndarray,
    tiles: np.ndarray,
    G: int,
    *,
    n_perm: int,
    seed: int,
    use_weight: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Tile-shuffle null for adjacency exposure.
    Returns (mean[p], std[p]) in float64.
    """
    rng = np.random.default_rng(int(seed))
    P = int(G) * int(G)
    sum_ = np.zeros(P, dtype=np.float64)
    sumsq = np.zeros(P, dtype=np.float64)

    for _ in range(int(n_perm)):
        perm = permute_within_tiles(group_codes, tiles, rng)
        pid = pair_ids(perm[i], perm[j], G, ordered=False)

        if use_weight:
            x = np.bincount(pid, weights=w.astype(np.float64), minlength=P).astype(np.float64)
        else:
            x = np.bincount(pid, minlength=P).astype(np.float64)

        sum_ += x
        sumsq += x * x

    mean = sum_ / float(n_perm)
    var = np.maximum(sumsq / float(n_perm) - mean * mean, 1e-12)
    std = np.sqrt(var)
    return mean, std


def adjacency_z_and_fdr(
    obs: np.ndarray,
    null_mean: np.ndarray,
    null_std: np.ndarray,
    *,
    eps: float = 1e-6,
    alpha_fdr: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (Z, qvals, sigmask).
    """
    Z = (obs - null_mean) / (null_std + float(eps))
    p = 2.0 * (1.0 - norm.cdf(np.abs(Z)))
    q = bh_fdr(p, alpha=float(alpha_fdr))
    sig = (q < float(alpha_fdr))
    return Z.astype(np.float32), q.astype(np.float32), sig