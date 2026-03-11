from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import scipy.sparse as sp

from .grouping import pair_ids


@dataclass(frozen=True)
class AggConfig:
    chunk_size: int = 40_000
    ordered_pairs: bool = False


def aggregate_S1_S2_chunked(
    F: np.ndarray,                   # (N,T) float32
    i: np.ndarray, j: np.ndarray,     # (E,) int
    w: np.ndarray,                   # (E,) float
    group_codes: np.ndarray,         # (N,) int
    G: int,
    cfg: AggConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Notebook-style chunked aggregation using sparse incidence matrix:
      prod  = (F[i]*F[j]) * w
      S1 += S @ prod
      S2 += S @ (prod^2)
    Also returns edge_count and weight_sum per pair.
    """
    i = i.astype(np.int32, copy=False)
    j = j.astype(np.int32, copy=False)
    w = w.astype(np.float32, copy=False)
    group_codes = group_codes.astype(np.int32, copy=False)

    E = int(i.shape[0])
    T = int(F.shape[1])
    P = int(G) * int(G)

    S1 = np.zeros((P, T), dtype=np.float32)
    S2 = np.zeros((P, T), dtype=np.float32)
    edge_count = np.zeros((P,), dtype=np.int64)
    weight_sum = np.zeros((P,), dtype=np.float32)

    for e0 in range(0, E, int(cfg.chunk_size)):
        e1 = min(E, e0 + int(cfg.chunk_size))
        ii = i[e0:e1]
        jj = j[e0:e1]
        ww = w[e0:e1]

        gi = group_codes[ii]
        gj = group_codes[jj]
        pid = pair_ids(gi, gj, G, ordered=cfg.ordered_pairs)  # (m,)

        # prod per edge per concept
        prod = (F[ii, :] * F[jj, :]) * ww[:, None]            # (m,T)
        prod2 = prod * prod

        m = int(pid.shape[0])
        cols = np.arange(m, dtype=np.int64)
        S = sp.coo_matrix((np.ones(m, dtype=np.float32), (pid, cols)), shape=(P, m)).tocsr()

        S1 += (S @ prod).astype(np.float32)
        S2 += (S @ prod2).astype(np.float32)

        edge_count += np.bincount(pid, minlength=P).astype(np.int64)
        weight_sum += np.bincount(pid, weights=ww, minlength=P).astype(np.float32)

    return S1, S2, edge_count, weight_sum


def null_tile_shuffle_S1(
    F: np.ndarray,
    i: np.ndarray, j: np.ndarray,
    w: np.ndarray,
    group_codes: np.ndarray,
    G: int,
    tiles: np.ndarray,
    *,
    n_perm: int,
    seed: int,
    cfg: AggConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Notebook-style null accumulation:
      sum(acc), sum(acc^2) over permutations
      mean = sum/n
      std  = sqrt(E[x^2]-mean^2)
    """
    rng = np.random.default_rng(int(seed))

    P = int(G) * int(G)
    T = int(F.shape[1])

    sum_ = np.zeros((P, T), dtype=np.float64)
    sumsq = np.zeros((P, T), dtype=np.float64)

    for _ in range(int(n_perm)):
        from .grouping import permute_within_tiles
        perm_codes = permute_within_tiles(group_codes, tiles, rng)

        S1n, _, _, _ = aggregate_S1_S2_chunked(F, i, j, w, perm_codes, G, cfg)
        acc = S1n.astype(np.float64)

        sum_ += acc
        sumsq += acc * acc

    mean = (sum_ / float(n_perm)).astype(np.float32)
    ex2 = (sumsq / float(n_perm))
    var = np.maximum(ex2 - (mean.astype(np.float64) ** 2), 0.0)
    std = np.sqrt(var).astype(np.float32)

    return mean, std