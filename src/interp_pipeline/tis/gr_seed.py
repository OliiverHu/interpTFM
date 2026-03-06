from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

from interp_pipeline.tis.core import (
    TISConfig,
    build_pools_quantile,
    compute_tis_mis,
)


@dataclass(frozen=True)
class TISGridConfig:
    seeds: List[int]
    Ks: List[int]
    n_trials_list: List[int]

    # pooling quantiles (keep same as main)
    q_high: float = 0.75
    q_low: float = 0.25

    # reuse other core settings
    exclude_query_from_pools: bool = True
    subsample_eval: Optional[int] = None  # optional speed knob


def _summarize(arr: np.ndarray) -> Dict[str, float]:
    arr = arr.astype(np.float32)
    good = np.isfinite(arr)
    out = {
        "mean": float(np.nanmean(arr)),
        "median": float(np.nanmedian(arr)),
        "p90": float(np.nanpercentile(arr[good], 90)) if good.any() else float("nan"),
        "p99": float(np.nanpercentile(arr[good], 99)) if good.any() else float("nan"),
        "n_valid": int(good.sum()),
    }
    return out


def run_seed_reproducibility(
    A_use: np.ndarray,
    J_use,
    *,
    base_cfg: TISConfig,
    seeds: List[int],
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Returns:
      tis_mat: (len(seeds), U)
      stats: correlation + summary
    """
    top_idx, bot_idx, med = build_pools_quantile(A_use, base_cfg.q_low, base_cfg.q_high)

    tis_list = []
    for sd in seeds:
        cfg = TISConfig(
            K=base_cfg.K,
            n_trials=base_cfg.n_trials,
            seed=int(sd),
            q_high=base_cfg.q_high,
            q_low=base_cfg.q_low,
            exclude_query_from_pools=base_cfg.exclude_query_from_pools,
            subsample_eval=base_cfg.subsample_eval,
            eps=base_cfg.eps,
        )
        tis = compute_tis_mis(A_use, J_use, top_idx, bot_idx, med, cfg)
        tis_list.append(tis)

    tis_mat = np.stack(tis_list, axis=0)  # (S,U)

    # reproducibility metrics: pairwise Pearson corr of TIS vectors
    # compute corr on finite dims only
    S, U = tis_mat.shape
    cors = []
    for i in range(S):
        for j in range(i + 1, S):
            xi = tis_mat[i]
            xj = tis_mat[j]
            m = np.isfinite(xi) & np.isfinite(xj)
            if m.sum() < 10:
                continue
            ci = np.corrcoef(xi[m], xj[m])[0, 1]
            if np.isfinite(ci):
                cors.append(float(ci))

    stats = {
        "n_seeds": int(S),
        "mean_pairwise_corr": float(np.mean(cors)) if cors else float("nan"),
        "median_pairwise_corr": float(np.median(cors)) if cors else float("nan"),
        "min_pairwise_corr": float(np.min(cors)) if cors else float("nan"),
        "max_pairwise_corr": float(np.max(cors)) if cors else float("nan"),
    }
    return tis_mat, stats


def run_light_grid(
    A_use: np.ndarray,
    J_use,
    *,
    grid: TISGridConfig,
) -> "np.ndarray":
    """
    Returns a list-of-dicts style table (as a numpy object array) or you can build DataFrame in caller.
    """
    rows: List[Dict[str, float]] = []

    # pools depend only on A and quantiles, not K/trials/seed
    # (still valid if K changes; pools are indices)
    top_idx, bot_idx, med = build_pools_quantile(A_use, grid.q_low, grid.q_high)

    for K in grid.Ks:
        for ntr in grid.n_trials_list:
            for sd in grid.seeds:
                cfg = TISConfig(
                    K=int(K),
                    n_trials=int(ntr),
                    seed=int(sd),
                    q_high=float(grid.q_high),
                    q_low=float(grid.q_low),
                    exclude_query_from_pools=bool(grid.exclude_query_from_pools),
                    subsample_eval=grid.subsample_eval,
                )
                tis = compute_tis_mis(A_use, J_use, top_idx, bot_idx, med, cfg)
                s = _summarize(tis)
                rows.append(
                    {
                        "K": int(K),
                        "n_trials": int(ntr),
                        "seed": int(sd),
                        **s,
                    }
                )

    return rows


def save_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)