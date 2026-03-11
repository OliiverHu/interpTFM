from __future__ import annotations

from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from .grouping import decode_pair


def top_pairs_summary(
    intensity: np.ndarray,
    edge_count: np.ndarray,
    weight_sum: np.ndarray,
    group_names: List[str],
    G: int,
    top_n: int = 50,
) -> pd.DataFrame:
    ok = np.isfinite(intensity)
    idx = np.where(ok)[0]
    if idx.size == 0:
        return pd.DataFrame()

    idx = idx[np.argsort(-intensity[idx])]
    idx = idx[: int(top_n)]

    rows = []
    for p in idx:
        a, b = decode_pair(int(p), G)
        rows.append(
            dict(
                pair_id=int(p),
                group_a=group_names[a],
                group_b=group_names[b],
                intensity=float(intensity[p]),
                edge_count=int(edge_count[p]),
                weight_sum=float(weight_sum[p]),
            )
        )
    return pd.DataFrame(rows)


def top_drivers_for_pair(
    pair_id: int,
    Z: np.ndarray,
    neff: np.ndarray,
    concept_names: List[str],
    *,
    min_neff: float,
    top_k: int = 20,
) -> pd.DataFrame:
    z = Z[int(pair_id)]
    n = neff[int(pair_id)]
    m = np.isfinite(z) & (n >= float(min_neff))
    if not np.any(m):
        return pd.DataFrame(columns=["concept", "z", "neff"])

    idx = np.where(m)[0]
    idx = idx[np.argsort(-z[idx])]
    idx = idx[: int(top_k)]

    return pd.DataFrame(
        {
            "concept": [concept_names[i] for i in idx],
            "z": [float(z[i]) for i in idx],
            "neff": [float(n[i]) for i in idx],
        }
    )


def pair_table(
    score_vec: np.ndarray,
    group_names: list[str],
    G: int,
    *,
    score_name: str,
    mask_diag: bool = True,
) -> pd.DataFrame:
    rows = []
    for p in range(int(G) * int(G)):
        a, b = decode_pair(p, G)
        if mask_diag and a == b:
            continue
        rows.append((p, group_names[a], group_names[b], float(score_vec[p])))
    return pd.DataFrame(rows, columns=["pair_id", "group_a", "group_b", score_name])