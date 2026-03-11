from __future__ import annotations

from typing import List, Tuple
import numpy as np
import pandas as pd


def encode_groups(labels: pd.Series) -> Tuple[np.ndarray, List[str]]:
    names = pd.Index(labels.astype(str).fillna("NA").unique()).tolist()
    name_to_id = {n: i for i, n in enumerate(names)}
    codes = labels.astype(str).fillna("NA").map(name_to_id).to_numpy(dtype=np.int32)
    return codes, names


def pair_ids(g1: np.ndarray, g2: np.ndarray, G: int, ordered: bool = False) -> np.ndarray:
    """
    If ordered=False: symmetric unordered pairs => min/max.
    Always returns ids in [0, G*G).
    """
    g1 = g1.astype(np.int32, copy=False)
    g2 = g2.astype(np.int32, copy=False)
    if ordered:
        a, b = g1, g2
    else:
        a = np.minimum(g1, g2)
        b = np.maximum(g1, g2)
    return (a.astype(np.int64) * int(G) + b.astype(np.int64)).astype(np.int64)


def decode_pair(p: int, G: int) -> Tuple[int, int]:
    a = int(p) // int(G)
    b = int(p) % int(G)
    return a, b


def permute_within_tiles(group_codes: np.ndarray, tiles: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Notebook-style: sort by tile, then shuffle within contiguous blocks.
    """
    group_codes = group_codes.astype(np.int32, copy=False)
    tiles = tiles.astype(np.int64, copy=False)

    order = np.argsort(tiles, kind="mergesort")  # stable
    tiles_sorted = tiles[order]
    out = group_codes.copy()
    out_sorted = out[order]

    # find block boundaries
    # boundaries at indices where tile changes
    changes = np.nonzero(tiles_sorted[1:] != tiles_sorted[:-1])[0] + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes, [len(order)]))

    for s, e in zip(starts, ends):
        if e - s <= 1:
            continue
        perm = rng.permutation(e - s)
        out_sorted[s:e] = out_sorted[s:e][perm]

    out[order] = out_sorted
    return out