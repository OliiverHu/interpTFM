from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def plot_intensity_scatter(
    intensity: np.ndarray,
    edge_count: np.ndarray,
    weight_sum: np.ndarray,
    outdir: str,
    title_prefix: str,
) -> None:
    """
    Notebook-style confound plots:
      intensity vs edge_count
      intensity vs weight_sum
    """
    import matplotlib.pyplot as plt

    _ensure_dir(outdir)
    ok = np.isfinite(intensity)

    # intensity vs edge_count
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(edge_count[ok], intensity[ok], s=10, alpha=0.5)
    ax.set_xlabel("edge_count")
    ax.set_ylabel("intensity")
    ax.set_title(f"{title_prefix}: intensity vs edge_count")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "scatter_intensity_vs_edge_count.png"), dpi=200)
    plt.close(fig)

    # intensity vs weight_sum
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(weight_sum[ok], intensity[ok], s=10, alpha=0.5)
    ax.set_xlabel("weight_sum")
    ax.set_ylabel("intensity")
    ax.set_title(f"{title_prefix}: intensity vs weight_sum")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "scatter_intensity_vs_weight_sum.png"), dpi=200)
    plt.close(fig)


def plot_within_vs_cross_boxplot(
    intensity: np.ndarray,
    G: int,
    outdir: str,
    title_prefix: str,
) -> None:
    """
    Notebook-style within-vs-cross boxplot:
      within pairs: a==b
      cross pairs: a<b
    intensity is length P=G*G with pair_id = a*G+b (unordered in our code: we store only a<=b but still in that index).
    """
    import matplotlib.pyplot as plt

    _ensure_dir(outdir)

    within = []
    cross = []

    for a in range(G):
        for b in range(G):
            p = a * G + b
            val = intensity[p]
            if not np.isfinite(val):
                continue
            if a == b:
                within.append(val)
            elif a < b:
                cross.append(val)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot([within, cross], labels=["within", "cross"], showfliers=False)
    ax.set_ylabel("intensity")
    ax.set_title(f"{title_prefix}: within vs cross intensity")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "boxplot_within_vs_cross.png"), dpi=220)
    plt.close(fig)


def intensity_matrix(intensity: np.ndarray, G: int) -> np.ndarray:
    """
    Build GxG matrix from length P=G*G.
    """
    M = np.full((G, G), np.nan, dtype=np.float32)
    for a in range(G):
        for b in range(G):
            M[a, b] = intensity[a * G + b]
    return M


def plot_intensity_heatmap(
    intensity: np.ndarray,
    group_names: List[str],
    outdir: str,
    title_prefix: str,
    max_groups: int = 40,
) -> None:
    """
    Heatmap of intensity matrix. If too many groups, keep top-N by within-group intensity.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    _ensure_dir(outdir)
    G = len(group_names)
    M = intensity_matrix(intensity, G)

    # If too many groups, choose subset with highest within intensity (diagonal)
    if G > max_groups:
        diag = np.array([M[i, i] if np.isfinite(M[i, i]) else -np.inf for i in range(G)])
        keep = np.argsort(-diag)[:max_groups]
        keep = np.sort(keep)

        M = M[np.ix_(keep, keep)]
        names = [group_names[i] for i in keep]
    else:
        names = group_names

    fig_h = max(6, 0.25 * len(names))
    fig_w = max(7, 0.25 * len(names))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(M, ax=ax, cmap="viridis", square=True, cbar_kws={"label": "intensity"})
    ax.set_title(f"{title_prefix}: intensity heatmap")
    ax.set_xticks(np.arange(len(names)) + 0.5)
    ax.set_yticks(np.arange(len(names)) + 0.5)
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticklabels(names, rotation=0, fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "heatmap_intensity.png"), dpi=250)
    plt.close(fig)


def plot_top_pair_drivers(
    drivers_df: pd.DataFrame,
    outdir: str,
    title_prefix: str,
    top_n: int = 20,
) -> None:
    """
    Barplot of top driver Z scores for the top pair (or any subset passed in).
    Expects columns: concept, z
    """
    import matplotlib.pyplot as plt

    _ensure_dir(outdir)
    if drivers_df is None or len(drivers_df) == 0:
        return

    d = drivers_df.sort_values("z", ascending=False).head(int(top_n)).copy()
    # reverse for barh
    d = d.iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.25 * len(d) + 1)))
    ax.barh(d["concept"].astype(str), d["z"].astype(float))
    ax.set_xlabel("Z")
    ax.set_title(f"{title_prefix}: top driver concepts (top {top_n})")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "top_pair_driver_concepts.png"), dpi=250)
    plt.close(fig)


def heatmap_masked(
    mat: np.ndarray,
    *,
    title: str,
    names: list[str],
    outpath: str,
    fmt: str = "{:.2f}",
    fontsize: int = 7,
    mask_diag: bool = True,
) -> None:
    import numpy as np
    import matplotlib.pyplot as plt

    M = np.asarray(mat).copy()
    if mask_diag:
        np.fill_diagonal(M, np.nan)

    plt.figure(figsize=(12, 9))
    im = plt.imshow(M, aspect="auto")
    plt.colorbar(im, label=title)
    plt.xticks(range(len(names)), names, rotation=90)
    plt.yticks(range(len(names)), names)
    plt.title(title)

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if mask_diag and i == j:
                continue
            val = M[i, j]
            if np.isfinite(val):
                plt.text(j, i, fmt.format(val), ha="center", va="center", fontsize=fontsize)

    plt.tight_layout()
    plt.savefig(outpath, dpi=250)
    plt.close()

def vec_to_square(vec, G: int):
    import numpy as np
    return np.asarray(vec).reshape(int(G), int(G))