from __future__ import annotations

import os
import numpy as np
import pandas as pd


def plot_metric_tradeoffs(filtered: pd.DataFrame, outdir: str) -> None:
    """
    Notebook cell '#5 Visualize tradeoffs':
    plot each metric vs k, grouped by method and radius.
    """
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)

    def plot_metric(metric: str, title: str, fname: str):
        fig, ax = plt.subplots(figsize=(7, 4))
        for method in filtered["method"].unique():
            sub = filtered[filtered["method"] == method]
            for radius in sorted(sub["radius"].unique()):
                s = sub[sub["radius"] == radius]
                # notebook: ax.plot(s["k"], s[metric], marker="o", label=f"{method}, r={radius}")
                # our CSV uses n_clusters not k
                xcol = "k" if "k" in s.columns else "n_clusters"
                ax.plot(s[xcol], s[metric], marker="o", label=f"{method}, r={radius}")
        ax.set_xlabel("k")
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, fname), dpi=200)
        plt.close(fig)

    plot_metric("ari_mean" if "ari_mean" in filtered.columns else "ARI_mean",
                "Stability (ARI vs seed0)",
                "tradeoff_ari.png")
    plot_metric("purity_z",
                "Spatial coherence over null (z)",
                "tradeoff_purity_z.png")
    plot_metric("separability",
                "Functional separability (cosine dist of cluster means)",
                "tradeoff_separability.png")


def plot_spatial_labels(coords: np.ndarray, labels: np.ndarray, outpath: str, title: str = "", s: int = 6, alpha: float = 0.9) -> None:
    """
    Notebook cell: plot_spatial_labels(ax, coords, labels)
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.scatter(coords[:, 0], coords[:, 1], c=labels, s=s, alpha=alpha)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(outpath, dpi=250)
    plt.close(fig)


def plot_spatial_side_by_side(coords: np.ndarray, labels_a: np.ndarray, labels_b: np.ndarray, outpath: str, title_a: str, title_b: str, s: int = 6, alpha: float = 0.9) -> None:
    """
    Notebook cell: side-by-side plots for k=3 and k=4.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(coords[:, 0], coords[:, 1], c=labels_a, s=s, alpha=alpha)
    axes[0].set_title(title_a)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    axes[1].scatter(coords[:, 0], coords[:, 1], c=labels_b, s=s, alpha=alpha)
    axes[1].set_title(title_b)
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    fig.tight_layout()
    fig.savefig(outpath, dpi=250)
    plt.close(fig)


def stacked_bar_composition(
    tab: pd.DataFrame,
    outpath: str,
    title: str = "",
    min_frac: float = 0.02,
    *,
    top_n: int = 15,
    other_label: str = "Other",
) -> None:
    """
    Stacked bar plot of niche x celltype composition.

    Improvements for many celltypes:
      - keep only top_n celltypes (by total count across niches)
      - merge the rest into `Other`
      - legend outside, wider fig
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # tab is counts (niche x celltype)
    tab = tab.copy()

    # Optional: drop very rare types by global fraction first
    frac_global = tab.sum(axis=0) / max(1, tab.values.sum())
    tab = tab.loc[:, frac_global >= float(min_frac)]

    if tab.shape[1] == 0:
        # nothing to plot
        with open(outpath + ".txt", "w") as f:
            f.write("No cell types passed min_frac filter.\n")
        return

    # Keep top_n by total counts, merge rest into Other
    totals = tab.sum(axis=0).sort_values(ascending=False)
    keep = totals.head(int(top_n)).index.tolist()
    drop = [c for c in tab.columns if c not in keep]

    tab_keep = tab[keep].copy()
    if drop:
        tab_keep[other_label] = tab[drop].sum(axis=1)

    # Convert to fractions per niche
    frac = tab_keep.div(tab_keep.sum(axis=1), axis=0).fillna(0.0)

    niches = frac.index.astype(str).tolist()
    cell_types = frac.columns.tolist()

    # Categorical colormap with enough distinct colors for <= ~20 categories
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % cmap.N) for i in range(len(cell_types))]

    bottom = np.zeros(len(niches), dtype=float)

    fig = plt.figure(figsize=(max(10, 0.6 * len(niches) + 6), 4.5))
    for ct, col in zip(cell_types, colors):
        vals = frac[ct].values
        plt.bar(niches, vals, bottom=bottom, label=ct, color=col)
        bottom += vals

    plt.ylabel("Fraction of cells")
    plt.xlabel("Niche")
    plt.title(title)
    plt.xticks(rotation=0)

    # Legend outside
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, fontsize=8)
    plt.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_top_positive_residuals(resid: pd.DataFrame, outpath: str, title: str = "", top_n: int = 5, z_thresh: float = 1.5) -> None:
    """
    Notebook cell: plot_top_positive_residuals(resid, ...)
    resid: DataFrame niche x cell_type (standardized residuals)
    """
    import matplotlib.pyplot as plt

    niches = resid.index
    cols = set()
    top_per = {}
    for n in niches:
        r = resid.loc[n].copy()
        r = r[r >= float(z_thresh)].sort_values(ascending=False).head(int(top_n))
        top_per[n] = r
        cols.update(r.index.tolist())
    cols = sorted(list(cols))

    if len(cols) == 0:
        # still write an empty note file for reproducibility
        with open(outpath + ".txt", "w") as f:
            f.write(f"No residuals >= {z_thresh}. Try lowering z_thresh.\n")
        return

    M = pd.DataFrame(0.0, index=niches, columns=cols)
    for n in niches:
        for ct, val in top_per[n].items():
            M.loc[n, ct] = float(val)

    fig = plt.figure(figsize=(10, 4))
    bottom = np.zeros(len(niches), dtype=float)
    for ct in M.columns:
        vals = M[ct].values
        if np.all(vals == 0):
            continue
        plt.bar(niches.astype(str), vals, bottom=bottom, label=ct)
        bottom += vals

    plt.ylabel("Positive enrichment (std residual)")
    plt.xlabel("Niche")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def collapse_celltypes(tab: pd.DataFrame, mapping: dict, other_label: str = "Other") -> pd.DataFrame:
    """
    Notebook cell: collapse_celltypes(tab, mapping)
    tab: niche x cell_type counts
    mapping: cell_type -> lineage
    """
    lineage_cols = [mapping.get(ct, other_label) for ct in tab.columns]
    collapsed = tab.copy()
    collapsed.columns = lineage_cols
    collapsed = collapsed.groupby(level=0, axis=1).sum()
    return collapsed