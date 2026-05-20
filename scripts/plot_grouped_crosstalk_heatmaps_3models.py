#!/usr/bin/env python
"""
Plot grouped celltype-group x celltype-group interpretable crosstalk heatmaps.

Input:
  combined_pair_crosstalk.csv from test_crosstalk_3models.py

This script makes combined/group-level heatmaps, not full 22-celltype heatmaps.

It aggregates celltype-celltype pairs into group-group pairs using the existing
group_a/group_b columns in combined_pair_crosstalk.csv.

Recommended first use:
python plot_grouped_crosstalk_heatmaps_3models.py \
  --pair-csv /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/crosstalk_3models_v1/combined_pair_crosstalk.csv \
  --out-dir /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/crosstalk_3models_v1/grouped_heatmaps \
  --domains global niche0 niche1 niche2 \
  --score-col mean_norm_z_score \
  --min-edges 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_GROUP_ORDER = [
    "tumor",
    "macrophage",
    "t_cd8_memory",
    "t_cd4_memory",
    "fibroblast",
    "endothelial",
    "treg",
    "nk",
    "b_plasma",
    "dendritic",
    "monocyte",
    "mast",
    "neutrophil",
    "t_cd4_naive",
    "t_cd8_naive",
    "epithelial_non_tumor",
    "other",
]


TARGET_AXES = [
    ("fibroblast", "t_cd4_memory", "fibroblast/endothelial ↔ T CD4 memory"),
    ("endothelial", "t_cd4_memory", "fibroblast/endothelial ↔ T CD4 memory"),
    ("macrophage", "tumor", "macrophage/T CD8 memory ↔ tumor"),
    ("t_cd8_memory", "tumor", "macrophage/T CD8 memory ↔ tumor"),
]


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def pair_key(a: str, b: str) -> str:
    a, b = str(a), str(b)
    return f"{a}||{b}" if a <= b else f"{b}||{a}"


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    w = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not np.any(m):
        return np.nan
    return float(np.sum(v[m] * w[m]) / np.sum(w[m]))


def load_pairs(path: str, min_edges: int, score_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"model", "domain", "group_a", "group_b", "n_edges", score_col}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"pair CSV missing columns: {missing}")

    df["group_a"] = df["group_a"].astype(str)
    df["group_b"] = df["group_b"].astype(str)
    df["group_pair_key"] = [pair_key(a, b) for a, b in zip(df["group_a"], df["group_b"])]
    df["n_edges"] = pd.to_numeric(df["n_edges"], errors="coerce").fillna(0).astype(float)
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df = df[df["n_edges"] >= int(min_edges)].copy()
    return df


def aggregate_group_pairs(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    rows = []
    for (model, domain, gkey), sub in df.groupby(["model", "domain", "group_pair_key"], dropna=False):
        ga, gb = gkey.split("||", 1)
        rows.append(
            {
                "model": model,
                "domain": domain,
                "group_pair_key": gkey,
                "group_a": ga,
                "group_b": gb,
                "n_celltype_pairs": int(len(sub)),
                "n_edges_total": int(sub["n_edges"].sum()),
                "weighted_score": weighted_mean(sub[score_col], sub["n_edges"]),
                "max_score": float(sub[score_col].max()),
                "median_score": float(sub[score_col].median()),
                "n_flagged_celltype_pairs": int(sub.get("flag_significant_pair", pd.Series(False, index=sub.index)).astype(bool).sum())
                if "flag_significant_pair" in sub.columns else np.nan,
            }
        )
    return pd.DataFrame(rows)


def make_matrix(agg: pd.DataFrame, model: str, domain: str, groups: Sequence[str], value_col: str) -> Tuple[np.ndarray, np.ndarray]:
    sub = agg[(agg["model"] == model) & (agg["domain"] == domain)].copy()
    mat = pd.DataFrame(np.nan, index=list(groups), columns=list(groups), dtype=float)
    nmat = pd.DataFrame(0.0, index=list(groups), columns=list(groups), dtype=float)
    for _, r in sub.iterrows():
        a, b = str(r["group_a"]), str(r["group_b"])
        if a not in mat.index or b not in mat.index:
            continue
        val = r[value_col]
        mat.loc[a, b] = val
        mat.loc[b, a] = val
        nmat.loc[a, b] = r["n_edges_total"]
        nmat.loc[b, a] = r["n_edges_total"]
    return mat.to_numpy(dtype=float), nmat.to_numpy(dtype=float)


def add_target_boxes(ax, groups: Sequence[str]) -> None:
    idx = {g: i for i, g in enumerate(groups)}
    for a, b, _name in TARGET_AXES:
        if a not in idx or b not in idx:
            continue
        i, j = idx[a], idx[b]
        for x, y in [(j, i), (i, j)]:
            rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, linewidth=2)
            ax.add_patch(rect)


def plot_heatmap(
    mat: np.ndarray,
    groups: Sequence[str],
    title: str,
    out_path: Path,
    vmin: float,
    vmax: float,
    annotate: bool,
    nmat: np.ndarray | None = None,
    min_annot_edges: int = 0,
) -> None:
    fig_w = max(7, 0.55 * len(groups) + 2)
    fig_h = max(6, 0.55 * len(groups) + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(mat, vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xticks(np.arange(len(groups)))
    ax.set_yticks(np.arange(len(groups)))
    ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(groups, fontsize=8)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("grouped crosstalk score", rotation=90)

    add_target_boxes(ax, groups)

    if annotate:
        for i in range(len(groups)):
            for j in range(len(groups)):
                val = mat[i, j]
                if not np.isfinite(val):
                    continue
                if nmat is not None and nmat[i, j] < min_annot_edges:
                    continue
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def select_group_order(agg: pd.DataFrame, explicit_order: Sequence[str] | None) -> List[str]:
    present = sorted(set(agg["group_a"].astype(str)).union(set(agg["group_b"].astype(str))))
    if explicit_order:
        order = [g for g in explicit_order if g in present]
    else:
        order = [g for g in DEFAULT_GROUP_ORDER if g in present]
    order += [g for g in present if g not in order]
    return order


def summarize_target_axes(agg: pd.DataFrame) -> pd.DataFrame:
    target_keys = []
    for a, b, name in TARGET_AXES:
        target_keys.append((pair_key(a, b), name, a, b))

    rows = []
    for _, r in agg.iterrows():
        gkey = str(r["group_pair_key"])
        for tk, name, a, b in target_keys:
            if gkey == tk:
                row = r.to_dict()
                row["target_axis"] = name
                row["target_side_a"] = a
                row["target_side_b"] = b
                rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--domains", nargs="+", default=["global", "niche0", "niche1", "niche2"])
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--score-col", default="mean_norm_z_score")
    ap.add_argument("--value-col", default="weighted_score", choices=["weighted_score", "max_score", "median_score"])
    ap.add_argument("--min-edges", type=int, default=50)
    ap.add_argument("--vmin", type=float, default=-1.5)
    ap.add_argument("--vmax", type=float, default=1.5)
    ap.add_argument("--annotate", action="store_true")
    ap.add_argument("--min-annot-edges", type=int, default=200)
    ap.add_argument("--group-order", nargs="*", default=None)
    args = ap.parse_args()

    out_dir = ensure_dir(Path(args.out_dir))
    plot_dir = ensure_dir(out_dir / "plots")
    table_dir = ensure_dir(out_dir / "tables")

    df = load_pairs(args.pair_csv, args.min_edges, args.score_col)
    agg = aggregate_group_pairs(df, args.score_col)
    agg.to_csv(table_dir / "grouped_pair_crosstalk_summary.csv", index=False)

    target_df = summarize_target_axes(agg)
    target_df.to_csv(table_dir / "target_axes_grouped_summary.csv", index=False)

    groups = select_group_order(agg, args.group_order)
    models = args.models if args.models is not None else list(pd.unique(agg["model"]))

    # Save matrices as CSV too.
    matrix_records = []
    for model in models:
        for domain in args.domains:
            mat, nmat = make_matrix(agg, model, domain, groups, args.value_col)
            mat_df = pd.DataFrame(mat, index=groups, columns=groups)
            n_df = pd.DataFrame(nmat, index=groups, columns=groups)
            mat_df.to_csv(table_dir / f"matrix_{model}_{domain}_{args.value_col}.csv")
            n_df.to_csv(table_dir / f"matrix_{model}_{domain}_n_edges.csv")

            safe_domain = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in domain)
            safe_model = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in model)
            title = f"{model} {domain}: grouped crosstalk ({args.value_col}, min_edges={args.min_edges})"
            plot_heatmap(
                mat=mat,
                groups=groups,
                title=title,
                out_path=plot_dir / f"grouped_heatmap_{safe_model}_{safe_domain}_{args.value_col}.png",
                vmin=args.vmin,
                vmax=args.vmax,
                annotate=args.annotate,
                nmat=nmat,
                min_annot_edges=args.min_annot_edges,
            )

            for i, ga in enumerate(groups):
                for j, gb in enumerate(groups):
                    if j < i:
                        continue
                    matrix_records.append(
                        {
                            "model": model,
                            "domain": domain,
                            "group_a": ga,
                            "group_b": gb,
                            "value": mat[i, j],
                            "n_edges_total": nmat[i, j],
                        }
                    )

    pd.DataFrame(matrix_records).to_csv(table_dir / "grouped_heatmap_matrix_long.csv", index=False)

    with open(out_dir / "run_summary.json", "w") as f:
        json.dump(
            {
                "pair_csv": args.pair_csv,
                "score_col": args.score_col,
                "value_col": args.value_col,
                "min_edges": args.min_edges,
                "domains": args.domains,
                "models": models,
                "group_order": groups,
                "target_axes": TARGET_AXES,
            },
            f,
            indent=2,
        )

    print("Done.")
    print(f"Tables: {table_dir}")
    print(f"Plots:  {plot_dir}")


if __name__ == "__main__":
    main()
