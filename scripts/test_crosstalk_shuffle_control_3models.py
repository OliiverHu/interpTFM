#!/usr/bin/env python
"""
Tile-shuffle controlled grouped crosstalk analysis for 3 models.

This is NOT ligand-receptor CCC. It tests whether model-derived interpretable
program load is elevated on a cell-group pair/domain beyond a local spatial
shuffle null.

Null:
  shuffle cell-level raw/norm interpretable scores within spatial tiles,
  while preserving the spatial graph, cell-type locations, niche labels, and
  local tissue/activation density.

Main outputs:
  combined_grouped_crosstalk_shuffle_control.csv
  combined_target_axes_shuffle_control.csv
  plots/grouped_heatmap_<model>_<domain>_shuffle_norm_z.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree

DEFAULT_GROUP_ORDER = [
    "tumor", "macrophage", "t_cd8_memory", "t_cd4_memory", "fibroblast", "endothelial",
    "treg", "nk", "b_plasma", "dendritic", "monocyte", "mast", "neutrophil",
    "t_cd4_naive", "t_cd8_naive", "epithelial_non_tumor", "other",
]


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def pair_key(a: str, b: str) -> str:
    a, b = str(a), str(b)
    return f"{a}||{b}" if a <= b else f"{b}||{a}"


def split_pair_key(k: str) -> Tuple[str, str]:
    return tuple(str(k).split("||", 1))  # type: ignore


def split_semicolon(x: str) -> List[str]:
    return [z.strip() for z in str(x).split(";") if z.strip()]


def load_mapping(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"cell_type", "group"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"celltype group CSV missing columns: {miss}")
    df["cell_type"] = df["cell_type"].astype(str)
    df["group"] = df["group"].astype(str)
    return df


def load_targets(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"interaction_name", "side_a_groups", "side_b_groups"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"target interaction CSV missing columns: {miss}")
    for c in need:
        df[c] = df[c].astype(str)
    return df


def load_X(a: ad.AnnData) -> np.ndarray:
    X = a.X.toarray() if sparse.issparse(a.X) else np.asarray(a.X)
    X = X.astype(np.float32, copy=False)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def cell_program_scores(X: np.ndarray, clip_quantile: float) -> Tuple[np.ndarray, np.ndarray]:
    Xp = np.maximum(X, 0.0)
    raw = Xp.mean(axis=1).astype(np.float32)
    Xc = Xp.copy()
    if 0.0 < clip_quantile < 1.0:
        hi = np.quantile(Xc, clip_quantile, axis=0).astype(np.float32)
        Xc = np.minimum(Xc, hi[None, :])
    mu = Xc.mean(axis=0)
    sd = Xc.std(axis=0)
    sd[sd < 1e-6] = 1.0
    Z = (Xc - mu[None, :]) / sd[None, :]
    norm = np.maximum(Z, 0.0).mean(axis=1).astype(np.float32)
    return raw, norm


def build_edges(coords: np.ndarray, radius: float) -> np.ndarray:
    tree = cKDTree(coords)
    try:
        pairs = tree.query_pairs(float(radius), output_type="ndarray")
    except TypeError:
        pairs = np.array(list(tree.query_pairs(float(radius))), dtype=np.int64)
    if pairs.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(pairs, dtype=np.int64).reshape((-1, 2))


def make_tiles(coords: np.ndarray, tile_size: float) -> List[np.ndarray]:
    xy = np.floor(coords / float(tile_size)).astype(np.int64)
    tile_id = np.array([f"{x}_{y}" for x, y in xy], dtype=object)
    return [np.where(tile_id == t)[0] for t in pd.unique(tile_id)]


def shuffle_within_tiles(values: np.ndarray, tile_indices: Sequence[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    out = values.copy()
    for idx in tile_indices:
        if len(idx) > 1:
            out[idx] = out[rng.permutation(idx)]
    return out


def edge_score(edges: np.ndarray, cell_score: np.ndarray) -> np.ndarray:
    return (0.5 * (cell_score[edges[:, 0]] + cell_score[edges[:, 1]])).astype(np.float32)


def domain_edge_masks(edges: np.ndarray, niche: np.ndarray) -> dict:
    niche = pd.Series(niche).astype(str).to_numpy()
    out = {"global": np.ones(len(edges), dtype=bool)}
    for v in sorted(pd.unique(niche)):
        cm = niche == str(v)
        out[f"niche{v}"] = cm[edges[:, 0]] & cm[edges[:, 1]]
    return out


def group_pair_keys_for_edges(edges: np.ndarray, groups: np.ndarray) -> np.ndarray:
    return np.array([pair_key(groups[i], groups[j]) for i, j in edges], dtype=object)


def build_metric_table(group_pair_keys: np.ndarray, domain_masks: dict, domains: Sequence[str], min_edges: int) -> pd.DataFrame:
    rows = []
    for dom in domains:
        if dom not in domain_masks:
            print(f"WARNING: requested domain {dom!r} not present; skipping")
            continue
        dm = domain_masks[dom]
        vc = pd.Series(group_pair_keys[dm]).value_counts()
        for gkey, n in vc.items():
            if int(n) < int(min_edges):
                continue
            a, b = split_pair_key(gkey)
            rows.append({"domain": dom, "group_pair_key": gkey, "group_a": a, "group_b": b, "n_edges": int(n)})
    return pd.DataFrame(rows)


def target_axis_for_pair(gkey: str, targets: pd.DataFrame) -> str:
    ga, gb = split_pair_key(gkey)
    hits = []
    for _, r in targets.iterrows():
        A = set(split_semicolon(r["side_a_groups"]))
        B = set(split_semicolon(r["side_b_groups"]))
        if (ga in A and gb in B) or (gb in A and ga in B):
            hits.append(str(r["interaction_name"]))
    return ";".join(hits)


def compute_null_stats(
    metric_df: pd.DataFrame,
    edges: np.ndarray,
    group_pair_keys: np.ndarray,
    domain_masks: dict,
    obs_edge_norm: np.ndarray,
    obs_edge_raw: np.ndarray,
    cell_norm: np.ndarray,
    cell_raw: np.ndarray,
    tile_indices: Sequence[np.ndarray],
    n_perm: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    masks = []
    for _, r in metric_df.iterrows():
        m = domain_masks[str(r["domain"])] & (group_pair_keys == str(r["group_pair_key"]))
        masks.append(np.where(m)[0])

    out = metric_df.copy()
    obs_norm = np.array([float(obs_edge_norm[idx].mean()) if len(idx) else np.nan for idx in masks])
    obs_raw = np.array([float(obs_edge_raw[idx].mean()) if len(idx) else np.nan for idx in masks])

    edge_mu = float(np.mean(obs_edge_norm))
    edge_sd = float(np.std(obs_edge_norm))
    obs_global_edge_z = np.zeros_like(obs_norm) if edge_sd < 1e-9 else (obs_norm - edge_mu) / edge_sd

    null_norm = np.zeros((len(out), int(n_perm)), dtype=np.float32)
    null_raw = np.zeros((len(out), int(n_perm)), dtype=np.float32)

    for p in range(int(n_perm)):
        if (p + 1) % max(1, int(n_perm) // 10) == 0:
            print(f"  permutation {p+1}/{n_perm}", flush=True)
        sn = shuffle_within_tiles(cell_norm, tile_indices, rng)
        sr = shuffle_within_tiles(cell_raw, tile_indices, rng)
        en = edge_score(edges, sn)
        er = edge_score(edges, sr)
        for k, idx in enumerate(masks):
            null_norm[k, p] = float(en[idx].mean()) if len(idx) else np.nan
            null_raw[k, p] = float(er[idx].mean()) if len(idx) else np.nan

    def fill(prefix: str, obs: np.ndarray, null: np.ndarray):
        mu = np.nanmean(null, axis=1)
        sd = np.nanstd(null, axis=1, ddof=1)
        z = np.where(sd > 1e-9, (obs - mu) / sd, np.nan)
        p = (np.sum(null >= obs[:, None], axis=1) + 1.0) / (null.shape[1] + 1.0)
        out[f"obs_{prefix}_score"] = obs
        out[f"null_{prefix}_mean"] = mu
        out[f"null_{prefix}_sd"] = sd
        out[f"shuffle_{prefix}_z"] = z
        out[f"empirical_p_upper_{prefix}"] = p

    fill("norm", obs_norm, null_norm)
    fill("raw", obs_raw, null_raw)
    out["obs_global_edge_z"] = obs_global_edge_z
    out["n_perm"] = int(n_perm)
    return out


def select_group_order(df: pd.DataFrame) -> List[str]:
    present = sorted(set(df["group_a"].astype(str)).union(set(df["group_b"].astype(str))))
    order = [g for g in DEFAULT_GROUP_ORDER if g in present]
    order += [g for g in present if g not in order]
    return order


def make_matrix(df: pd.DataFrame, model: str, domain: str, groups: Sequence[str], value_col: str) -> np.ndarray:
    mat = pd.DataFrame(np.nan, index=list(groups), columns=list(groups), dtype=float)
    sub = df[(df["model"] == model) & (df["domain"] == domain)]
    for _, r in sub.iterrows():
        a, b = str(r["group_a"]), str(r["group_b"])
        if a in mat.index and b in mat.columns:
            val = float(r[value_col]) if pd.notnull(r[value_col]) else np.nan
            mat.loc[a, b] = val
            mat.loc[b, a] = val
    return mat.to_numpy(dtype=float)


def add_target_boxes(ax, groups: Sequence[str], targets: pd.DataFrame):
    idx = {g: i for i, g in enumerate(groups)}
    for _, r in targets.iterrows():
        A = split_semicolon(r["side_a_groups"])
        B = split_semicolon(r["side_b_groups"])
        for a in A:
            for b in B:
                if a in idx and b in idx:
                    for x, y in [(idx[b], idx[a]), (idx[a], idx[b])]:
                        ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, linewidth=2))


def plot_heatmaps(df: pd.DataFrame, out_root: Path, targets: pd.DataFrame, domains: Sequence[str], value_col: str, vmin: float, vmax: float):
    plot_dir = ensure_dir(out_root / "plots")
    mat_dir = ensure_dir(out_root / "matrix_tables")
    groups = select_group_order(df)
    for model in list(pd.unique(df["model"])):
        for dom in domains:
            mat = make_matrix(df, model, dom, groups, value_col)
            pd.DataFrame(mat, index=groups, columns=groups).to_csv(mat_dir / f"matrix_{model}_{dom}_{value_col}.csv")
            fig, ax = plt.subplots(figsize=(max(7, 0.55 * len(groups) + 2), max(6, 0.55 * len(groups) + 1.5)))
            im = ax.imshow(mat, vmin=vmin, vmax=vmax, aspect="equal")
            ax.set_xticks(np.arange(len(groups)))
            ax.set_yticks(np.arange(len(groups)))
            ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(groups, fontsize=8)
            ax.set_title(f"{model} {dom}: {value_col}")
            add_target_boxes(ax, groups, targets)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.set_ylabel(value_col)
            fig.tight_layout()
            fig.savefig(plot_dir / f"grouped_heatmap_{model}_{dom}_{value_col}.png", dpi=220)
            plt.close(fig)


def run_one_model(label: str, h5ad_path: str, mapping: pd.DataFrame, targets: pd.DataFrame, out_root: Path, args, model_i: int) -> pd.DataFrame:
    print(f"\n=== {label} ===")
    a = ad.read_h5ad(h5ad_path)
    coords = np.asarray(a.obsm[args.spatial_key], dtype=np.float32)
    if coords.shape[1] > 2:
        coords = coords[:, :2]
    cell_types = a.obs[args.celltype_col].astype(str).to_numpy()
    niche = a.obs[args.niche_col].astype(str).to_numpy()
    group_map = dict(zip(mapping["cell_type"].astype(str), mapping["group"].astype(str)))
    missing = sorted(set(cell_types) - set(group_map))
    if missing:
        raise ValueError(f"{label}: missing cell types in mapping CSV: {missing}")
    groups = np.array([group_map[x] for x in cell_types], dtype=object)

    print("  loading X / computing scores")
    X = load_X(a)
    cell_raw, cell_norm = cell_program_scores(X, args.clip_quantile)

    print(f"  building edges radius={args.edge_radius}")
    edges = build_edges(coords, args.edge_radius)
    print(f"  n_edges={len(edges):,}")
    obs_edge_norm = edge_score(edges, cell_norm)
    obs_edge_raw = edge_score(edges, cell_raw)

    group_pair_keys = group_pair_keys_for_edges(edges, groups)
    dmasks = domain_edge_masks(edges, niche)
    metric_df = build_metric_table(group_pair_keys, dmasks, args.domains, args.min_edges)
    print(f"  metric rows={len(metric_df)}")
    tiles = make_tiles(coords, args.tile_size)
    print(f"  n_tiles={len(tiles)}, n_perm={args.n_perm}")

    res = compute_null_stats(
        metric_df, edges, group_pair_keys, dmasks,
        obs_edge_norm, obs_edge_raw, cell_norm, cell_raw, tiles,
        args.n_perm, args.seed + model_i * 1000,
    )
    res.insert(0, "model", label)
    res["target_axis"] = res["group_pair_key"].map(lambda x: target_axis_for_pair(x, targets))
    res["is_target_axis"] = res["target_axis"].astype(str).str.len() > 0
    res["shuffle_norm_sig_z2"] = res["shuffle_norm_z"] >= 2.0
    res["shuffle_norm_sig_p05"] = res["empirical_p_upper_norm"] <= 0.05

    out_dir = ensure_dir(out_root / label / "tables")
    res.to_csv(out_dir / "grouped_crosstalk_shuffle_control.csv", index=False)
    with open(out_dir / "shuffle_control_summary.json", "w") as f:
        json.dump({
            "model": label, "h5ad": h5ad_path, "n_obs": int(a.n_obs), "n_vars": int(a.n_vars),
            "n_edges": int(len(edges)), "n_metric_rows": int(len(metric_df)), "n_perm": int(args.n_perm),
            "edge_radius": float(args.edge_radius), "tile_size": float(args.tile_size), "min_edges": int(args.min_edges),
        }, f, indent=2)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--h5ads", nargs="+", required=True)
    ap.add_argument("--celltype-group-csv", required=True)
    ap.add_argument("--target-interaction-csv", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--celltype-col", default="author_cell_type")
    ap.add_argument("--niche-col", default="niche")
    ap.add_argument("--spatial-key", default="spatial")
    ap.add_argument("--edge-radius", type=float, default=120.0)
    ap.add_argument("--tile-size", type=float, default=400.0)
    ap.add_argument("--domains", nargs="+", default=["global", "niche0", "niche1", "niche2"])
    ap.add_argument("--n-perm", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-edges", type=int, default=50)
    ap.add_argument("--clip-quantile", type=float, default=0.995)
    ap.add_argument("--plot-value-col", default="shuffle_norm_z", choices=["shuffle_norm_z", "obs_global_edge_z", "obs_norm_score"])
    ap.add_argument("--vmin", type=float, default=-3.0)
    ap.add_argument("--vmax", type=float, default=3.0)
    args = ap.parse_args()

    if len(args.labels) != len(args.h5ads):
        raise ValueError("--labels and --h5ads must have same length")
    out_root = ensure_dir(Path(args.out_root))
    mapping = load_mapping(args.celltype_group_csv)
    targets = load_targets(args.target_interaction_csv)

    all_res = []
    for i, (label, h5) in enumerate(zip(args.labels, args.h5ads)):
        all_res.append(run_one_model(label, h5, mapping, targets, out_root, args, i))

    combined = pd.concat(all_res, ignore_index=True)
    combined.to_csv(out_root / "combined_grouped_crosstalk_shuffle_control.csv", index=False)
    target = combined[combined["is_target_axis"]].copy()
    target.to_csv(out_root / "combined_target_axes_shuffle_control.csv", index=False)
    plot_heatmaps(combined, out_root, targets, args.domains, args.plot_value_col, args.vmin, args.vmax)
    with open(out_root / "run_summary.json", "w") as f:
        json.dump({"args": vars(args), "note": "Tile-shuffle null shuffles cell-level interpretable scores within spatial tiles."}, f, indent=2)
    print("\nDone.")
    print(out_root / "combined_grouped_crosstalk_shuffle_control.csv")
    print(out_root / "combined_target_axes_shuffle_control.csv")
    print(out_root / "plots")


if __name__ == "__main__":
    main()


# python test_crosstalk_shuffle_control_3models.py \
#   --labels scgpt c2sscale geneformer \
#   --h5ads \
#     /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/niche_validation_3models_shared/scgpt/layer_4.norm2/r120p0_xm_gmm_k3/adata_with_niche_labels.h5ad \
#     /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/niche_validation_3models_shared/c2sscale/layer_17/r120p0_xm_gmm_k3/adata_with_niche_labels.h5ad \
#     /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/niche_validation_3models_shared/geneformer/layer_4/r120p0_xm_gmm_k3/adata_with_niche_labels.h5ad \
#   --celltype-group-csv /maiziezhou_lab2/yunfei/Projects/interpTFM/resources/celltype_group_mapping_crosstalk_DRAFT.csv \
#   --target-interaction-csv /maiziezhou_lab2/yunfei/Projects/interpTFM/resources/crosstalk_target_interaction_sets.csv \
#   --out-root /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/crosstalk_3models_shuffle_v1 \
#   --celltype-col author_cell_type \
#   --niche-col niche \
#   --spatial-key spatial \
#   --edge-radius 138 \
#   --tile-size 400 \
#   --n-perm 500 \
#   --min-edges 50