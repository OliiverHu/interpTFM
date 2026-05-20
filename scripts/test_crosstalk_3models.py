#!/usr/bin/env python
"""
3-model interpretable spatial crosstalk validation.

This is NOT ligand-receptor CCC. It measures model-derived interpretable spatial
crosstalk: spatial cell-type pair adjacency + elevated interpretable feature signal
on edges between neighboring cells.

Design implemented here:
  A1: niche 0 vs non-0 boundary
  B1: edge-midpoint signed distance to niche-0 boundary
  C1: raw crosstalk density
  C2: normalized/z-scored crosstalk density
  C3: significant-pair-only density
  D2: explicit cell type -> group mapping CSV + target interaction set CSV

Outputs per model:
  tables/
    pair_crosstalk_<domain>.csv
    target_interaction_summary_<domain>.csv
    top_features_by_target_interaction_<domain>.csv
    boundary_density_niche0_vs_non0.csv
    edges_summary.json
  plots/
    boundary_density_<interaction>.png

Combined outputs:
  combined_pair_crosstalk.csv
  combined_target_interaction_summary.csv
  combined_boundary_density.csv
  combined_top_features_by_target_interaction.csv
  combined_edges_summary.csv

Example:
python test_crosstalk_3models.py \
  --labels scgpt c2sscale geneformer \
  --h5ads \
    /.../scgpt/.../adata_with_niche_labels.h5ad \
    /.../c2sscale/.../adata_with_niche_labels.h5ad \
    /.../geneformer/.../adata_with_niche_labels.h5ad \
  --celltype-group-csv /.../resources/celltype_group_mapping_crosstalk_DRAFT.csv \
  --target-interaction-csv /.../resources/crosstalk_target_interaction_sets.csv \
  --out-root /.../runs/crosstalk_3models \
  --celltype-col author_cell_type \
  --niche-col niche \
  --spatial-key spatial \
  --edge-radius 120 \
  --boundary-max-dist 400 \
  --boundary-bin-width 25 \
  --n-perm-adj 0
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree


def parse_none_int(x: str) -> Optional[int]:
    if str(x).upper() == "NONE":
        return None
    return int(x)


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_group_mapping(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"cell_type", "group"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"celltype group CSV missing columns: {missing}")
    df["cell_type"] = df["cell_type"].astype(str)
    df["group"] = df["group"].astype(str)
    return df


def load_target_sets(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"interaction_name", "side_a_groups", "side_b_groups"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"target interaction CSV missing columns: {missing}")
    for c in ["interaction_name", "side_a_groups", "side_b_groups"]:
        df[c] = df[c].astype(str)
    return df


def split_semicolon(x: str) -> List[str]:
    return [z.strip() for z in str(x).split(";") if z.strip()]


def unordered_pair(a: str, b: str) -> str:
    a = str(a)
    b = str(b)
    return f"{a}||{b}" if a <= b else f"{b}||{a}"


def pair_cols_from_key(key: str) -> Tuple[str, str]:
    a, b = str(key).split("||", 1)
    return a, b


def load_dense_X(a: ad.AnnData) -> np.ndarray:
    X = a.X
    if sparse.issparse(X):
        X = X.toarray()
    else:
        X = np.asarray(X)
    X = X.astype(np.float32, copy=False)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def robust_cell_scores(X: np.ndarray, clip_quantile: float = 0.995) -> Tuple[np.ndarray, np.ndarray]:
    """
    C1 raw cell score:
      mean nonnegative interpretable feature activation per cell.

    C2 normalized cell score:
      per-feature robust clipped z-score, positive part, then mean per cell.
      This gives a scale-normalized "active interpretable program load" per cell.
    """
    X_nonneg = np.maximum(X, 0.0)
    raw = X_nonneg.mean(axis=1).astype(np.float32)

    Xc = X_nonneg.copy()
    if 0.0 < clip_quantile < 1.0:
        hi = np.quantile(Xc, clip_quantile, axis=0)
        hi = np.asarray(hi, dtype=np.float32)
        Xc = np.minimum(Xc, hi[None, :])

    mu = Xc.mean(axis=0)
    sd = Xc.std(axis=0)
    sd[sd < 1e-6] = 1.0
    Z = (Xc - mu[None, :]) / sd[None, :]
    zpos = np.maximum(Z, 0.0)
    norm = zpos.mean(axis=1).astype(np.float32)
    return raw, norm


def build_edges(coords: np.ndarray, radius: float) -> np.ndarray:
    tree = cKDTree(coords)
    try:
        pairs = tree.query_pairs(float(radius), output_type="ndarray")
    except TypeError:
        pairs = np.array(list(tree.query_pairs(float(radius))), dtype=np.int64)
    if pairs.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    pairs = np.asarray(pairs, dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        pairs = pairs.reshape((-1, 2))
    return pairs


def make_domain_masks(niche: np.ndarray) -> Dict[str, np.ndarray]:
    out = {"global": np.ones(len(niche), dtype=bool)}
    vals = sorted(pd.unique(pd.Series(niche).dropna()))
    for v in vals:
        out[f"niche{v}"] = niche == v
    return out


def edge_mask_for_cells(edges: np.ndarray, cell_mask: np.ndarray) -> np.ndarray:
    return cell_mask[edges[:, 0]] & cell_mask[edges[:, 1]]


def make_tiles(coords: np.ndarray, tile_size: float) -> np.ndarray:
    xy = np.floor(coords / float(tile_size)).astype(np.int64)
    # stable compact-ish string tile id
    return np.array([f"{x}_{y}" for x, y in xy], dtype=object)


def shuffle_within_tiles(values: np.ndarray, tiles: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = values.copy()
    for t in pd.unique(tiles):
        idx = np.where(tiles == t)[0]
        if len(idx) > 1:
            out[idx] = rng.permutation(out[idx])
    return out


def count_pairs_for_edges(edges: np.ndarray, labels: np.ndarray) -> Counter:
    a = labels[edges[:, 0]]
    b = labels[edges[:, 1]]
    keys = [unordered_pair(x, y) for x, y in zip(a, b)]
    return Counter(keys)


def adjacency_null_z(
    edges: np.ndarray,
    labels: np.ndarray,
    coords: np.ndarray,
    observed_keys: Sequence[str],
    tile_size: float,
    n_perm: int,
    seed: int,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    if n_perm <= 0:
        return {k: np.nan for k in observed_keys}, {k: np.nan for k in observed_keys}

    rng = np.random.default_rng(seed)
    tiles = make_tiles(coords, tile_size)
    vals = {k: [] for k in observed_keys}

    for p in range(n_perm):
        shuf = shuffle_within_tiles(labels, tiles, rng)
        c = count_pairs_for_edges(edges, shuf)
        for k in observed_keys:
            vals[k].append(float(c.get(k, 0)))

    z = {}
    mean = {}
    for k, arr in vals.items():
        x = np.asarray(arr, dtype=float)
        mu = float(x.mean())
        sd = float(x.std(ddof=1)) if len(x) > 1 else 0.0
        mean[k] = mu
        z[k] = np.nan if sd < 1e-9 else (np.nan)  # filled by caller if obs provided
    return mean, vals


def summarize_pairs(
    edges: np.ndarray,
    cell_types: np.ndarray,
    groups: np.ndarray,
    edge_raw: np.ndarray,
    edge_norm_z: np.ndarray,
    coords: np.ndarray,
    tile_size: float,
    n_perm_adj: int,
    seed: int,
    min_pair_edges: int,
    adj_z_min: float,
    pair_norm_z_min: float,
) -> pd.DataFrame:
    if len(edges) == 0:
        return pd.DataFrame()

    ct_a = cell_types[edges[:, 0]]
    ct_b = cell_types[edges[:, 1]]
    gp_a = groups[edges[:, 0]]
    gp_b = groups[edges[:, 1]]
    pair_key = np.array([unordered_pair(a, b) for a, b in zip(ct_a, ct_b)], dtype=object)
    group_pair_key = np.array([unordered_pair(a, b) for a, b in zip(gp_a, gp_b)], dtype=object)

    df = pd.DataFrame(
        {
            "pair_key": pair_key,
            "group_pair_key": group_pair_key,
            "edge_raw": edge_raw,
            "edge_norm_z": edge_norm_z,
        }
    )

    obs = (
        df.groupby(["pair_key", "group_pair_key"], as_index=False)
        .agg(
            n_edges=("edge_raw", "size"),
            mean_raw_score=("edge_raw", "mean"),
            median_raw_score=("edge_raw", "median"),
            mean_norm_z_score=("edge_norm_z", "mean"),
            median_norm_z_score=("edge_norm_z", "median"),
        )
    )

    keys = list(obs["pair_key"].astype(str))
    obs_counts = dict(zip(obs["pair_key"].astype(str), obs["n_edges"].astype(float)))

    if n_perm_adj > 0:
        null_mean, null_vals = adjacency_null_z(
            edges=edges,
            labels=cell_types,
            coords=coords,
            observed_keys=keys,
            tile_size=tile_size,
            n_perm=n_perm_adj,
            seed=seed,
        )
        adj_mean = []
        adj_z = []
        for k in keys:
            arr = np.asarray(null_vals[k], dtype=float)
            mu = float(arr.mean())
            sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
            adj_mean.append(mu)
            adj_z.append(np.nan if sd < 1e-9 else (obs_counts[k] - mu) / sd)
        obs["adjacency_null_mean"] = adj_mean
        obs["adjacency_z"] = adj_z
    else:
        obs["adjacency_null_mean"] = np.nan
        obs["adjacency_z"] = np.nan

    pair_a = []
    pair_b = []
    group_a = []
    group_b = []
    for pk, gpk in zip(obs["pair_key"], obs["group_pair_key"]):
        a, b = pair_cols_from_key(pk)
        ga, gb = pair_cols_from_key(gpk)
        pair_a.append(a)
        pair_b.append(b)
        group_a.append(ga)
        group_b.append(gb)
    obs.insert(1, "cell_type_a", pair_a)
    obs.insert(2, "cell_type_b", pair_b)
    obs.insert(4, "group_a", group_a)
    obs.insert(5, "group_b", group_b)

    if n_perm_adj > 0:
        adj_ok = obs["adjacency_z"].fillna(-np.inf) >= float(adj_z_min)
    else:
        adj_ok = True

    obs["flag_significant_pair"] = (
        (obs["n_edges"] >= int(min_pair_edges))
        & adj_ok
        & (obs["mean_norm_z_score"] >= float(pair_norm_z_min))
    )
    obs = obs.sort_values(["flag_significant_pair", "mean_norm_z_score", "n_edges"], ascending=[False, False, False])
    return obs


def edge_scores_for_edges(
    edges: np.ndarray,
    cell_raw: np.ndarray,
    cell_norm: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    raw = 0.5 * (cell_raw[edges[:, 0]] + cell_raw[edges[:, 1]])
    norm = 0.5 * (cell_norm[edges[:, 0]] + cell_norm[edges[:, 1]])
    sd = float(np.std(norm))
    if sd < 1e-9:
        norm_z = np.zeros_like(norm, dtype=np.float32)
    else:
        norm_z = ((norm - float(np.mean(norm))) / sd).astype(np.float32)
    return raw.astype(np.float32), norm_z.astype(np.float32)


def target_edge_mask(edges: np.ndarray, groups: np.ndarray, side_a: Sequence[str], side_b: Sequence[str]) -> np.ndarray:
    A = set(side_a)
    B = set(side_b)
    gi = groups[edges[:, 0]]
    gj = groups[edges[:, 1]]
    return (np.isin(gi, list(A)) & np.isin(gj, list(B))) | (np.isin(gi, list(B)) & np.isin(gj, list(A)))


def target_summary(
    edges: np.ndarray,
    groups: np.ndarray,
    edge_raw: np.ndarray,
    edge_norm_z: np.ndarray,
    target_sets: pd.DataFrame,
    domain_name: str,
) -> pd.DataFrame:
    rows = []
    for _, r in target_sets.iterrows():
        name = str(r["interaction_name"])
        A = split_semicolon(r["side_a_groups"])
        B = split_semicolon(r["side_b_groups"])
        m = target_edge_mask(edges, groups, A, B)
        n = int(m.sum())
        rows.append(
            {
                "domain": domain_name,
                "interaction_name": name,
                "side_a_groups": ";".join(A),
                "side_b_groups": ";".join(B),
                "n_edges": n,
                "mean_raw_score": float(np.mean(edge_raw[m])) if n else np.nan,
                "median_raw_score": float(np.median(edge_raw[m])) if n else np.nan,
                "mean_norm_z_score": float(np.mean(edge_norm_z[m])) if n else np.nan,
                "median_norm_z_score": float(np.median(edge_norm_z[m])) if n else np.nan,
            }
        )
    return pd.DataFrame(rows)


def top_features_for_targets(
    X: np.ndarray,
    var_names: Sequence[str],
    var_df: pd.DataFrame,
    edges: np.ndarray,
    groups: np.ndarray,
    target_sets: pd.DataFrame,
    domain_name: str,
    top_n: int,
    max_edges_for_features: Optional[int],
    seed: int,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    for _, r in target_sets.iterrows():
        name = str(r["interaction_name"])
        A = split_semicolon(r["side_a_groups"])
        B = split_semicolon(r["side_b_groups"])
        m = target_edge_mask(edges, groups, A, B)
        idx = np.where(m)[0]
        if len(idx) == 0:
            continue
        if max_edges_for_features is not None and len(idx) > max_edges_for_features:
            idx = rng.choice(idx, size=max_edges_for_features, replace=False)

        ei = edges[idx, 0]
        ej = edges[idx, 1]
        feat_mean = 0.5 * (X[ei].mean(axis=0) + X[ej].mean(axis=0))
        feat_mean = np.asarray(feat_mean).ravel()
        order = np.argsort(-feat_mean)[: int(top_n)]

        for rank, j in enumerate(order, start=1):
            row = {
                "domain": domain_name,
                "interaction_name": name,
                "rank": rank,
                "feature_index": int(j),
                "feature_name": str(var_names[j]),
                "mean_edge_feature_signal": float(feat_mean[j]),
                "n_edges_used": int(len(idx)),
            }
            # Carry a few useful metadata columns if present.
            for c in ["concept_name", "concept_id", "concept_key_id", "concept_key_name_id", "term", "feature_id", "broad_categories", "primary_category"]:
                if c in var_df.columns:
                    row[c] = var_df.iloc[j][c]
            rows.append(row)
    return pd.DataFrame(rows)


def signed_distance_to_niche0_boundary(points: np.ndarray, coords: np.ndarray, niche: np.ndarray, niche0_value: str = "0") -> np.ndarray:
    nstr = pd.Series(niche).astype(str).to_numpy()
    inside = nstr == str(niche0_value)
    if inside.sum() == 0 or (~inside).sum() == 0:
        return np.full(points.shape[0], np.nan, dtype=np.float32)
    tree_in = cKDTree(coords[inside])
    tree_out = cKDTree(coords[~inside])
    d_in, _ = tree_in.query(points, k=1)
    d_out, _ = tree_out.query(points, k=1)
    # Negative means closer to niche0 than non0, i.e. inside niche0 side.
    return (d_in - d_out).astype(np.float32)


def boundary_density(
    edges: np.ndarray,
    coords: np.ndarray,
    niche: np.ndarray,
    groups: np.ndarray,
    cell_types: np.ndarray,
    edge_raw: np.ndarray,
    edge_norm_z: np.ndarray,
    pair_flags_global: Dict[str, bool],
    target_sets: pd.DataFrame,
    max_dist: float,
    bin_width: float,
    niche0_value: str,
) -> pd.DataFrame:
    mid = 0.5 * (coords[edges[:, 0]] + coords[edges[:, 1]])
    sd = signed_distance_to_niche0_boundary(mid, coords, niche, niche0_value=niche0_value)
    keep = np.isfinite(sd) & (sd >= -float(max_dist)) & (sd <= float(max_dist))

    bins = np.arange(-float(max_dist), float(max_dist) + float(bin_width), float(bin_width))
    if len(bins) < 2:
        raise ValueError("bad boundary bins")
    bin_id = np.digitize(sd, bins) - 1
    centers = 0.5 * (bins[:-1] + bins[1:])

    pair_key = np.array([unordered_pair(a, b) for a, b in zip(cell_types[edges[:, 0]], cell_types[edges[:, 1]])], dtype=object)
    sig_edge = np.array([bool(pair_flags_global.get(k, False)) for k in pair_key], dtype=bool)

    masks = [("all_edges", keep)]
    for _, r in target_sets.iterrows():
        name = str(r["interaction_name"])
        A = split_semicolon(r["side_a_groups"])
        B = split_semicolon(r["side_b_groups"])
        masks.append((name, keep & target_edge_mask(edges, groups, A, B)))

    rows = []
    for name, m0 in masks:
        for sig_only in [False, True]:
            m = m0 & (sig_edge if sig_only else True)
            for b in range(len(centers)):
                mb = m & (bin_id == b)
                n = int(mb.sum())
                rows.append(
                    {
                        "interaction_name": name,
                        "significant_pairs_only": bool(sig_only),
                        "signed_distance_bin_left": float(bins[b]),
                        "signed_distance_bin_right": float(bins[b + 1]),
                        "signed_distance_bin_center": float(centers[b]),
                        "n_edges": n,
                        "mean_raw_score": float(np.mean(edge_raw[mb])) if n else np.nan,
                        "mean_norm_z_score": float(np.mean(edge_norm_z[mb])) if n else np.nan,
                        "median_raw_score": float(np.median(edge_raw[mb])) if n else np.nan,
                        "median_norm_z_score": float(np.median(edge_norm_z[mb])) if n else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def plot_boundary_density(df: pd.DataFrame, out_dir: Path, model: str) -> None:
    plot_dir = ensure_dir(out_dir / "plots")
    for interaction in sorted(df["interaction_name"].dropna().unique()):
        sub = df[df["interaction_name"] == interaction].copy()
        if sub.empty:
            continue

        for ycol, label in [("mean_raw_score", "raw"), ("mean_norm_z_score", "norm_z")]:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for sig_only in [False, True]:
                ss = sub[sub["significant_pairs_only"] == sig_only].sort_values("signed_distance_bin_center")
                if ss["n_edges"].sum() == 0:
                    continue
                lab = "significant pairs only" if sig_only else "all matching edges"
                ax.plot(ss["signed_distance_bin_center"], ss[ycol], marker="o", markersize=2, linewidth=1, label=lab)
            ax.axvline(0, linestyle="--", linewidth=1)
            ax.set_xlabel("Signed distance to niche 0 boundary\n(negative = niche 0 side; positive = non-0 side)")
            ax.set_ylabel(ycol)
            ax.set_title(f"{model}: {interaction} boundary density ({label})")
            ax.legend(frameon=False)
            fig.tight_layout()
            safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in interaction)
            fig.savefig(plot_dir / f"boundary_density_{safe}_{label}.png", dpi=200)
            plt.close(fig)


def run_one_model(
    label: str,
    h5ad_path: str,
    mapping_df: pd.DataFrame,
    target_sets: pd.DataFrame,
    out_root: Path,
    celltype_col: str,
    niche_col: str,
    spatial_key: str,
    edge_radius: float,
    tile_size: float,
    n_perm_adj: int,
    seed: int,
    min_pair_edges: int,
    adj_z_min: float,
    pair_norm_z_min: float,
    boundary_max_dist: float,
    boundary_bin_width: float,
    niche0_value: str,
    top_n_features: int,
    max_edges_for_features: Optional[int],
    clip_quantile: float,
) -> Dict[str, object]:
    out_dir = ensure_dir(out_root / label)
    tables_dir = ensure_dir(out_dir / "tables")

    print(f"\n=== {label} ===")
    print(f"Loading {h5ad_path}")
    a = ad.read_h5ad(h5ad_path)

    if celltype_col not in a.obs.columns:
        raise KeyError(f"{celltype_col!r} not in adata.obs. Available: {list(a.obs.columns[:80])}")
    if niche_col not in a.obs.columns:
        raise KeyError(f"{niche_col!r} not in adata.obs. Available: {list(a.obs.columns[:80])}")
    if spatial_key not in a.obsm:
        raise KeyError(f"{spatial_key!r} not in adata.obsm. Available: {list(a.obsm.keys())}")

    coords = np.asarray(a.obsm[spatial_key], dtype=np.float32)
    if coords.shape[1] > 2:
        coords = coords[:, :2]
    cell_types = a.obs[celltype_col].astype(str).to_numpy()
    niche = a.obs[niche_col].astype(str).to_numpy()

    group_map = dict(zip(mapping_df["cell_type"].astype(str), mapping_df["group"].astype(str)))
    missing_ct = sorted(set(cell_types) - set(group_map.keys()))
    if missing_ct:
        raise ValueError(
            f"{label}: cell types missing from mapping CSV: {missing_ct}. "
            "Add them to celltype_group_mapping CSV."
        )
    groups = np.array([group_map[x] for x in cell_types], dtype=object)

    X = load_dense_X(a)
    var_names = list(map(str, a.var_names))
    var_df = a.var.copy()
    print(f"n_obs={a.n_obs}, n_vars={a.n_vars}, coords={coords.shape}, X={X.shape}")

    cell_raw, cell_norm = robust_cell_scores(X, clip_quantile=clip_quantile)

    print(f"Building radius edges, r={edge_radius}")
    edges = build_edges(coords, edge_radius)
    print(f"n_edges={len(edges):,}")
    edge_raw_global, edge_norm_z_global = edge_scores_for_edges(edges, cell_raw, cell_norm)

    domains = make_domain_masks(niche)

    pair_tables = []
    target_tables = []
    feature_tables = []
    pair_flags_global: Dict[str, bool] = {}

    for domain_name, cmask in domains.items():
        emask = edge_mask_for_cells(edges, cmask)
        ed = edges[emask]
        eraw = edge_raw_global[emask]
        enz = edge_norm_z_global[emask]
        print(f"Domain {domain_name}: n_cells={int(cmask.sum()):,}, n_edges={len(ed):,}")

        pair_df = summarize_pairs(
            edges=ed,
            cell_types=cell_types,
            groups=groups,
            edge_raw=eraw,
            edge_norm_z=enz,
            coords=coords,
            tile_size=tile_size,
            n_perm_adj=n_perm_adj,
            seed=seed,
            min_pair_edges=min_pair_edges,
            adj_z_min=adj_z_min,
            pair_norm_z_min=pair_norm_z_min,
        )
        if len(pair_df):
            pair_df.insert(0, "model", label)
            pair_df.insert(1, "domain", domain_name)
            pair_df.to_csv(tables_dir / f"pair_crosstalk_{domain_name}.csv", index=False)
            pair_tables.append(pair_df)

            if domain_name == "global":
                pair_flags_global = dict(zip(pair_df["pair_key"].astype(str), pair_df["flag_significant_pair"].astype(bool)))

        tdf = target_summary(ed, groups, eraw, enz, target_sets, domain_name)
        tdf.insert(0, "model", label)
        tdf.to_csv(tables_dir / f"target_interaction_summary_{domain_name}.csv", index=False)
        target_tables.append(tdf)

        fdf = top_features_for_targets(
            X=X,
            var_names=var_names,
            var_df=var_df,
            edges=ed,
            groups=groups,
            target_sets=target_sets,
            domain_name=domain_name,
            top_n=top_n_features,
            max_edges_for_features=max_edges_for_features,
            seed=seed,
        )
        if len(fdf):
            fdf.insert(0, "model", label)
            fdf.to_csv(tables_dir / f"top_features_by_target_interaction_{domain_name}.csv", index=False)
            feature_tables.append(fdf)

    bdf = boundary_density(
        edges=edges,
        coords=coords,
        niche=niche,
        groups=groups,
        cell_types=cell_types,
        edge_raw=edge_raw_global,
        edge_norm_z=edge_norm_z_global,
        pair_flags_global=pair_flags_global,
        target_sets=target_sets,
        max_dist=boundary_max_dist,
        bin_width=boundary_bin_width,
        niche0_value=niche0_value,
    )
    bdf.insert(0, "model", label)
    bdf.to_csv(tables_dir / "boundary_density_niche0_vs_non0.csv", index=False)
    plot_boundary_density(bdf, out_dir, label)

    summary = {
        "model": label,
        "h5ad": str(h5ad_path),
        "n_obs": int(a.n_obs),
        "n_vars": int(a.n_vars),
        "n_edges": int(len(edges)),
        "edge_radius": float(edge_radius),
        "celltype_col": celltype_col,
        "niche_col": niche_col,
        "spatial_key": spatial_key,
        "n_cell_types": int(len(set(cell_types))),
        "n_groups": int(len(set(groups))),
        "domains": {k: {"n_cells": int(v.sum()), "n_edges": int(edge_mask_for_cells(edges, v).sum())} for k, v in domains.items()},
        "n_perm_adj": int(n_perm_adj),
        "min_pair_edges": int(min_pair_edges),
        "adj_z_min": float(adj_z_min),
        "pair_norm_z_min": float(pair_norm_z_min),
    }
    with open(tables_dir / "edges_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return {
        "summary": summary,
        "pairs": pd.concat(pair_tables, ignore_index=True) if pair_tables else pd.DataFrame(),
        "targets": pd.concat(target_tables, ignore_index=True) if target_tables else pd.DataFrame(),
        "features": pd.concat(feature_tables, ignore_index=True) if feature_tables else pd.DataFrame(),
        "boundary": bdf,
    }


def main() -> None:
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
    ap.add_argument("--n-perm-adj", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--min-pair-edges", type=int, default=50)
    ap.add_argument("--adj-z-min", type=float, default=2.0)
    ap.add_argument("--pair-norm-z-min", type=float, default=0.25)

    ap.add_argument("--boundary-max-dist", type=float, default=400.0)
    ap.add_argument("--boundary-bin-width", type=float, default=25.0)
    ap.add_argument("--niche0-value", default="0")

    ap.add_argument("--top-n-features", type=int, default=25)
    ap.add_argument("--max-edges-for-features", default="50000", help="Integer or NONE.")
    ap.add_argument("--clip-quantile", type=float, default=0.995)

    args = ap.parse_args()

    if len(args.labels) != len(args.h5ads):
        raise ValueError("--labels and --h5ads must have same length")

    out_root = ensure_dir(Path(args.out_root))
    mapping_df = load_group_mapping(args.celltype_group_csv)
    target_sets = load_target_sets(args.target_interaction_csv)
    max_edges_for_features = parse_none_int(args.max_edges_for_features)

    all_pairs = []
    all_targets = []
    all_features = []
    all_boundary = []
    summaries = []

    for label, h5 in zip(args.labels, args.h5ads):
        res = run_one_model(
            label=label,
            h5ad_path=h5,
            mapping_df=mapping_df,
            target_sets=target_sets,
            out_root=out_root,
            celltype_col=args.celltype_col,
            niche_col=args.niche_col,
            spatial_key=args.spatial_key,
            edge_radius=args.edge_radius,
            tile_size=args.tile_size,
            n_perm_adj=args.n_perm_adj,
            seed=args.seed,
            min_pair_edges=args.min_pair_edges,
            adj_z_min=args.adj_z_min,
            pair_norm_z_min=args.pair_norm_z_min,
            boundary_max_dist=args.boundary_max_dist,
            boundary_bin_width=args.boundary_bin_width,
            niche0_value=args.niche0_value,
            top_n_features=args.top_n_features,
            max_edges_for_features=max_edges_for_features,
            clip_quantile=args.clip_quantile,
        )
        summaries.append(res["summary"])
        if len(res["pairs"]):
            all_pairs.append(res["pairs"])
        if len(res["targets"]):
            all_targets.append(res["targets"])
        if len(res["features"]):
            all_features.append(res["features"])
        if len(res["boundary"]):
            all_boundary.append(res["boundary"])

    if all_pairs:
        pd.concat(all_pairs, ignore_index=True).to_csv(out_root / "combined_pair_crosstalk.csv", index=False)
    if all_targets:
        pd.concat(all_targets, ignore_index=True).to_csv(out_root / "combined_target_interaction_summary.csv", index=False)
    if all_features:
        pd.concat(all_features, ignore_index=True).to_csv(out_root / "combined_top_features_by_target_interaction.csv", index=False)
    if all_boundary:
        pd.concat(all_boundary, ignore_index=True).to_csv(out_root / "combined_boundary_density.csv", index=False)

    pd.DataFrame(summaries).to_csv(out_root / "combined_edges_summary.csv", index=False)
    with open(out_root / "run_summary.json", "w") as f:
        json.dump(
            {
                "args": vars(args),
                "summaries": summaries,
                "note": "This is interpretable spatial crosstalk, not ligand-receptor CCC.",
            },
            f,
            indent=2,
        )

    print("\nDone. Wrote combined outputs to:")
    print(out_root)


if __name__ == "__main__":
    main()
