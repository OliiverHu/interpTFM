#!/usr/bin/env python
"""
Integrated immune-infiltration follow-up analyses for interpretable spatial crosstalk.

This script runs six immune-infiltration analyses and saves each in its own subfolder:

01_tumor_contact_immune_gradient
  For selected immune groups, relate distance to nearest tumor cell to cell-level
  interpretable program load and tumor-edge crosstalk.

02_tumor_boundary_asymmetry
  Summarize target-axis crosstalk by signed distance regions around the niche0 boundary:
  niche0_core, niche0_inner_edge, boundary, outside_inner_edge, outside_far.

03_immune_hot_cold_tumors
  Classify tumor cells by local immune-neighbor fraction and compare tumor-cell
  interpretable program load.

04_cd8_macrophage_balance
  Compare T CD8 memory-tumor versus macrophage-tumor crosstalk around tumor cells.

05_tcell_stromal_endothelial_state
  Compare T CD4 memory cells near fibroblast/endothelial cells versus far from them.

06_cross_model_consensus
  Rank group-pair/domain shuffle-controlled crosstalk within each model and compute
  cross-model consensus.

Inputs:
  - 3 model h5ads with adata_with_niche_labels.h5ad
  - cell type -> group CSV
  - target interaction CSV
  - shuffle-control grouped CSV from test_crosstalk_shuffle_control_3models.py
  - boundary density CSV from test_crosstalk_3models.py

Recommended:
python test_immune_infiltration_followups_3models.py \
  --labels scgpt c2sscale geneformer \
  --h5ads \
    /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/niche_validation_3models_shared/scgpt/layer_4.norm2/r120p0_xm_gmm_k3/adata_with_niche_labels.h5ad \
    /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/niche_validation_3models_shared/c2sscale/layer_17/r120p0_xm_gmm_k3/adata_with_niche_labels.h5ad \
    /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/niche_validation_3models_shared/geneformer/layer_4/r120p0_xm_gmm_k3/adata_with_niche_labels.h5ad \
  --celltype-group-csv /maiziezhou_lab2/yunfei/Projects/interpTFM/resources/celltype_group_mapping_crosstalk_DRAFT.csv \
  --target-interaction-csv /maiziezhou_lab2/yunfei/Projects/interpTFM/resources/crosstalk_target_interaction_sets.csv \
  --shuffle-control-csv /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/crosstalk_3models_shuffle_v1/combined_grouped_crosstalk_shuffle_control.csv \
  --boundary-density-csv /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/crosstalk_3models_v1/combined_boundary_density.csv \
  --out-root /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/immune_infiltration_followups_3models_v1 \
  --celltype-col author_cell_type \
  --niche-col niche \
  --spatial-key spatial \
  --edge-radius 138 \
  --tumor-groups tumor \
  --immune-groups macrophage t_cd8_memory t_cd4_memory treg nk dendritic monocyte \
  --stromal-endothelial-groups fibroblast endothelial
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree


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


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def pair_key(a: str, b: str) -> str:
    a, b = str(a), str(b)
    return f"{a}||{b}" if a <= b else f"{b}||{a}"


def split_pair_key(k: str) -> Tuple[str, str]:
    a, b = str(k).split("||", 1)
    return a, b


def split_semicolon(x: str) -> List[str]:
    return [z.strip() for z in str(x).split(";") if z.strip()]


def load_mapping(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not {"cell_type", "group"}.issubset(df.columns):
        raise ValueError("mapping CSV requires columns: cell_type, group")
    df["cell_type"] = df["cell_type"].astype(str)
    df["group"] = df["group"].astype(str)
    return df


def load_targets(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not {"interaction_name", "side_a_groups", "side_b_groups"}.issubset(df.columns):
        raise ValueError("target CSV requires columns: interaction_name, side_a_groups, side_b_groups")
    return df


def load_X(a: ad.AnnData) -> np.ndarray:
    X = a.X
    if sparse.issparse(X):
        X = X.toarray()
    else:
        X = np.asarray(X)
    X = X.astype(np.float32, copy=False)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def cell_program_scores(X: np.ndarray, clip_quantile: float = 0.995) -> Tuple[np.ndarray, np.ndarray]:
    """
    raw = mean nonnegative feature activation per cell
    norm = robust per-feature zscore positive part averaged per cell
    """
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


def edge_score(edges: np.ndarray, cell_score: np.ndarray) -> np.ndarray:
    return (0.5 * (cell_score[edges[:, 0]] + cell_score[edges[:, 1]])).astype(np.float32)


def make_groups(cell_types: np.ndarray, mapping_df: pd.DataFrame, label: str) -> np.ndarray:
    mp = dict(zip(mapping_df["cell_type"].astype(str), mapping_df["group"].astype(str)))
    missing = sorted(set(map(str, cell_types)) - set(mp))
    if missing:
        raise ValueError(f"{label}: cell types missing from mapping: {missing}")
    return np.array([mp[str(x)] for x in cell_types], dtype=object)


def signed_distance_to_niche0_boundary(points: np.ndarray, coords: np.ndarray, niche: np.ndarray, niche0_value: str) -> np.ndarray:
    nstr = pd.Series(niche).astype(str).to_numpy()
    inside = nstr == str(niche0_value)
    if inside.sum() == 0 or (~inside).sum() == 0:
        return np.full(points.shape[0], np.nan, dtype=np.float32)
    tree_in = cKDTree(coords[inside])
    tree_out = cKDTree(coords[~inside])
    d_in, _ = tree_in.query(points, k=1)
    d_out, _ = tree_out.query(points, k=1)
    # Negative = niche0 side.
    return (d_in - d_out).astype(np.float32)


def region_from_signed_distance(x: float, boundary_width: float = 50.0, far_width: float = 150.0) -> str:
    if not np.isfinite(x):
        return "nan"
    if -boundary_width <= x <= boundary_width:
        return "boundary"
    if x < -far_width:
        return "niche0_core"
    if x < 0:
        return "niche0_inner_edge"
    if x > far_width:
        return "outside_far"
    return "outside_inner_edge"


def quantile_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.array([])
    qs = np.linspace(0, 1, int(n_bins) + 1)
    bins = np.unique(np.quantile(vals, qs))
    if len(bins) < 2:
        eps = 1e-6
        return np.array([vals.min() - eps, vals.max() + eps])
    bins[0] -= 1e-6
    bins[-1] += 1e-6
    return bins


def plot_line(df: pd.DataFrame, x: str, y: str, hue: str, title: str, out_path: Path, xlabel: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, sub in df.groupby(hue):
        sub = sub.sort_values(x)
        ax.plot(sub[x], sub[y], marker="o", linewidth=1.5, markersize=3, label=str(name))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_bar(df: pd.DataFrame, x: str, y: str, hue: Optional[str], title: str, out_path: Path, xlabel: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(max(7, 0.45 * len(df)), 4.5))
    if hue is None:
        ax.bar(np.arange(len(df)), df[y].to_numpy())
        ax.set_xticks(np.arange(len(df)))
        ax.set_xticklabels(df[x].astype(str), rotation=45, ha="right", fontsize=8)
    else:
        piv = df.pivot_table(index=x, columns=hue, values=y, aggfunc="mean")
        piv.plot(kind="bar", ax=ax)
        ax.legend(frameon=False, fontsize=8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def model_data(
    label: str,
    h5ad_path: str,
    mapping_df: pd.DataFrame,
    celltype_col: str,
    niche_col: str,
    spatial_key: str,
    edge_radius: float,
    clip_quantile: float,
) -> Dict[str, object]:
    print(f"\nLoading {label}: {h5ad_path}", flush=True)
    a = ad.read_h5ad(h5ad_path)
    if celltype_col not in a.obs:
        raise KeyError(f"{label}: {celltype_col} missing from obs")
    if niche_col not in a.obs:
        raise KeyError(f"{label}: {niche_col} missing from obs")
    if spatial_key not in a.obsm:
        raise KeyError(f"{label}: {spatial_key} missing from obsm")

    coords = np.asarray(a.obsm[spatial_key], dtype=np.float32)
    if coords.shape[1] > 2:
        coords = coords[:, :2]
    cell_types = a.obs[celltype_col].astype(str).to_numpy()
    niche = a.obs[niche_col].astype(str).to_numpy()
    groups = make_groups(cell_types, mapping_df, label)

    X = load_X(a)
    raw, norm = cell_program_scores(X, clip_quantile=clip_quantile)
    edges = build_edges(coords, edge_radius)
    e_raw = edge_score(edges, raw)
    e_norm = edge_score(edges, norm)
    if np.std(e_norm) > 1e-9:
        e_norm_z = ((e_norm - np.mean(e_norm)) / np.std(e_norm)).astype(np.float32)
    else:
        e_norm_z = np.zeros_like(e_norm, dtype=np.float32)

    print(f"  n_obs={a.n_obs:,}, n_vars={a.n_vars:,}, n_edges={len(edges):,}", flush=True)
    return {
        "adata": a,
        "coords": coords,
        "cell_types": cell_types,
        "niche": niche,
        "groups": groups,
        "X": X,
        "cell_raw": raw,
        "cell_norm": norm,
        "edges": edges,
        "edge_raw": e_raw,
        "edge_norm": e_norm,
        "edge_norm_z": e_norm_z,
    }


# 01
def analysis_tumor_contact_immune_gradient(
    label: str,
    data: Dict[str, object],
    out_dir: Path,
    tumor_groups: Sequence[str],
    immune_groups: Sequence[str],
    distance_bins: int,
) -> pd.DataFrame:
    coords = data["coords"]
    groups = data["groups"]
    cell_norm = data["cell_norm"]
    edges = data["edges"]
    edge_norm_z = data["edge_norm_z"]

    tumor_mask = np.isin(groups, list(tumor_groups))
    if tumor_mask.sum() == 0:
        return pd.DataFrame()
    tree_tumor = cKDTree(coords[tumor_mask])
    dist_to_tumor, _ = tree_tumor.query(coords, k=1)

    rows = []
    for ig in immune_groups:
        m = groups == ig
        if m.sum() == 0:
            continue
        bins = quantile_bins(dist_to_tumor[m], distance_bins)
        if len(bins) < 2:
            continue
        b = np.digitize(dist_to_tumor[m], bins) - 1
        idx_all = np.where(m)[0]
        for bi in range(len(bins) - 1):
            idx = idx_all[b == bi]
            if len(idx) == 0:
                continue
            rows.append({
                "model": label,
                "immune_group": ig,
                "distance_bin": bi,
                "distance_left": float(bins[bi]),
                "distance_right": float(bins[bi + 1]),
                "distance_mid": float(0.5 * (bins[bi] + bins[bi + 1])),
                "n_cells": int(len(idx)),
                "mean_dist_to_tumor": float(np.mean(dist_to_tumor[idx])),
                "mean_cell_norm_score": float(np.mean(cell_norm[idx])),
            })

    # Edge-level tumor contact crosstalk by immune group.
    gi = groups[edges[:, 0]]
    gj = groups[edges[:, 1]]
    for ig in immune_groups:
        em = ((gi == ig) & np.isin(gj, list(tumor_groups))) | ((gj == ig) & np.isin(gi, list(tumor_groups)))
        n = int(em.sum())
        rows.append({
            "model": label,
            "immune_group": ig,
            "distance_bin": -1,
            "distance_left": np.nan,
            "distance_right": np.nan,
            "distance_mid": np.nan,
            "n_cells": int((groups == ig).sum()),
            "n_tumor_contact_edges": n,
            "mean_tumor_contact_edge_norm_z": float(np.mean(edge_norm_z[em])) if n else np.nan,
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"{label}_tumor_contact_immune_gradient.csv", index=False)

    plot_df = df[df["distance_bin"] >= 0].copy()
    if len(plot_df):
        plot_line(
            plot_df,
            x="mean_dist_to_tumor",
            y="mean_cell_norm_score",
            hue="immune_group",
            title=f"{label}: immune cell program load vs distance to tumor",
            out_path=out_dir / f"{label}_immune_gradient_distance_to_tumor.png",
            xlabel="Distance to nearest tumor cell",
            ylabel="Mean cell norm score",
        )
    return df


# 02
def analysis_tumor_boundary_asymmetry(
    label: str,
    data: Dict[str, object],
    out_dir: Path,
    target_df: pd.DataFrame,
    niche0_value: str,
    boundary_width: float,
    far_width: float,
) -> pd.DataFrame:
    coords = data["coords"]
    niche = data["niche"]
    groups = data["groups"]
    edges = data["edges"]
    edge_norm_z = data["edge_norm_z"]

    mid = 0.5 * (coords[edges[:, 0]] + coords[edges[:, 1]])
    sd = signed_distance_to_niche0_boundary(mid, coords, niche, niche0_value=niche0_value)
    regions = np.array([region_from_signed_distance(x, boundary_width, far_width) for x in sd], dtype=object)

    gi = groups[edges[:, 0]]
    gj = groups[edges[:, 1]]
    rows = []
    for _, r in target_df.iterrows():
        name = str(r["interaction_name"])
        A = split_semicolon(r["side_a_groups"])
        B = split_semicolon(r["side_b_groups"])
        m0 = (np.isin(gi, A) & np.isin(gj, B)) | (np.isin(gi, B) & np.isin(gj, A))
        for region in ["niche0_core", "niche0_inner_edge", "boundary", "outside_inner_edge", "outside_far"]:
            m = m0 & (regions == region)
            n = int(m.sum())
            rows.append({
                "model": label,
                "interaction_name": name,
                "region": region,
                "n_edges": n,
                "mean_signed_distance": float(np.mean(sd[m])) if n else np.nan,
                "mean_edge_norm_z": float(np.mean(edge_norm_z[m])) if n else np.nan,
                "median_edge_norm_z": float(np.median(edge_norm_z[m])) if n else np.nan,
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"{label}_tumor_boundary_asymmetry.csv", index=False)
    if len(df):
        plot_bar(
            df,
            x="region",
            y="mean_edge_norm_z",
            hue="interaction_name",
            title=f"{label}: target crosstalk by niche0 boundary region",
            out_path=out_dir / f"{label}_boundary_asymmetry_target_axes.png",
            xlabel="Boundary region",
            ylabel="Mean edge norm z",
        )
    return df


# 03
def analysis_immune_hot_cold_tumors(
    label: str,
    data: Dict[str, object],
    out_dir: Path,
    tumor_groups: Sequence[str],
    immune_groups: Sequence[str],
    hot_quantile: float,
) -> pd.DataFrame:
    groups = data["groups"]
    edges = data["edges"]
    cell_norm = data["cell_norm"]
    niche = data["niche"]

    tumor_mask = np.isin(groups, list(tumor_groups))
    immune_mask = np.isin(groups, list(immune_groups))
    tumor_idx = np.where(tumor_mask)[0]
    if len(tumor_idx) == 0:
        return pd.DataFrame()

    deg = np.zeros(len(groups), dtype=np.float32)
    immune_deg = np.zeros(len(groups), dtype=np.float32)
    for i, j in edges:
        deg[i] += 1
        deg[j] += 1
        if immune_mask[j]:
            immune_deg[i] += 1
        if immune_mask[i]:
            immune_deg[j] += 1

    frac = np.divide(immune_deg, deg, out=np.zeros_like(immune_deg), where=deg > 0)
    tfrac = frac[tumor_idx]
    lo = np.quantile(tfrac, 1 - hot_quantile)
    hi = np.quantile(tfrac, hot_quantile)

    state = np.full(len(groups), "non_tumor", dtype=object)
    state[tumor_idx[tfrac >= hi]] = "immune_hot_tumor"
    state[tumor_idx[tfrac <= lo]] = "immune_cold_tumor"
    mid = tumor_mask & (state == "non_tumor")
    state[mid] = "immune_mid_tumor"

    rows = []
    for st in ["immune_hot_tumor", "immune_mid_tumor", "immune_cold_tumor"]:
        m = state == st
        rows.append({
            "model": label,
            "tumor_state": st,
            "n_cells": int(m.sum()),
            "mean_immune_neighbor_fraction": float(np.mean(frac[m])) if m.sum() else np.nan,
            "mean_tumor_cell_norm_score": float(np.mean(cell_norm[m])) if m.sum() else np.nan,
        })
        # by niche
        for nv in sorted(pd.unique(pd.Series(niche[tumor_mask]).astype(str))):
            mm = m & (pd.Series(niche).astype(str).to_numpy() == str(nv))
            rows.append({
                "model": label,
                "tumor_state": st,
                "niche": f"niche{nv}",
                "n_cells": int(mm.sum()),
                "mean_immune_neighbor_fraction": float(np.mean(frac[mm])) if mm.sum() else np.nan,
                "mean_tumor_cell_norm_score": float(np.mean(cell_norm[mm])) if mm.sum() else np.nan,
            })

    cell_table = pd.DataFrame({
        "cell_index": np.arange(len(groups)),
        "model": label,
        "group": groups,
        "niche": niche,
        "is_tumor": tumor_mask,
        "degree": deg,
        "immune_neighbor_count": immune_deg,
        "immune_neighbor_fraction": frac,
        "cell_norm_score": cell_norm,
        "tumor_state": state,
    })
    cell_table[cell_table["is_tumor"]].to_csv(out_dir / f"{label}_tumor_hot_cold_cell_table.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"{label}_immune_hot_cold_tumors_summary.csv", index=False)
    base = df[df["niche"].isna()] if "niche" in df.columns else df
    if len(base):
        plot_bar(
            base,
            x="tumor_state",
            y="mean_tumor_cell_norm_score",
            hue=None,
            title=f"{label}: tumor program load in immune-hot vs cold tumors",
            out_path=out_dir / f"{label}_immune_hot_cold_tumors.png",
            xlabel="Tumor immune-neighborhood state",
            ylabel="Mean tumor cell norm score",
        )
    return df


# 04
def analysis_cd8_macrophage_balance(
    label: str,
    data: Dict[str, object],
    out_dir: Path,
    tumor_groups: Sequence[str],
    cd8_group: str,
    macrophage_group: str,
) -> pd.DataFrame:
    groups = data["groups"]
    edges = data["edges"]
    edge_norm_z = data["edge_norm_z"]
    niche = data["niche"]

    is_tumor = np.isin(groups, list(tumor_groups))
    gi = groups[edges[:, 0]]
    gj = groups[edges[:, 1]]
    tumor_edge = np.isin(gi, list(tumor_groups)) | np.isin(gj, list(tumor_groups))
    cd8_tumor = ((gi == cd8_group) & np.isin(gj, list(tumor_groups))) | ((gj == cd8_group) & np.isin(gi, list(tumor_groups)))
    mac_tumor = ((gi == macrophage_group) & np.isin(gj, list(tumor_groups))) | ((gj == macrophage_group) & np.isin(gi, list(tumor_groups)))

    rows = []
    domains = {"global": np.ones(len(edges), dtype=bool)}
    for nv in sorted(pd.unique(pd.Series(niche).astype(str))):
        cm = pd.Series(niche).astype(str).to_numpy() == str(nv)
        domains[f"niche{nv}"] = cm[edges[:, 0]] & cm[edges[:, 1]]

    for dom, dm in domains.items():
        for name, m0 in [("cd8_tumor", cd8_tumor), ("macrophage_tumor", mac_tumor)]:
            m = dm & m0
            n = int(m.sum())
            rows.append({
                "model": label,
                "domain": dom,
                "axis": name,
                "n_edges": n,
                "mean_edge_norm_z": float(np.mean(edge_norm_z[m])) if n else np.nan,
            })
        m_cd8 = dm & cd8_tumor
        m_mac = dm & mac_tumor
        cd8_mean = float(np.mean(edge_norm_z[m_cd8])) if m_cd8.sum() else np.nan
        mac_mean = float(np.mean(edge_norm_z[m_mac])) if m_mac.sum() else np.nan
        rows.append({
            "model": label,
            "domain": dom,
            "axis": "cd8_minus_macrophage_balance",
            "n_edges": int(m_cd8.sum() + m_mac.sum()),
            "mean_edge_norm_z": cd8_mean - mac_mean if np.isfinite(cd8_mean) and np.isfinite(mac_mean) else np.nan,
            "cd8_mean_edge_norm_z": cd8_mean,
            "macrophage_mean_edge_norm_z": mac_mean,
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"{label}_cd8_macrophage_balance.csv", index=False)
    plot_df = df[df["axis"].isin(["cd8_tumor", "macrophage_tumor"])].copy()
    if len(plot_df):
        plot_bar(
            plot_df,
            x="domain",
            y="mean_edge_norm_z",
            hue="axis",
            title=f"{label}: CD8-tumor vs macrophage-tumor crosstalk",
            out_path=out_dir / f"{label}_cd8_macrophage_balance.png",
            xlabel="Domain",
            ylabel="Mean edge norm z",
        )
    return df


# 05
def analysis_tcell_stromal_endothelial_state(
    label: str,
    data: Dict[str, object],
    out_dir: Path,
    tcell_group: str,
    stromal_endothelial_groups: Sequence[str],
    near_radius: float,
) -> pd.DataFrame:
    coords = data["coords"]
    groups = data["groups"]
    cell_norm = data["cell_norm"]
    niche = data["niche"]

    tmask = groups == tcell_group
    smask = np.isin(groups, list(stromal_endothelial_groups))
    if tmask.sum() == 0 or smask.sum() == 0:
        return pd.DataFrame()

    tree = cKDTree(coords[smask])
    d, _ = tree.query(coords[tmask], k=1)
    tidx = np.where(tmask)[0]
    state = np.where(d <= float(near_radius), "near_stromal_endothelial", "far_from_stromal_endothelial")

    rows = []
    for st in ["near_stromal_endothelial", "far_from_stromal_endothelial"]:
        idx = tidx[state == st]
        rows.append({
            "model": label,
            "tcell_group": tcell_group,
            "state": st,
            "n_cells": int(len(idx)),
            "mean_dist_to_stromal_endothelial": float(np.mean(d[state == st])) if len(idx) else np.nan,
            "mean_tcell_norm_score": float(np.mean(cell_norm[idx])) if len(idx) else np.nan,
        })
        for nv in sorted(pd.unique(pd.Series(niche[tidx]).astype(str))):
            mm = (state == st) & (pd.Series(niche[tidx]).astype(str).to_numpy() == str(nv))
            idx2 = tidx[mm]
            rows.append({
                "model": label,
                "tcell_group": tcell_group,
                "state": st,
                "niche": f"niche{nv}",
                "n_cells": int(len(idx2)),
                "mean_dist_to_stromal_endothelial": float(np.mean(d[mm])) if len(idx2) else np.nan,
                "mean_tcell_norm_score": float(np.mean(cell_norm[idx2])) if len(idx2) else np.nan,
            })

    cell_table = pd.DataFrame({
        "model": label,
        "cell_index": tidx,
        "tcell_group": tcell_group,
        "niche": niche[tidx],
        "dist_to_stromal_endothelial": d,
        "state": state,
        "cell_norm_score": cell_norm[tidx],
    })
    cell_table.to_csv(out_dir / f"{label}_tcell_stromal_endothelial_cell_table.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"{label}_tcell_stromal_endothelial_state.csv", index=False)
    base = df[df["niche"].isna()] if "niche" in df.columns else df
    if len(base):
        plot_bar(
            base,
            x="state",
            y="mean_tcell_norm_score",
            hue=None,
            title=f"{label}: {tcell_group} program load near stromal/endothelial",
            out_path=out_dir / f"{label}_tcell_stromal_endothelial_state.png",
            xlabel="T-cell spatial state",
            ylabel="Mean T-cell norm score",
        )
    return df


# 06
def analysis_cross_model_consensus(
    shuffle_csv: str,
    out_dir: Path,
    min_edges: int,
    value_col: str = "shuffle_norm_z",
) -> pd.DataFrame:
    df = pd.read_csv(shuffle_csv)
    df = df[df["n_edges"] >= int(min_edges)].copy()
    if value_col not in df.columns:
        raise ValueError(f"{value_col} missing from shuffle CSV")

    df["rank_within_model"] = (
        df.groupby(["model"])[value_col]
        .rank(method="average", ascending=False)
    )
    df["rank_within_model_domain"] = (
        df.groupby(["model", "domain"])[value_col]
        .rank(method="average", ascending=False)
    )

    group_cols = ["domain", "group_pair_key", "group_a", "group_b"]
    agg = (
        df.groupby(group_cols, as_index=False)
        .agg(
            n_models=("model", "nunique"),
            mean_shuffle_norm_z=(value_col, "mean"),
            median_shuffle_norm_z=(value_col, "median"),
            min_shuffle_norm_z=(value_col, "min"),
            max_shuffle_norm_z=(value_col, "max"),
            mean_rank_within_model=("rank_within_model", "mean"),
            mean_rank_within_model_domain=("rank_within_model_domain", "mean"),
            n_sig_z2=("shuffle_norm_sig_z2", "sum") if "shuffle_norm_sig_z2" in df.columns else (value_col, lambda x: int((x >= 2).sum())),
        )
        .sort_values(["n_models", "mean_shuffle_norm_z"], ascending=[False, False])
    )

    # Add target axis if present.
    if "target_axis" in df.columns:
        target = (
            df.groupby(group_cols, as_index=False)
            .agg(target_axis=("target_axis", lambda x: ";".join(sorted(set([str(v) for v in x if str(v) and str(v) != "nan"]))))
        ))
        agg = agg.merge(target, on=group_cols, how="left")
        agg["is_target_axis"] = agg["target_axis"].fillna("").astype(str).str.len() > 0

    df.to_csv(out_dir / "model_level_ranked_shuffle_control.csv", index=False)
    agg.to_csv(out_dir / "cross_model_consensus_group_pairs.csv", index=False)

    # Per-domain top consensus.
    top = agg.groupby("domain", group_keys=False).head(25)
    top.to_csv(out_dir / "cross_model_consensus_top25_by_domain.csv", index=False)

    for dom, sub in top.groupby("domain"):
        sub = sub.sort_values("mean_shuffle_norm_z", ascending=True).tail(20)
        fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(sub))))
        labels = sub["group_pair_key"].astype(str)
        ax.barh(np.arange(len(sub)), sub["mean_shuffle_norm_z"])
        ax.set_yticks(np.arange(len(sub)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Mean shuffle_norm_z across models")
        ax.set_title(f"Consensus crosstalk group pairs: {dom}")
        fig.tight_layout()
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(dom))
        fig.savefig(out_dir / f"consensus_top_pairs_{safe}.png", dpi=220)
        plt.close(fig)
    return agg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--h5ads", nargs="+", required=True)
    ap.add_argument("--celltype-group-csv", required=True)
    ap.add_argument("--target-interaction-csv", required=True)
    ap.add_argument("--shuffle-control-csv", required=True)
    ap.add_argument("--boundary-density-csv", default=None, help="Optional, not required by current analyses.")
    ap.add_argument("--out-root", required=True)

    ap.add_argument("--celltype-col", default="author_cell_type")
    ap.add_argument("--niche-col", default="niche")
    ap.add_argument("--spatial-key", default="spatial")
    ap.add_argument("--edge-radius", type=float, default=120.0)
    ap.add_argument("--clip-quantile", type=float, default=0.995)

    ap.add_argument("--tumor-groups", nargs="+", default=["tumor"])
    ap.add_argument("--immune-groups", nargs="+", default=["macrophage", "t_cd8_memory", "t_cd4_memory", "treg", "nk", "dendritic", "monocyte"])
    ap.add_argument("--stromal-endothelial-groups", nargs="+", default=["fibroblast", "endothelial"])
    ap.add_argument("--niche0-value", default="0")
    ap.add_argument("--distance-bins", type=int, default=8)
    ap.add_argument("--hot-quantile", type=float, default=0.75)
    ap.add_argument("--near-radius", type=float, default=120.0)
    ap.add_argument("--boundary-width", type=float, default=50.0)
    ap.add_argument("--far-width", type=float, default=150.0)
    ap.add_argument("--consensus-min-edges", type=int, default=50)

    args = ap.parse_args()
    if len(args.labels) != len(args.h5ads):
        raise ValueError("--labels and --h5ads length mismatch")

    out_root = ensure_dir(Path(args.out_root))
    subdirs = {
        "01": ensure_dir(out_root / "01_tumor_contact_immune_gradient"),
        "02": ensure_dir(out_root / "02_tumor_boundary_asymmetry"),
        "03": ensure_dir(out_root / "03_immune_hot_cold_tumors"),
        "04": ensure_dir(out_root / "04_cd8_macrophage_balance"),
        "05": ensure_dir(out_root / "05_tcell_stromal_endothelial_state"),
        "06": ensure_dir(out_root / "06_cross_model_consensus"),
    }

    mapping = load_mapping(args.celltype_group_csv)
    targets = load_targets(args.target_interaction_csv)

    all_01, all_02, all_03, all_04, all_05 = [], [], [], [], []

    for label, h5 in zip(args.labels, args.h5ads):
        data = model_data(
            label=label,
            h5ad_path=h5,
            mapping_df=mapping,
            celltype_col=args.celltype_col,
            niche_col=args.niche_col,
            spatial_key=args.spatial_key,
            edge_radius=args.edge_radius,
            clip_quantile=args.clip_quantile,
        )

        all_01.append(analysis_tumor_contact_immune_gradient(
            label, data, subdirs["01"], args.tumor_groups, args.immune_groups, args.distance_bins
        ))
        all_02.append(analysis_tumor_boundary_asymmetry(
            label, data, subdirs["02"], targets, args.niche0_value, args.boundary_width, args.far_width
        ))
        all_03.append(analysis_immune_hot_cold_tumors(
            label, data, subdirs["03"], args.tumor_groups, args.immune_groups, args.hot_quantile
        ))
        all_04.append(analysis_cd8_macrophage_balance(
            label, data, subdirs["04"], args.tumor_groups, "t_cd8_memory", "macrophage"
        ))
        all_05.append(analysis_tcell_stromal_endothelial_state(
            label, data, subdirs["05"], "t_cd4_memory", args.stromal_endothelial_groups, args.near_radius
        ))

        # Free some memory between models.
        del data

    def concat_write(items: List[pd.DataFrame], out_path: Path) -> None:
        items = [x for x in items if x is not None and len(x)]
        if items:
            pd.concat(items, ignore_index=True).to_csv(out_path, index=False)

    concat_write(all_01, subdirs["01"] / "combined_tumor_contact_immune_gradient.csv")
    concat_write(all_02, subdirs["02"] / "combined_tumor_boundary_asymmetry.csv")
    concat_write(all_03, subdirs["03"] / "combined_immune_hot_cold_tumors_summary.csv")
    concat_write(all_04, subdirs["04"] / "combined_cd8_macrophage_balance.csv")
    concat_write(all_05, subdirs["05"] / "combined_tcell_stromal_endothelial_state.csv")

    consensus = analysis_cross_model_consensus(
        args.shuffle_control_csv,
        subdirs["06"],
        min_edges=args.consensus_min_edges,
        value_col="shuffle_norm_z",
    )

    with open(out_root / "run_summary.json", "w") as f:
        json.dump(
            {
                "args": vars(args),
                "subfolders": {k: str(v) for k, v in subdirs.items()},
                "notes": [
                    "All scores are model-derived interpretable program scores, not ligand-receptor CCC.",
                    "Negative signed distance is niche0/tumor-rich side; positive is non0 side.",
                    "The cross-model consensus uses shuffle_norm_z from the tile-shuffle crosstalk control.",
                ],
            },
            f,
            indent=2,
        )

    print("\nDone. Wrote six analysis folders:")
    for k, v in subdirs.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
