#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import scanpy as sc

from interp_pipeline.downstream.context.neighbors import NeighborConfig
from interp_pipeline.downstream.context.microenv import microenv_embedding
from interp_pipeline.downstream.context.cluster import build_cluster_matrix, pca_embed, ClusterConfig, cluster_labels
from interp_pipeline.downstream.context.validate import (
    split_into_3x3_blocks,
    block_signature_matching,
    leave_one_block_out_lr,
    niche_celltype_table,
    celltype_chi2_residuals,
)
from interp_pipeline.downstream.context.plot import (
    plot_spatial_side_by_side,
    plot_top_positive_residuals,
    collapse_celltypes,
    plot_spatial_labels,
)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def dense_X(adata) -> np.ndarray:
    X = adata.X
    if not isinstance(X, np.ndarray):
        X = X.toarray()
    return X.astype(np.float32)


def save_json(path: str | Path, obj: Dict[str, Any]) -> None:
    def conv(x):
        if hasattr(x, "item"):
            return x.item()
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        return x
    with open(path, "w") as f:
        json.dump({k: conv(v) for k, v in obj.items()}, f, indent=2)


def load_configs_csv(path: Optional[str]) -> Optional[pd.DataFrame]:
    if path is None or str(path).strip().lower() in {"", "none", "null", "na"}:
        return None
    df = pd.read_csv(path)
    needed = {"model", "radius", "space", "kernel", "method", "n_clusters"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"--configs-csv missing columns {missing}. Have={list(df.columns)}")
    return df


def get_model_config(label: str, layer: str, configs: Optional[pd.DataFrame], args) -> Dict[str, Any]:
    if configs is None:
        return {
            "radius": float(args.radius),
            "space": args.space,
            "kernel": args.kernel,
            "sigma_frac": float(args.sigma_frac),
            "pca_components": int(args.pca_components),
            "method": args.method,
            "n_clusters": int(args.n_clusters),
            "seed": int(args.seed),
            "ref_block": int(args.ref_block),
        }

    sub = configs[configs["model"].astype(str) == str(label)].copy()
    if "layer" in configs.columns:
        exact = sub[sub["layer"].astype(str) == str(layer)].copy()
        if len(exact):
            sub = exact
    if sub.empty:
        raise RuntimeError(f"No config row found for model={label}, layer={layer}")

    row = sub.iloc[0].to_dict()
    return {
        "radius": float(row["radius"]),
        "space": str(row["space"]),
        "kernel": str(row["kernel"]),
        "sigma_frac": float(row.get("sigma_frac", args.sigma_frac)),
        "pca_components": int(row.get("pca_components", args.pca_components)),
        "method": str(row["method"]),
        "n_clusters": int(row["n_clusters"]),
        "seed": int(row.get("seed", args.seed)),
        "ref_block": int(row.get("ref_block", args.ref_block)),
    }


def make_lineage_mapping() -> Dict[str, str]:
    return {
        "tumor 13": "Tumor",
        "tumor 9": "Tumor",
        "tumor 12": "Tumor",
        "tumor 5": "Tumor",
        "tumor 6": "Tumor",
        "T CD4 memory": "T cell",
        "T CD8 memory": "T cell",
        "T CD4 naive": "T cell",
        "T CD8 naive": "T cell",
        "Treg": "T cell",
        "NK": "NK",
        "macrophage": "Myeloid",
        "monocyte": "Myeloid",
        "neutrophil": "Myeloid",
        "pDC": "Myeloid",
        "mDC": "Myeloid",
        "fibroblast": "Stromal",
        "endothelial": "Stromal",
        "epithelial": "Epithelial",
        "mast": "Mast",
        "plasmablast": "B cell",
        "B-cell": "B cell",
    }


# Fixed orders and colors for cross-model composition plots.
# The important point is that a given cell type/lineage always maps to the same color,
# even if top_n/min_frac causes a different subset of categories to be shown for a model.
CELLTYPE_ORDER = [
    "tumor 13",
    "tumor 9",
    "tumor 12",
    "tumor 5",
    "tumor 6",
    "epithelial",
    "fibroblast",
    "endothelial",
    "macrophage",
    "monocyte",
    "neutrophil",
    "pDC",
    "mDC",
    "T CD4 memory",
    "T CD8 memory",
    "T CD4 naive",
    "T CD8 naive",
    "Treg",
    "NK",
    "B-cell",
    "plasmablast",
    "mast",
    "Other",
]

CELLTYPE_COLORS = {
    "tumor 13": "#1f77b4",
    "tumor 9": "#aec7e8",
    "tumor 12": "#6baed6",
    "tumor 5": "#08519c",
    "tumor 6": "#3182bd",
    "epithelial": "#9ecae1",
    "fibroblast": "#ff7f0e",
    "endothelial": "#ffbb78",
    "macrophage": "#2ca02c",
    "monocyte": "#98df8a",
    "neutrophil": "#006d2c",
    "pDC": "#74c476",
    "mDC": "#31a354",
    "T CD4 memory": "#d62728",
    "T CD8 memory": "#9467bd",
    "T CD4 naive": "#ff9896",
    "T CD8 naive": "#c5b0d5",
    "Treg": "#8c564b",
    "NK": "#e377c2",
    "B-cell": "#bcbd22",
    "plasmablast": "#dbdb8d",
    "mast": "#7f7f7f",
    "Other": "#bdbdbd",
}

LINEAGE_ORDER = [
    "Tumor",
    "Epithelial",
    "Stromal",
    "Myeloid",
    "T cell",
    "NK",
    "B cell",
    "Mast",
    "Other",
]

LINEAGE_COLORS = {
    "Tumor": "#1f77b4",
    "Epithelial": "#9ecae1",
    "Stromal": "#ff7f0e",
    "Myeloid": "#2ca02c",
    "T cell": "#d62728",
    "NK": "#e377c2",
    "B cell": "#bcbd22",
    "Mast": "#7f7f7f",
    "Other": "#bdbdbd",
}


def _order_columns(tab: pd.DataFrame, order: List[str]) -> pd.DataFrame:
    known = [c for c in order if c in tab.columns]
    extra = [c for c in tab.columns if c not in known]
    return tab.loc[:, known + sorted(extra)]


def stacked_bar_composition_fixed_colors(
    tab: pd.DataFrame,
    *,
    outpath: str,
    title: str,
    min_frac: float = 0.005,
    top_n: Optional[int] = None,
    color_map: Optional[Dict[str, str]] = None,
    order: Optional[List[str]] = None,
    other_label: str = "Other",
) -> None:
    """
    Stacked niche-composition bar plot with stable category colors.

    This local wrapper avoids model-dependent matplotlib/pandas color cycling.
    It keeps the same behavior as the existing composition plot in spirit:
    low-abundance or non-top categories are collapsed into `Other`, then columns
    are plotted in a fixed order and colored by category name.
    """
    import matplotlib.pyplot as plt

    counts = tab.copy()
    counts.columns = counts.columns.astype(str)
    counts.index = counts.index.astype(str)

    frac = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    keep = list(frac.columns)
    if min_frac is not None and float(min_frac) > 0:
        keep = [c for c in keep if float(frac[c].max()) >= float(min_frac)]

    if top_n is not None and int(top_n) > 0 and len(keep) > int(top_n):
        totals = counts[keep].sum(axis=0).sort_values(ascending=False)
        keep = totals.head(int(top_n)).index.astype(str).tolist()

    dropped = [c for c in counts.columns if c not in keep]
    plot_counts = counts[keep].copy()
    if dropped:
        plot_counts[other_label] = counts[dropped].sum(axis=1)

    plot_frac = plot_counts.div(plot_counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    if order is not None:
        plot_frac = _order_columns(plot_frac, order)

    if color_map is not None:
        colors = [color_map.get(str(c), "#bdbdbd") for c in plot_frac.columns]
    else:
        colors = None

    fig, ax = plt.subplots(figsize=(max(5.5, 1.4 * len(plot_frac.index) + 2.5), 4.5))
    plot_frac.plot(kind="bar", stacked=True, ax=ax, width=0.85, color=colors)

    ax.set_title(title)
    ax.set_xlabel("niche")
    ax.set_ylabel("fraction of cells")
    ax.set_ylim(0, 1)
    ax.legend(
        title=None,
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        borderaxespad=0.0,
        fontsize=8,
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def relabel_niches_by_celltype_composition(
    y: np.ndarray,
    celltypes: pd.Series,
    *,
    target_celltype: str = "tumor 13",
) -> tuple[np.ndarray, Dict[str, Any]]:
    """
    Re-ID niche labels by descending composition of target_celltype.

    Example for k=3:
      old niche with highest tumor13 fraction -> new niche 0
      next highest                         -> new niche 1
      lowest                              -> new niche 2

    Ties are broken by descending target count, then descending cluster size,
    then old label ascending for deterministic behavior.
    """
    y = np.asarray(y).astype(int)
    ct = celltypes.astype(str).reset_index(drop=True)
    if len(ct) != len(y):
        raise ValueError(f"celltypes length={len(ct)} but y length={len(y)}")

    rows = []
    for old_lab in sorted(np.unique(y).tolist()):
        mask = y == int(old_lab)
        n_total = int(mask.sum())
        n_target = int((ct[mask] == str(target_celltype)).sum())
        frac = float(n_target / max(1, n_total))
        rows.append(
            {
                "old_niche": int(old_lab),
                "target_celltype": str(target_celltype),
                "target_count": n_target,
                "niche_size": n_total,
                "target_fraction": frac,
            }
        )

    comp = pd.DataFrame(rows)
    comp = comp.sort_values(
        ["target_fraction", "target_count", "niche_size", "old_niche"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    comp["new_niche"] = np.arange(len(comp), dtype=int)

    old_to_new = {int(r.old_niche): int(r.new_niche) for r in comp.itertuples(index=False)}
    y_new = np.asarray([old_to_new[int(v)] for v in y], dtype=int)

    info = {
        "target_celltype": str(target_celltype),
        "old_to_new": {str(k): int(v) for k, v in old_to_new.items()},
        "composition": comp.to_dict(orient="records"),
    }
    return y_new, info


def validate_one_model(
    label: str,
    layer: str,
    interp_h5ad: str,
    out_dir: Path,
    cfg: Dict[str, Any],
    celltype_col: str,
    lr_max_iter: int,
    lr_c: float,
    side_by_side_delta_k: int,
    celltype_min_frac: float,
    celltype_top_n: int,
    residual_top_n: int,
    residual_z_thresh: float,
    relabel_by_celltype: Optional[str],
) -> Dict[str, Any]:
    ensure_dir(out_dir)
    plot_dir = ensure_dir(out_dir / "plots")

    print("=" * 100)
    print(f"[niche validation] {label} | {layer}")
    print(f"  h5ad={interp_h5ad}")
    print(f"  out_dir={out_dir}")
    print(f"  cfg={cfg}")
    print("=" * 100)

    adata = sc.read_h5ad(interp_h5ad)
    if "spatial" not in adata.obsm:
        raise RuntimeError(f"Missing adata.obsm['spatial'] in {interp_h5ad}")

    X = dense_X(adata)
    coords = adata.obsm["spatial"].astype(np.float32)
    print("  X:", X.shape, "coords:", coords.shape)

    print("[1] Build microenvironment embedding")
    ncfg = NeighborConfig(radius=float(cfg["radius"]), kernel=str(cfg["kernel"]), sigma_frac=float(cfg["sigma_frac"]))
    m, neigh = microenv_embedding(X, coords, ncfg)
    Z = build_cluster_matrix(X, m, space=str(cfg["space"]))
    E = pca_embed(Z, n_components=int(cfg["pca_components"]), seed=int(cfg["seed"]))

    print("[2] Cluster global niches")
    ccfg = ClusterConfig(
        space=str(cfg["space"]),
        pca_components=int(cfg["pca_components"]),
        method=str(cfg["method"]),
        n_clusters=int(cfg["n_clusters"]),
        seed=int(cfg["seed"]),
    )
    y_raw = cluster_labels(E, ccfg).astype(int)

    relabel_info = {
        "enabled": False,
        "target_celltype": None,
        "old_to_new": {},
        "composition": [],
    }
    if (
        relabel_by_celltype is not None
        and str(relabel_by_celltype).strip().lower() not in {"", "none", "null", "na"}
        and celltype_col in adata.obs.columns
    ):
        y, info = relabel_niches_by_celltype_composition(
            y_raw,
            adata.obs[celltype_col],
            target_celltype=str(relabel_by_celltype),
        )
        relabel_info = {"enabled": True, **info}
        pd.DataFrame(info["composition"]).to_csv(out_dir / "niche_relabel_by_celltype.csv", index=False)
        with open(out_dir / "niche_relabel_by_celltype.json", "w") as f:
            json.dump(relabel_info, f, indent=2)
        print(f"[relabel] by {relabel_by_celltype!r}: old_to_new={info['old_to_new']}")
    else:
        y = y_raw
        if relabel_by_celltype is not None and celltype_col not in adata.obs.columns:
            print(f"[relabel skip] celltype_col={celltype_col!r} not in adata.obs")

    pd.DataFrame(
        {
            "cell": adata.obs_names.astype(str),
            "niche_raw": y_raw,
            "niche": y,
        }
    ).to_csv(out_dir / "global_labels.csv", index=False)
    adata.obs["niche_raw"] = y_raw.astype(str)
    adata.obs["niche"] = y.astype(str)
    adata.write_h5ad(out_dir / "adata_with_niche_labels.h5ad")

    print("[3] Spatial plots")
    plot_spatial_labels(
        coords,
        y,
        str(plot_dir / f"spatial_niches_k{int(cfg['n_clusters'])}.png"),
        title=f"{label}: spatial niches (r={cfg['radius']}, k={cfg['n_clusters']}, {cfg['space']}, {cfg['method']})",
        s=1,
    )

    k2 = int(cfg["n_clusters"]) + int(side_by_side_delta_k)
    if k2 > 1:
        ccfg2 = ClusterConfig(
            space=str(cfg["space"]),
            pca_components=int(cfg["pca_components"]),
            method=str(cfg["method"]),
            n_clusters=k2,
            seed=int(cfg["seed"]),
        )
        y2 = cluster_labels(E, ccfg2).astype(int)
        plot_spatial_side_by_side(
            coords,
            y,
            y2,
            outpath=str(plot_dir / f"spatial_side_by_side_k{int(cfg['n_clusters'])}_k{k2}.png"),
            title_a=f"{label}: k={int(cfg['n_clusters'])}",
            title_b=f"{label}: k={k2}",
        )

    print("[4] Cell-type composition/enrichment")
    celltype_outputs = {}
    if celltype_col in adata.obs.columns:
        celltypes = adata.obs[celltype_col].astype(str)

        tab = pd.crosstab(pd.Series(y.astype(int), name="niche").to_numpy(), celltypes.astype(str).to_numpy())
        tab.to_csv(out_dir / "celltype_crosstab_counts_wide.csv")

        stacked_bar_composition_fixed_colors(
            tab,
            outpath=str(plot_dir / f"celltype_composition_k{int(cfg['n_clusters'])}.png"),
            title=f"{label}: cell type composition per niche",
            min_frac=float(celltype_min_frac),
            top_n=int(celltype_top_n),
            color_map=CELLTYPE_COLORS,
            order=CELLTYPE_ORDER,
        )

        resid_all = celltype_chi2_residuals(y, celltypes)
        resid_all.to_csv(out_dir / "celltype_residuals_all.csv", index=False)

        resid_mat = resid_all.pivot_table(index="niche", columns="celltype", values="std_resid", fill_value=0.0)
        plot_top_positive_residuals(
            resid_mat,
            outpath=str(plot_dir / f"top_positive_residuals_k{int(cfg['n_clusters'])}.png"),
            title=f"{label}: top positive cell-type enrichments per niche",
            top_n=int(residual_top_n),
            z_thresh=float(residual_z_thresh),
        )

        ctab = niche_celltype_table(y, celltypes)
        ctab.to_csv(out_dir / "celltype_crosstab_count_frac.csv", index=False)

        top_pos = resid_all.sort_values("std_resid", ascending=False).head(50)
        top_neg = resid_all.sort_values("std_resid", ascending=True).head(50)
        pd.concat(
            [top_pos.assign(direction="enriched"), top_neg.assign(direction="depleted")],
            ignore_index=True,
        ).to_csv(out_dir / "celltype_residuals_top.csv", index=False)

        tab_lin = collapse_celltypes(tab, make_lineage_mapping(), other_label="Other")
        tab_lin.to_csv(out_dir / "celltype_lineage_crosstab_counts_wide.csv")
        stacked_bar_composition_fixed_colors(
            tab_lin,
            outpath=str(plot_dir / f"celltype_lineage_composition_k{int(cfg['n_clusters'])}.png"),
            title=f"{label}: lineage composition per niche",
            min_frac=0.02,
            color_map=LINEAGE_COLORS,
            order=LINEAGE_ORDER,
        )

        celltype_outputs = {
            "celltype_crosstab": str(out_dir / "celltype_crosstab_count_frac.csv"),
            "celltype_residuals_all": str(out_dir / "celltype_residuals_all.csv"),
            "celltype_residuals_top": str(out_dir / "celltype_residuals_top.csv"),
        }
    else:
        print(f"[skip] adata.obs does not contain {celltype_col!r}")

    print("[5] 3x3 block split")
    block_id = split_into_3x3_blocks(coords)
    pd.DataFrame({"cell": adata.obs_names.astype(str), "block": block_id.astype(int)}).to_csv(
        out_dir / "block_ids.csv", index=False
    )

    print("[6] Block signature matching")
    match_df = block_signature_matching(E, y, block_id, ref_block=int(cfg["ref_block"]))
    match_df.to_csv(out_dir / "block_signature_matches.csv", index=False)

    print("[7] Leave-one-block-out LR")
    loo = leave_one_block_out_lr(E, y, block_id, seed=int(cfg["seed"]), max_iter=int(lr_max_iter), C=float(lr_c))
    loo.to_csv(out_dir / "loo_lr_metrics.csv", index=False)

    summary = {
        "model": label,
        "layer": layer,
        "interp_h5ad": interp_h5ad,
        "out_dir": str(out_dir),
        "n_cells": int(adata.n_obs),
        "n_features": int(adata.n_vars),
        "celltype_col": celltype_col,
        **cfg,
        "relabel_by_celltype_enabled": bool(relabel_info.get("enabled", False)),
        "relabel_target_celltype": relabel_info.get("target_celltype", None),
        "relabel_old_to_new": relabel_info.get("old_to_new", {}),
        "niche_counts": {str(k): int(v) for k, v in pd.Series(y).value_counts().sort_index().items()},
        "block_signature_mean_cosine_mean": float(match_df["mean_cosine_sim"].mean()) if len(match_df) else float("nan"),
        "block_signature_pair_cosine_mean": float(match_df["cosine_sim"].mean()) if len(match_df) else float("nan"),
        "loo_acc_mean": float(loo["acc"].mean()) if len(loo) else float("nan"),
        "loo_bal_acc_mean": float(loo["bal_acc"].mean()) if len(loo) else float("nan"),
        "loo_macro_f1_mean": float(loo["macro_f1"].mean()) if len(loo) else float("nan"),
        **celltype_outputs,
    }
    save_json(out_dir / "validation_summary.json", summary)
    print("[OK]", json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Run 3-model niche validation and visualization.")
    ap.add_argument("--labels", nargs=3, required=True)
    ap.add_argument("--layers", nargs=3, required=True)
    ap.add_argument("--interp-h5ads", nargs=3, required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--configs-csv", default=None)

    ap.add_argument("--radius", type=float, default=120.0)
    ap.add_argument("--space", default="xm", choices=["m", "xm"])
    ap.add_argument("--kernel", default="uniform", choices=["uniform", "gaussian"])
    ap.add_argument("--sigma-frac", type=float, default=0.25)
    ap.add_argument("--pca-components", type=int, default=50)
    ap.add_argument("--method", default="gmm", choices=["kmeans", "gmm"])
    ap.add_argument("--n-clusters", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ref-block", type=int, default=4)

    ap.add_argument("--celltype-col", default="author_cell_type")
    ap.add_argument("--lr-max-iter", type=int, default=2000)
    ap.add_argument("--lr-c", type=float, default=1.0)
    ap.add_argument("--side-by-side-delta-k", type=int, default=1)
    ap.add_argument("--celltype-min-frac", type=float, default=0.005)
    ap.add_argument("--celltype-top-n", type=int, default=15)
    ap.add_argument("--residual-top-n", type=int, default=5)
    ap.add_argument("--residual-z-thresh", type=float, default=1.5)
    ap.add_argument(
        "--relabel-by-celltype",
        default="tumor 13",
        help="Re-ID niche labels by descending composition of this cell type. Use NONE to disable.",
    )

    args = ap.parse_args()
    out_root = ensure_dir(args.out_root)
    configs = load_configs_csv(args.configs_csv)

    summaries: List[Dict[str, Any]] = []
    for label, layer, h5ad in zip(args.labels, args.layers, args.interp_h5ads):
        cfg = get_model_config(label, layer, configs, args)
        out_dir = out_root / label / layer.replace("/", "_") / (
            f"r{str(cfg['radius']).replace('.', 'p')}_{cfg['space']}_{cfg['method']}_k{int(cfg['n_clusters'])}"
        )

        summary = validate_one_model(
            label=label,
            layer=layer,
            interp_h5ad=h5ad,
            out_dir=out_dir,
            cfg=cfg,
            celltype_col=args.celltype_col,
            lr_max_iter=args.lr_max_iter,
            lr_c=args.lr_c,
            side_by_side_delta_k=args.side_by_side_delta_k,
            celltype_min_frac=args.celltype_min_frac,
            celltype_top_n=args.celltype_top_n,
            residual_top_n=args.residual_top_n,
            residual_z_thresh=args.residual_z_thresh,
            relabel_by_celltype=args.relabel_by_celltype,
        )
        summaries.append(summary)

    df = pd.DataFrame(summaries)
    df.to_csv(out_root / "combined_validation_summary.csv", index=False)

    print("\n[OK] wrote:", out_root)
    keep = [
        "model", "layer", "radius", "space", "method", "n_clusters", "n_features",
        "relabel_by_celltype_enabled", "relabel_target_celltype",
        "block_signature_mean_cosine_mean", "loo_acc_mean", "loo_bal_acc_mean",
        "loo_macro_f1_mean", "out_dir",
    ]
    print(df[[c for c in keep if c in df.columns]].to_string(index=False))


if __name__ == "__main__":
    main()


# python test_niche_validation_3models.py \
#   --labels scgpt c2sscale geneformer \
#   --layers layer_4.norm2 layer_17 layer_4 \
#   --interp-h5ads \
#     /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/interpretable_adata_3models_smoke/scgpt/layer_4.norm2/adata_interpretable_layer_4.norm2_saeThr0p15_f1cut0p4_top300_mean.h5ad \
#     /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/interpretable_adata_3models_smoke/c2sscale/layer_17/adata_interpretable_layer_17_saeThr0p15_f1cut0p4_top300_mean.h5ad \
#     /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/interpretable_adata_3models_smoke/geneformer/layer_4/adata_interpretable_layer_4_saeThr0p15_f1cut0p4_top300_mean.h5ad \
#   --out-root /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/niche_validation_3models_shared \
#   --celltype-col author_cell_type \
#   --radius 120 \
#   --space xm \
#   --kernel uniform \
#   --method gmm \
#   --n-clusters 3 \
#   --seed 0

# python test_niche_validation_3models.py \
#   --labels scgpt c2sscale geneformer \
#   --layers layer_4.norm2 layer_17 layer_4 \
#   --interp-h5ads \
#     /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/interpretable_adata_3models_smoke/scgpt/layer_4.norm2/adata_interpretable_layer_4.norm2_saeThr0p15_f1cut0p4_top300_mean.h5ad \
#     /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/interpretable_adata_3models_smoke/c2sscale/layer_17/adata_interpretable_layer_17_saeThr0p15_f1cut0p4_top300_mean.h5ad \
#     /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/interpretable_adata_3models_smoke/geneformer/layer_4/adata_interpretable_layer_4_saeThr0p15_f1cut0p4_top300_mean.h5ad \
#   --configs-csv /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/niche_discovery_3models/chosen_validation_configs.csv \
#   --out-root /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/niche_validation_3models_best \
#   --celltype-col author_cell_type