#!/usr/bin/env python3
from __future__ import annotations

"""
3-model niche discovery sweep.

This adapts scripts/test_niche_sweep.py to run the same microenvironment/niche
sweep over three interpretable AnnData files.

Inputs:
  Interpretable .h5ad files produced by build_interpretable_adata_3models.py

For each model:
  - load interpretable h5ad
  - X = adata.X
  - coords = adata.obsm["spatial"]
  - run run_niche_sweep(X, coords, SweepConfig)
  - save per-model niche_sweep_results.csv

Combined:
  - combined_niche_sweep_results.csv
  - combined_top_configs.csv
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import scanpy as sc

from interp_pipeline.downstream.context.sweep import SweepConfig, run_niche_sweep


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
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def jsonable(x: Any) -> Any:
    if hasattr(x, "item"):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    return x


def run_one(
    *,
    label: str,
    layer: str,
    interp_h5ad: str,
    out_dir: Path,
    cfg: SweepConfig,
) -> pd.DataFrame:
    ensure_dir(out_dir)

    print("=" * 100)
    print(f"[niche sweep] {label} | {layer}")
    print(f"  h5ad={interp_h5ad}")
    print(f"  out_dir={out_dir}")
    print("=" * 100)

    adata = sc.read_h5ad(interp_h5ad)
    if "spatial" not in adata.obsm:
        raise RuntimeError(f"Missing adata.obsm['spatial'] in {interp_h5ad}")

    X = dense_X(adata)
    coords = adata.obsm["spatial"].astype(np.float32)

    print("  X:", X.shape, "coords:", coords.shape)

    df = run_niche_sweep(X, coords, cfg)
    df.insert(0, "model", label)
    df.insert(1, "layer", layer)
    df.insert(2, "interp_h5ad", interp_h5ad)
    df.insert(3, "n_cells", int(adata.n_obs))
    df.insert(4, "n_features", int(adata.n_vars))

    out_csv = out_dir / "niche_sweep_results.csv"
    df.to_csv(out_csv, index=False)

    summary = {
        "model": label,
        "layer": layer,
        "interp_h5ad": interp_h5ad,
        "n_cells": int(adata.n_obs),
        "n_features": int(adata.n_vars),
        "out_csv": str(out_csv),
        "n_rows": int(len(df)),
    }
    if len(df):
        top = df.head(1).iloc[0].to_dict()
        summary["top_config"] = {k: jsonable(v) for k, v in top.items()}

    save_json(out_dir / "sweep_summary.json", summary)

    print("[OK] wrote:", out_csv)
    if len(df):
        print("\nTop 10 configs:")
        print(df.head(10).to_string(index=False))
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Run 3-model niche discovery sweep on interpretable h5ads.")
    ap.add_argument("--labels", nargs=3, required=True)
    ap.add_argument("--layers", nargs=3, required=True)
    ap.add_argument("--interp-h5ads", nargs=3, required=True)
    ap.add_argument("--out-root", required=True)

    ap.add_argument("--radii", nargs="+", type=float, default=[120, 150, 180])
    ap.add_argument("--n-clusters-list", nargs="+", type=int, default=[3, 4, 5])
    ap.add_argument("--methods", nargs="+", default=["kmeans", "gmm"])
    ap.add_argument("--spaces", nargs="+", default=["m", "xm"])
    ap.add_argument("--kernel", default="uniform", choices=["uniform", "gaussian"])
    ap.add_argument("--sigma-frac", type=float, default=0.25)
    ap.add_argument("--pca-components", type=int, default=50)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--min-cluster-size", type=int, default=150)
    ap.add_argument("--tile-mult", type=float, default=4.0)
    ap.add_argument("--n-null", type=int, default=20)
    ap.add_argument("--null-seed", type=int, default=0)

    args = ap.parse_args()
    out_root = ensure_dir(args.out_root)

    cfg = SweepConfig(
        radii=list(args.radii),
        n_clusters_list=list(args.n_clusters_list),
        method_list=list(args.methods),
        space_list=list(args.spaces),
        kernel=args.kernel,
        sigma_frac=float(args.sigma_frac),
        pca_components=int(args.pca_components),
        seeds=list(args.seeds),
        min_cluster_size=int(args.min_cluster_size),
        tile_mult=float(args.tile_mult),
        n_null=int(args.n_null),
        null_seed=int(args.null_seed),
    )

    all_frames: List[pd.DataFrame] = []
    for label, layer, h5ad in zip(args.labels, args.layers, args.interp_h5ads):
        model_out = out_root / label / layer.replace("/", "_")
        df = run_one(
            label=label,
            layer=layer,
            interp_h5ad=h5ad,
            out_dir=model_out,
            cfg=cfg,
        )
        all_frames.append(df)

    combined = pd.concat(all_frames, ignore_index=True)
    combined.to_csv(out_root / "combined_niche_sweep_results.csv", index=False)

    top_rows = []
    for (_model, _layer), g in combined.groupby(["model", "layer"], sort=False):
        top_rows.append(g.head(10))
    top = pd.concat(top_rows, ignore_index=True)
    top.to_csv(out_root / "combined_top_configs.csv", index=False)

    print("\n[OK] wrote:", out_root)
    print("Combined top configs:")
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()


# python test_niche_sweep_3models.py \
#   --labels scgpt c2sscale geneformer \
#   --layers layer_4.norm2 layer_17 layer_4 \
#   --interp-h5ads \
#     /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/interpretable_adata_3models_smoke/scgpt/layer_4.norm2/adata_interpretable_layer_4.norm2_saeThr0p15_f1cut0p4_top300_mean.h5ad \
#     /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/interpretable_adata_3models_smoke/c2sscale/layer_17/adata_interpretable_layer_17_saeThr0p15_f1cut0p4_top300_mean.h5ad \
#     /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/interpretable_adata_3models_smoke/geneformer/layer_4/adata_interpretable_layer_4_saeThr0p15_f1cut0p4_top300_mean.h5ad \
#   --out-root /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/niche_discovery_3models \
#   --radii 120 150 180 \
#   --n-clusters-list 3 4 5 6 \
#   --methods kmeans gmm \
#   --spaces m xm \
#   --seeds 0 1 2 3 4