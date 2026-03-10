from __future__ import annotations

import os
import numpy as np
import pandas as pd
import scanpy as sc

from interp_pipeline.downstream.context.sweep import SweepConfig, run_niche_sweep

# =========================
# EDIT HERE
# =========================
INTERP_H5AD = "runs/full_scgpt_cosmx/interpretable_adata/layer_4.norm2/sae_thr_0.6/adata_interpretable_layer_4.norm2_saeThr0.6_bestonly_conceptdedup_f1cut0.6_topall.h5ad"
OUTDIR = "runs/full_scgpt_cosmx/niche_discovery/layer_4.norm2"
os.makedirs(OUTDIR, exist_ok=True)

CFG = SweepConfig(
    radii=[120, 150, 180],
    n_clusters_list=[3,4,5],
    method_list=["kmeans", "gmm"],
    space_list=["m", "xm"],
    kernel="uniform",        # notebook supports gaussian too
    sigma_frac=0.25,
    pca_components=50,
    seeds=[0, 1, 2, 3, 4],
    min_cluster_size=150,
    tile_mult=4.0,
    n_null=20,
    null_seed=0,
)
# =========================


def main():
    print("[1] Load interpretable h5ad")
    adata = sc.read_h5ad(INTERP_H5AD)
    if "spatial" not in adata.obsm:
        raise RuntimeError("Missing adata.obsm['spatial'] in interpretable h5ad")

    X = adata.X
    if not isinstance(X, np.ndarray):
        X = X.toarray()
    X = X.astype(np.float32)
    coords = adata.obsm["spatial"].astype(np.float32)

    print("  X:", X.shape, "coords:", coords.shape)

    print("[2] Run sweep")
    df = run_niche_sweep(X, coords, CFG)

    out_csv = os.path.join(OUTDIR, "niche_sweep_results.csv")
    df.to_csv(out_csv, index=False)
    print("[OK] wrote:", out_csv)

    if len(df):
        print("\nTop 10 configs:")
        print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()