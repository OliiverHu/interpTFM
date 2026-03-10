from __future__ import annotations

import os
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
    stacked_bar_composition,
    plot_top_positive_residuals,
    collapse_celltypes,
    plot_spatial_labels
)

# =========================
# EDIT HERE
# =========================
INTERP_H5AD = "runs/full_scgpt_cosmx/interpretable_adata/layer_4.norm2/sae_thr_0.6/adata_interpretable_layer_4.norm2_saeThr0.6_bestonly_conceptdedup_f1cut0.6_topall.h5ad"
OUTDIR = "runs/full_scgpt_cosmx/niche_discovery/layer_4.norm2/validation_k3_r120_m"
os.makedirs(OUTDIR, exist_ok=True)

CELLTYPE_COL = "author_cell_type"

# chosen niche config (your "ideal" for cancer tissue)
RADIUS = 120.0
SPACE = "m"          # m-only for compartments
KERNEL = "uniform"
SIGMA_FRAC = 0.25
PCA_COMPONENTS = 50
METHOD = "kmeans"
N_CLUSTERS = 3
SEED = 0

# LOO LR hyperparams
LR_MAX_ITER = 2000
LR_C = 1.0

REF_BLOCK = 4  # center block in 3x3 grid
# =========================


def main():
    print("[1] Load interpretable h5ad")
    adata = sc.read_h5ad(INTERP_H5AD)
    if "spatial" not in adata.obsm:
        raise RuntimeError("Missing adata.obsm['spatial']")

    X = adata.X
    if not isinstance(X, np.ndarray):
        X = X.toarray()
    X = X.astype(np.float32)
    coords = adata.obsm["spatial"].astype(np.float32)

    print("  X:", X.shape, "coords:", coords.shape)

    # ---- Build microenv + clustering space ----
    print("[2] Build microenvironment embedding")
    ncfg = NeighborConfig(radius=RADIUS, kernel=KERNEL, sigma_frac=SIGMA_FRAC)
    m, neigh = microenv_embedding(X, coords, ncfg)

    Z = build_cluster_matrix(X, m, space=SPACE)  # m or xm
    E = pca_embed(Z, n_components=PCA_COMPONENTS, seed=SEED)

    print("[3] Cluster -> global labels")
    ccfg = ClusterConfig(space=SPACE, pca_components=PCA_COMPONENTS, method=METHOD, n_clusters=N_CLUSTERS, seed=SEED)
    y = cluster_labels(E, ccfg)

    pd.DataFrame({"cell": adata.obs_names.astype(str), "niche": y.astype(int)}).to_csv(
        os.path.join(OUTDIR, "global_labels.csv"), index=False
    )

    PLOT_DIR = os.path.join(OUTDIR, "plots")
    os.makedirs(PLOT_DIR, exist_ok=True)

    # --- Spatial plots ---
    plot_spatial_labels_path = os.path.join(PLOT_DIR, "spatial_niches_k3.png")
    # from interp_pipeline.downstream.niche_discovery.plot import plot_spatial_labels
    plot_spatial_labels(coords, y, plot_spatial_labels_path, title=f"Spatial niches (r={RADIUS}, k={N_CLUSTERS}, seed={SEED})")

    # Optional side-by-side k=3 vs k=4 (matches notebook)
    # If you want exactly like notebook, compute k=4 once:
    k4 = 4
    ccfg4 = ClusterConfig(space=SPACE, pca_components=PCA_COMPONENTS, method=METHOD, n_clusters=k4, seed=SEED)
    y4 = cluster_labels(E, ccfg4)
    plot_spatial_side_by_side(
        coords, y, y4,
        outpath=os.path.join(PLOT_DIR, "spatial_side_by_side_k3_k4.png"),
        title_a=f"Spatial niches (r={RADIUS}, k=3, seed={SEED})",
        title_b=f"Spatial niches (r={RADIUS}, k=4, seed={SEED})",
    )

    # --- Cell-type composition + residual plots ---
    if CELLTYPE_COL in adata.obs.columns:
        celltypes = adata.obs[CELLTYPE_COL].astype(str)

        # raw crosstab niche x celltype
        tab = pd.crosstab(
            pd.Series(y.astype(int), name="niche").to_numpy(),
            celltypes.astype(str).to_numpy(),
        )

        stacked_bar_composition(
            tab,
            outpath=os.path.join(PLOT_DIR, "celltype_composition_k3.png"),
            title="Cell type composition per niche (k=3)",
            min_frac=0.005,
            top_n=15,
        )

        # chi-square residuals matrix niche x celltype
        resid_all = celltype_chi2_residuals(y, celltypes)
        # convert to wide matrix like notebook expects: niche x celltype
        resid_mat = resid_all.pivot_table(index="niche", columns="celltype", values="std_resid", fill_value=0.0)

        plot_top_positive_residuals(
            resid_mat,
            outpath=os.path.join(PLOT_DIR, "top_positive_residuals_k3.png"),
            title="Top positive cell-type enrichments per niche (k=3)",
            top_n=5,
            z_thresh=1.5,
        )

        # lineage collapse (notebook-style)
        CELLTYPE_TO_LINEAGE = {
            "tumor 13": "Tumor",
            "T CD4 memory": "T cell",
            "T CD8 memory": "T cell",
            "Treg": "T cell",
            "macrophage": "Myeloid",
            "pDC": "Myeloid",
            "fibroblast": "Stromal",
            "endothelial": "Stromal",
            "mast": "Mast",
            "plasmablast": "B cell",
        }
        tab_lin = collapse_celltypes(tab, CELLTYPE_TO_LINEAGE, other_label="Other")
        stacked_bar_composition(
            tab_lin,
            outpath=os.path.join(PLOT_DIR, "celltype_lineage_composition_k3.png"),
            title="Lineage composition per niche (k=3)",
            min_frac=0.02,
        )

    # ---- Block split ----
    print("[4] 3x3 block split")
    block_id = split_into_3x3_blocks(coords)
    pd.DataFrame({"cell": adata.obs_names.astype(str), "block": block_id.astype(int)}).to_csv(
        os.path.join(OUTDIR, "block_ids.csv"), index=False
    )

    # ---- Signature matching ----
    print("[5] Block signature matching + Hungarian")
    match_df = block_signature_matching(E, y, block_id, ref_block=REF_BLOCK)
    match_df.to_csv(os.path.join(OUTDIR, "block_signature_matches.csv"), index=False)

    # ---- Leave-one-block-out LR ----
    print("[6] Leave-one-block-out multinomial LR (predict global niche)")
    loo = leave_one_block_out_lr(E, y, block_id, seed=SEED, max_iter=LR_MAX_ITER, C=LR_C)
    loo.to_csv(os.path.join(OUTDIR, "loo_lr_metrics.csv"), index=False)

    # ---- Cell-type enrichment ----
    if CELLTYPE_COL in adata.obs.columns:
        print("[7] Cell-type enrichment:", CELLTYPE_COL)
        celltypes = adata.obs[CELLTYPE_COL].astype(str)

        ctab = niche_celltype_table(y, celltypes)
        ctab.to_csv(os.path.join(OUTDIR, "celltype_crosstab.csv"), index=False)

        resid = celltype_chi2_residuals(y, celltypes)
        resid.to_csv(os.path.join(OUTDIR, "celltype_residuals_all.csv"), index=False)

        # top enriched/depleted
        top_pos = resid.sort_values("std_resid", ascending=False).head(50)
        top_neg = resid.sort_values("std_resid", ascending=True).head(50)
        pd.concat(
            [top_pos.assign(direction="enriched"), top_neg.assign(direction="depleted")],
            ignore_index=True,
        ).to_csv(os.path.join(OUTDIR, "celltype_residuals_top.csv"), index=False)
    else:
        print(f"[skip] adata.obs does not contain '{CELLTYPE_COL}'")

    print("[OK] wrote validation outputs to:", OUTDIR)
    if len(match_df):
        print("  signature matching mean_sim (by block):")
        print(match_df.groupby("block", as_index=False)["mean_cosine_sim"].mean().to_string(index=False))
    if len(loo):
        print("  LOO LR summary:")
        print(loo.describe().to_string())


if __name__ == "__main__":
    main()