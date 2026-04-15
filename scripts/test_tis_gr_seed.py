from __future__ import annotations

import os
import numpy as np
import scanpy as sc
import pandas as pd

from interp_pipeline.tis.core import TISConfig
from interp_pipeline.tis.io import extract_cls_from_shards_aligned, build_judge_matrix
from interp_pipeline.tis.gr_seed import (
    run_seed_reproducibility,
    run_light_grid,
    TISGridConfig,
    save_json,
)

# =========================
# HARD-CODE CONFIG HERE
# =========================
RUNS_ROOT = "runs/full_scgpt_cosmx"
LAYER = "layer_4.norm2"

ADATA_PATH = "/maiziezhou_lab2/yunfei/Projects/FM_temp/InterPLM/interplm/ge_shards/cosmx_human_lung_sec8.h5ad"
ACTS_ROOT = os.path.join(RUNS_ROOT, "activations", LAYER)

OUTDIR = os.path.join(RUNS_ROOT, "tis", LAYER, "seed_grid")
os.makedirs(OUTDIR, exist_ok=True)

MAX_SHARDS = 20

# Match your current best settings
CFG = TISConfig(
    K=32,
    n_trials=2000,
    seed=42,
    q_high=0.75,
    q_low=0.25,
    exclude_query_from_pools=True,
)

# Judge matrix preprocessing (this was the big mismatch driver)
J_MODE = "log1p_cp10k"  # "raw" or "log1p_cp10k" or "layer:<name>"
# =========================


def main():
    print("[1] Load AnnData")
    adata = sc.read_h5ad(ADATA_PATH)

    print("[2] Build judge matrix J")
    J = build_judge_matrix(adata, mode=J_MODE)
    print("  J shape:", J.shape, "mode:", J_MODE)

    print("[3] Extract CLS activations aligned by example_ids -> obs_names")
    A_aligned, has_row = extract_cls_from_shards_aligned(
        ACTS_ROOT,
        obs_names=adata.obs_names.tolist(),
        max_shards=MAX_SHARDS,
    )
    print("  A_aligned:", A_aligned.shape, "has_row:", int(has_row.sum()), "/", int(len(has_row)))

    A_use = A_aligned[has_row]
    J_use = J[has_row]
    print("  Using subset:", A_use.shape, getattr(J_use, "shape", None))

    # -------------------------
    # Seed reproducibility
    # -------------------------
    print("[4] Seed reproducibility")
    seeds = list(range(10))  # 0..9
    tis_mat, rep_stats = run_seed_reproducibility(A_use, J_use, base_cfg=CFG, seeds=seeds)

    np.save(os.path.join(OUTDIR, "tis_seed_repro_mat.npy"), tis_mat.astype(np.float32))
    save_json(os.path.join(OUTDIR, "tis_seed_repro_stats.json"), rep_stats)
    print("  repro:", rep_stats)

    # -------------------------
    # Lightweight grid search on K and n_trials
    # -------------------------
    print("[5] Lightweight grid (K x n_trials x seeds)")
    grid = TISGridConfig(
        seeds=[0, 1, 2],
        Ks=[15, 24, 32],
        n_trials_list=[400, 800, 1200, 2000],
        q_low=CFG.q_low,
        q_high=CFG.q_high,
        exclude_query_from_pools=CFG.exclude_query_from_pools,
        subsample_eval=None,  # set e.g. 5000 to speed up
    )

    rows = run_light_grid(A_use, J_use, grid=grid)
    df_grid = pd.DataFrame(rows)
    df_grid.to_csv(os.path.join(OUTDIR, "tis_grid_summary.csv"), index=False)

    agg = (
        df_grid.groupby(["K", "n_trials"], as_index=False)
        .agg(
            mean_mean=("mean", "mean"),
            mean_p90=("p90", "mean"),
            mean_p99=("p99", "mean"),
            mean_n_valid=("n_valid", "mean"),
            sd_mean=("mean", "std"),
        )
        .sort_values(["K", "n_trials"])
    )
    agg.to_csv(os.path.join(OUTDIR, "tis_grid_agg.csv"), index=False)
    print("  wrote tis_grid_summary.csv and tis_grid_agg.csv")

    print("[OK] wrote:", OUTDIR)


if __name__ == "__main__":
    main()