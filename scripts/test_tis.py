from __future__ import annotations

import os
import json
import numpy as np
import scanpy as sc

from interp_pipeline.tis.core import (
    TISConfig,
    build_pools_quantile,
    compute_tis_mis,
    shuffle_activations,
    pca_activations,
)
from interp_pipeline.tis.io import (
    load_expression_from_adata,
    extract_cls_from_shards_aligned,
    build_judge_matrix
)

# =========================
# HARD-CODE CONFIG HERE
# =========================
RUNS_ROOT = "runs/full_scgpt_cosmx"
LAYER = "layer_4.norm2"

ADATA_PATH = "/maiziezhou_lab2/yunfei/Projects/FM_temp/InterPLM/interplm/ge_shards/cosmx_human_lung_sec8.h5ad"
ACTS_ROOT = os.path.join(RUNS_ROOT, "activations", LAYER)

OUTDIR = os.path.join(RUNS_ROOT, "tis", LAYER)
os.makedirs(OUTDIR, exist_ok=True)

MAX_SHARDS = 50

CFG = TISConfig(K=32, n_trials=2000, seed=42, q_high=0.75, q_low=0.25, exclude_query_from_pools=True)
# =========================


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def summarize(arr: np.ndarray) -> dict:
    arr = arr.astype(np.float32)
    good = np.isfinite(arr)
    return {
        "mean": float(np.nanmean(arr)),
        "median": float(np.nanmedian(arr)),
        "p90": float(np.nanpercentile(arr[good], 90)) if good.any() else float("nan"),
        "n_valid": int(good.sum()),
        "shape": list(arr.shape),
    }


def main():
    print("[1] Load AnnData")
    adata = sc.read_h5ad(ADATA_PATH)
    # J = load_expression_from_adata(adata)  # sparse OK
    # print("  J shape:", J.shape)

    print("[2] Extract scGPT <CLS> activations (aligned to adata.obs via example_ids)")
    A_cls_aligned, has_row = extract_cls_from_shards_aligned(
        ACTS_ROOT,
        obs_names=adata.obs_names.tolist(),
        max_shards=MAX_SHARDS,
    )
    print("  A_cls_aligned shape:", A_cls_aligned.shape, "has_row:", int(has_row.sum()), "/", int(len(has_row)))

    # Restrict to rows with CLS extracted (critical!)
    A_use = A_cls_aligned[has_row]

    J = build_judge_matrix(adata, mode="log1p_cp10k")
    J_use = J[has_row]
    print("  J shape:", J.shape)
    print("  Using aligned subset:", A_use.shape, "J_use:", getattr(J_use, "shape", None))

    # TIS on aligned CLS
    print("[3] TIS on scGPT <CLS> space (aligned)")
    top_idx, bot_idx, med = build_pools_quantile(A_use, CFG.q_low, CFG.q_high)
    tis_cls = compute_tis_mis(A_use, J_use, top_idx, bot_idx, med, CFG)

    np.save(os.path.join(OUTDIR, "tis_scgpt_cls.npy"), tis_cls.astype(np.float32))
    save_json(os.path.join(OUTDIR, "tis_scgpt_cls_summary.json"), summarize(tis_cls))

    # Baselines
    print("[4] Baselines: shuffled + PCA (aligned)")
    Ash = shuffle_activations(A_use, seed=CFG.seed)
    top_idx_s, bot_idx_s, med_s = build_pools_quantile(Ash, CFG.q_low, CFG.q_high)
    tis_shuf = compute_tis_mis(Ash, J_use, top_idx_s, bot_idx_s, med_s, CFG)
    np.save(os.path.join(OUTDIR, "tis_scgpt_cls_shuffled.npy"), tis_shuf.astype(np.float32))
    save_json(os.path.join(OUTDIR, "tis_scgpt_cls_shuffled_summary.json"), summarize(tis_shuf))

    Ap = pca_activations(A_use, seed=CFG.seed)
    top_idx_p, bot_idx_p, med_p = build_pools_quantile(Ap, CFG.q_low, CFG.q_high)
    tis_pca = compute_tis_mis(Ap, J_use, top_idx_p, bot_idx_p, med_p, CFG)
    np.save(os.path.join(OUTDIR, "tis_scgpt_cls_pca.npy"), tis_pca.astype(np.float32))
    save_json(os.path.join(OUTDIR, "tis_scgpt_cls_pca_summary.json"), summarize(tis_pca))

    print("[OK] wrote:", OUTDIR)


if __name__ == "__main__":
    main()