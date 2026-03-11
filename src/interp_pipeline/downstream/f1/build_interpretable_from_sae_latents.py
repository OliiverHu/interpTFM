from __future__ import annotations

import os
import numpy as np
import pandas as pd
import scanpy as sc

from interp_pipeline.tis.io import extract_cls_from_shards_aligned
from interp_pipeline.tis.io import encode_sae_latents  # adjust import if your encode lives elsewhere


# =========================
# EDIT HERE
# =========================
RUNS_ROOT = "runs/full_scgpt_cosmx"
LAYER = "layer_4.norm2"
SAE_THRESHOLD = 0.6  # which per_feature_best threshold row to use

BASE_ADATA = "/maiziezhou_lab2/yunfei/Projects/FM_temp/InterPLM/interplm/ge_shards/cosmx_human_lung_sec8.h5ad"
ACTS_ROOT = os.path.join(RUNS_ROOT, "activations", LAYER)

SAE_CKPT = os.path.join(RUNS_ROOT, "sae", LAYER, f"sae_{LAYER}.pt")
PER_FEATURE_BEST = os.path.join(RUNS_ROOT, "f1_analysis", "per_feature_best.csv")

OUTDIR = os.path.join(RUNS_ROOT, "interpretable_adata", LAYER, f"sae_thr_{SAE_THRESHOLD}")
os.makedirs(OUTDIR, exist_ok=True)

MAX_SHARDS = 512  # can be 20 for test, 512 for full

# Selection knobs
F1_CUTOFF = 0.60
TOPK_MAX: int | None = None  # optional cap after filtering/dedup
VAR_NAME_MODE = "term_name"   # "term_id" or "term_name"

# NEW: remove duplicate concepts by GO id
DEDUP_BY_CONCEPT = True
# =========================


def main():
    print("[1] Load base AnnData")
    adata = sc.read_h5ad(BASE_ADATA)
    print("  base:", adata.shape)

    print("[2] Extract aligned <CLS> activations")
    A_aligned, has_row = extract_cls_from_shards_aligned(
        ACTS_ROOT,
        obs_names=adata.obs_names.tolist(),
        max_shards=MAX_SHARDS,
    )
    n_ok = int(has_row.sum())
    print(f"  aligned CLS: {n_ok}/{adata.n_obs}")

    # keep only rows that exist
    adata_sub = adata[has_row].copy()
    A_use = A_aligned[has_row]
    print("  A_use:", A_use.shape, "adata_sub:", adata_sub.shape)

    print("[3] Encode SAE latents (cells × K)")
    if not os.path.exists(SAE_CKPT):
        raise FileNotFoundError(f"Missing SAE checkpoint: {SAE_CKPT}")
    Z = encode_sae_latents(A_use, SAE_CKPT, device="cuda", batch_size=8192)
    print("  Z:", Z.shape)

    print("[4] Load per_feature_best and select latents")
    best = pd.read_csv(PER_FEATURE_BEST)

    needed = {"layer", "threshold", "best_term_id", "best_f1"}
    missing = needed - set(best.columns)
    if missing:
        raise RuntimeError(f"{PER_FEATURE_BEST} missing columns {missing}. have={list(best.columns)}")

    # latent column can be 'latent' or 'feature' depending on earlier scripts
    latent_col = "latent" if "latent" in best.columns else ("feature" if "feature" in best.columns else None)
    if latent_col is None:
        raise RuntimeError(f"Cannot find latent column in {PER_FEATURE_BEST}. Have: {list(best.columns)}")

    best = best[(best["layer"] == LAYER) & (best["threshold"] == float(SAE_THRESHOLD))].copy()
    if best.empty:
        raise RuntimeError(f"No rows for layer={LAYER}, thr={SAE_THRESHOLD} in {PER_FEATURE_BEST}")

    # Ensure term name exists
    if "best_term_name" not in best.columns:
        best["best_term_name"] = best["best_term_id"].astype(str)

    # (A) best-only per latent (keep highest best_f1 row per latent)
    best = best.sort_values("best_f1", ascending=False)
    best = best.drop_duplicates(subset=[latent_col], keep="first").copy()
    if latent_col != "latent":
        best = best.rename(columns={latent_col: "latent"})

    # (B) NEW: dedupe by concept (GO id), keep best latent for each concept
    if DEDUP_BY_CONCEPT:
        best = best.sort_values("best_f1", ascending=False)
        best = best.drop_duplicates(subset=["best_term_id"], keep="first").copy()

    # (C) apply F1 cutoff + optional cap
    best = best[best["best_f1"] >= float(F1_CUTOFF)].copy()
    if TOPK_MAX is not None:
        best = best.sort_values("best_f1", ascending=False).head(int(TOPK_MAX)).copy()

    if best.empty:
        raise RuntimeError(f"No latents pass best_f1 >= {F1_CUTOFF}. Lower cutoff or change threshold.")

    print(
        "  selection summary:",
        f"rows={len(best)}",
        f"unique_latents={best['latent'].nunique()}",
        f"unique_terms={best['best_term_id'].nunique()}",
        f"(dedup_by_concept={DEDUP_BY_CONCEPT})",
    )

    # selected latent indices
    latents = best["latent"].astype(int).to_numpy()
    if latents.max() >= Z.shape[1]:
        raise RuntimeError(f"Selected latent id exceeds Z dims. max_latent={latents.max()} Z.shape={Z.shape}")

    Z_sel = Z[:, latents].astype(np.float32)
    print("  selected Z:", Z_sel.shape, f"(cutoff={F1_CUTOFF}, topk={TOPK_MAX})")

    # var_names: term id or name
    if VAR_NAME_MODE == "term_id":
        var_names = best["best_term_id"].astype(str).to_list()
    else:
        var_names = best["best_term_name"].fillna(best["best_term_id"]).astype(str).to_list()

    # AnnData requires unique var_names
    seen = {}
    uniq = []
    for v in var_names:
        if v not in seen:
            seen[v] = 0
            uniq.append(v)
        else:
            seen[v] += 1
            uniq.append(f"{v}__{seen[v]}")
    var_names = uniq

    print("[5] Create interpretable AnnData")
    out_var = best.rename(columns={"latent": "sae_latent"})[
        ["sae_latent", "best_f1", "best_term_id", "best_term_name", "layer", "threshold"]
    ].reset_index(drop=True)

    out = sc.AnnData(
        X=Z_sel,
        obs=adata_sub.obs.copy(),
        var=out_var,
    )
    out.var_names = var_names

    # keep spatial coords if present
    if "spatial" in adata_sub.obsm:
        out.obsm["spatial"] = adata_sub.obsm["spatial"].copy()

    # write manifest + h5ad
    manifest_path = os.path.join(
        OUTDIR, f"selected_latents_bestonly_conceptdedup_f1cut{F1_CUTOFF}_top{TOPK_MAX if TOPK_MAX else 'all'}.csv"
    )
    out.var.to_csv(manifest_path, index=False)

    out_path = os.path.join(
        OUTDIR,
        f"adata_interpretable_{LAYER}_saeThr{SAE_THRESHOLD}_bestonly_conceptdedup_f1cut{F1_CUTOFF}_top{TOPK_MAX if TOPK_MAX else 'all'}.h5ad",
    )
    out.write_h5ad(out_path)

    print("[OK] wrote:")
    print("  ", out_path)
    print("  ", manifest_path)


if __name__ == "__main__":
    main()