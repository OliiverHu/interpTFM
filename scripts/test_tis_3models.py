#!/usr/bin/env python3
from __future__ import annotations

"""
3-model TIS/MIS validation on model activations.

This generalizes the original scGPT-only scripts/test_tis.py to:
  - scGPT: use token_id == 60695 as cell-level special token
  - c2s-scale: mean-pool gene-token activations per example_id
  - Geneformer: use token_id == <cls>

This validates base model activations, not SAE features.

Outputs per model:
  <out-root>/<label>/<layer>/
    has_row.npy
    tis_model.npy
    tis_model_summary.json
    tis_model_shuffled.npy
    tis_model_shuffled_summary.json
    tis_model_pca.npy
    tis_model_pca_summary.json
    extraction_summary.json

Combined:
  <out-root>/combined_summary.csv
  <out-root>/plots_cross_model/tis_summary_bar.png
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from tqdm import tqdm

from interp_pipeline.tis.core import (
    TISConfig,
    build_pools_quantile,
    compute_tis_mis,
    shuffle_activations,
    pca_activations,
)
from interp_pipeline.tis.io import build_judge_matrix


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def parse_optional_int(x: str | None) -> Optional[int]:
    if x is None:
        return None
    if str(x).strip().lower() in {"none", "null", "na", ""}:
        return None
    return int(x)


def none_token(x: str | None) -> Optional[str]:
    if x is None:
        return None
    if str(x).strip().lower() in {"none", "null", "na", ""}:
        return None
    return str(x)


def summarize(arr: np.ndarray) -> Dict[str, Any]:
    arr = arr.astype(np.float32)
    good = np.isfinite(arr)
    if not good.any():
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "n_valid": 0,
            "shape": list(arr.shape),
        }
    return {
        "mean": float(np.nanmean(arr)),
        "median": float(np.nanmedian(arr)),
        "p90": float(np.nanpercentile(arr[good], 90)),
        "p95": float(np.nanpercentile(arr[good], 95)),
        "n_valid": int(good.sum()),
        "shape": list(arr.shape),
    }


def save_json(path: str | Path, obj: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def list_shards(acts_root: str, max_shards: Optional[int]) -> List[str]:
    import glob
    shards = sorted([p for p in glob.glob(os.path.join(acts_root, "shard_*")) if os.path.isdir(p)])
    if max_shards is not None:
        shards = shards[: int(max_shards)]
    if not shards:
        raise RuntimeError(f"No shards under {acts_root}")
    return shards


def resolve_example_index(ex_raw: Any, obs_index: Dict[str, int], n_obs: int) -> Optional[int]:
    ex_str = str(ex_raw)
    if ex_str.strip().isdigit():
        idx = int(ex_str)
        if 0 <= idx < n_obs:
            return idx
    idx = obs_index.get(ex_str, None)
    if idx is not None and 0 <= int(idx) < n_obs:
        return int(idx)
    return None


def extract_token_aligned(
    acts_root: str,
    *,
    obs_names: List[str],
    token_value: str,
    max_shards: Optional[int],
    example_id_key: str = "example_ids",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Select rows where token_ids == token_value and align to adata.obs order."""
    shards = list_shards(acts_root, max_shards=max_shards)
    n_obs = len(obs_names)
    obs_index = {str(k): i for i, k in enumerate(obs_names)}

    A_aligned = None
    has_row = np.zeros((n_obs,), dtype=bool)

    n_rows_seen = 0
    n_token_rows = 0
    n_aligned = 0
    n_unmatched = 0
    n_duplicate = 0
    H = None

    token_value = str(token_value)

    for shard in tqdm(shards, desc=f"extract_token:{token_value}"):
        acts_path = os.path.join(shard, "activations.pt")
        idx_path = os.path.join(shard, "index.pt")
        if not (os.path.exists(acts_path) and os.path.exists(idx_path)):
            continue

        X = torch.load(acts_path, map_location="cpu")
        idx = torch.load(idx_path, map_location="cpu")
        token_ids = idx.get("token_ids", None)
        example_ids = idx.get(example_id_key, None)

        if token_ids is None or example_ids is None:
            continue
        if len(token_ids) != X.shape[0] or len(example_ids) != X.shape[0]:
            continue

        if A_aligned is None:
            H = int(X.shape[1])
            A_aligned = np.zeros((n_obs, H), dtype=np.float32)

        n_rows_seen += int(X.shape[0])
        rows = [i for i, t in enumerate(token_ids) if str(t) == token_value]
        n_token_rows += len(rows)

        for r in rows:
            ex_idx = resolve_example_index(example_ids[r], obs_index, n_obs)
            if ex_idx is None:
                n_unmatched += 1
                continue
            if has_row[ex_idx]:
                n_duplicate += 1
            A_aligned[ex_idx] = X[r].detach().cpu().numpy().astype(np.float32)
            has_row[ex_idx] = True
            n_aligned += 1

    if A_aligned is None or not has_row.any():
        raise RuntimeError(
            f"No aligned rows found for token={token_value} under {acts_root}. "
            f"Check token_ids and example_ids."
        )

    summary = {
        "pooling": "token",
        "token_value": token_value,
        "n_shards": int(len(shards)),
        "n_rows_seen": int(n_rows_seen),
        "n_token_rows": int(n_token_rows),
        "n_aligned_assignments": int(n_aligned),
        "n_unmatched": int(n_unmatched),
        "n_duplicate_assignments": int(n_duplicate),
        "n_obs": int(n_obs),
        "n_has_row": int(has_row.sum()),
        "H": int(H),
    }
    return A_aligned, has_row, summary


def extract_mean_aligned(
    acts_root: str,
    *,
    obs_names: List[str],
    max_shards: Optional[int],
    example_id_key: str = "example_ids",
    exclude_tokens: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Mean-pool token rows by example_id and align to adata.obs order."""
    shards = list_shards(acts_root, max_shards=max_shards)
    n_obs = len(obs_names)
    obs_index = {str(k): i for i, k in enumerate(obs_names)}
    exclude = set(str(x) for x in (exclude_tokens or []))

    A_sum = None
    counts = np.zeros((n_obs,), dtype=np.int64)

    n_rows_seen = 0
    n_rows_used = 0
    n_unmatched = 0
    H = None

    for shard in tqdm(shards, desc="extract_mean"):
        acts_path = os.path.join(shard, "activations.pt")
        idx_path = os.path.join(shard, "index.pt")
        if not (os.path.exists(acts_path) and os.path.exists(idx_path)):
            continue

        X = torch.load(acts_path, map_location="cpu")
        idx = torch.load(idx_path, map_location="cpu")
        example_ids = idx.get(example_id_key, None)
        token_ids = idx.get("token_ids", None)

        if example_ids is None:
            continue
        if len(example_ids) != X.shape[0]:
            continue

        if A_sum is None:
            H = int(X.shape[1])
            A_sum = np.zeros((n_obs, H), dtype=np.float64)

        n_rows_seen += int(X.shape[0])

        mapped_indices: List[int] = []
        used_rows: List[int] = []

        if token_ids is None:
            token_ids = [None] * len(example_ids)

        for r, (ex_raw, tok) in enumerate(zip(example_ids, token_ids)):
            if tok is not None and str(tok) in exclude:
                continue
            ex_idx = resolve_example_index(ex_raw, obs_index, n_obs)
            if ex_idx is None:
                n_unmatched += 1
                continue
            mapped_indices.append(ex_idx)
            used_rows.append(r)

        if not used_rows:
            continue

        X_np = X[used_rows].detach().cpu().numpy().astype(np.float32)
        idx_np = np.asarray(mapped_indices, dtype=np.int64)

        np.add.at(A_sum, idx_np, X_np.astype(np.float64))
        np.add.at(counts, idx_np, 1)

        n_rows_used += int(len(used_rows))

    if A_sum is None or counts.sum() == 0:
        raise RuntimeError(f"No rows mean-pooled under {acts_root}")

    has_row = counts > 0
    A_aligned = np.zeros((n_obs, int(H)), dtype=np.float32)
    A_aligned[has_row] = (A_sum[has_row] / counts[has_row, None]).astype(np.float32)

    summary = {
        "pooling": "mean",
        "exclude_tokens": sorted(exclude),
        "n_shards": int(len(shards)),
        "n_rows_seen": int(n_rows_seen),
        "n_rows_used": int(n_rows_used),
        "n_unmatched": int(n_unmatched),
        "n_obs": int(n_obs),
        "n_has_row": int(has_row.sum()),
        "H": int(H),
        "counts_min": int(counts[has_row].min()) if has_row.any() else 0,
        "counts_median": float(np.median(counts[has_row])) if has_row.any() else 0.0,
        "counts_max": int(counts[has_row].max()) if has_row.any() else 0,
    }
    return A_aligned, has_row, summary


def extract_cell_activations_aligned(
    *,
    store_root: str,
    layer: str,
    obs_names: List[str],
    pooling: str,
    token_value: Optional[str],
    max_shards: Optional[int],
    example_id_key: str,
    exclude_tokens: Optional[List[str]],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    acts_root = os.path.join(store_root, "activations", layer)

    if pooling == "token":
        if token_value is None:
            raise ValueError("pooling='token' requires token_value")
        return extract_token_aligned(
            acts_root,
            obs_names=obs_names,
            token_value=token_value,
            max_shards=max_shards,
            example_id_key=example_id_key,
        )

    if pooling == "mean":
        return extract_mean_aligned(
            acts_root,
            obs_names=obs_names,
            max_shards=max_shards,
            example_id_key=example_id_key,
            exclude_tokens=exclude_tokens,
        )

    raise ValueError(f"Unknown pooling={pooling}. Use token or mean.")


def run_tis_one_model(
    *,
    label: str,
    store_root: str,
    layer: str,
    pooling: str,
    token_value: Optional[str],
    adata,
    out_dir: Path,
    max_shards: Optional[int],
    example_id_key: str,
    exclude_tokens: Optional[List[str]],
    judge_mode: str,
    cfg: TISConfig,
    save_cell_activations: bool,
    run_pca: bool,
) -> Dict[str, Any]:
    ensure_dir(out_dir)

    print("=" * 100)
    print(f"[TIS] {label} | {layer}")
    print(f"  store_root={store_root}")
    print(f"  pooling={pooling}")
    print(f"  token_value={token_value}")
    print(f"  out_dir={out_dir}")
    print("=" * 100)

    A_aligned, has_row, extraction_summary = extract_cell_activations_aligned(
        store_root=store_root,
        layer=layer,
        obs_names=list(map(str, adata.obs_names.tolist())),
        pooling=pooling,
        token_value=token_value,
        max_shards=max_shards,
        example_id_key=example_id_key,
        exclude_tokens=exclude_tokens,
    )

    np.save(out_dir / "has_row.npy", has_row)
    save_json(out_dir / "extraction_summary.json", extraction_summary)

    if save_cell_activations:
        np.save(out_dir / "activations_cell_aligned.npy", A_aligned.astype(np.float32))

    A_use = A_aligned[has_row].astype(np.float32)
    print("  A_aligned:", A_aligned.shape, "has_row:", int(has_row.sum()), "/", len(has_row))
    print("  A_use:", A_use.shape)

    print("[judge] building expression-derived judge matrix:", judge_mode)
    J = build_judge_matrix(adata, mode=judge_mode)
    J_use = J[has_row]
    print("  J_use:", getattr(J_use, "shape", None))

    print("[TIS] model activations")
    top_idx, bot_idx, med = build_pools_quantile(A_use, cfg.q_low, cfg.q_high)
    tis = compute_tis_mis(A_use, J_use, top_idx, bot_idx, med, cfg)
    np.save(out_dir / "tis_model.npy", tis.astype(np.float32))
    tis_summary = summarize(tis)
    save_json(out_dir / "tis_model_summary.json", tis_summary)

    print("[baseline] shuffled activations")
    Ash = shuffle_activations(A_use, seed=cfg.seed)
    top_idx_s, bot_idx_s, med_s = build_pools_quantile(Ash, cfg.q_low, cfg.q_high)
    tis_shuf = compute_tis_mis(Ash, J_use, top_idx_s, bot_idx_s, med_s, cfg)
    np.save(out_dir / "tis_model_shuffled.npy", tis_shuf.astype(np.float32))
    shuf_summary = summarize(tis_shuf)
    save_json(out_dir / "tis_model_shuffled_summary.json", shuf_summary)

    pca_summary = {}
    if run_pca:
        print("[baseline] PCA activations")
        Ap = pca_activations(A_use, seed=cfg.seed)
        top_idx_p, bot_idx_p, med_p = build_pools_quantile(Ap, cfg.q_low, cfg.q_high)
        tis_pca = compute_tis_mis(Ap, J_use, top_idx_p, bot_idx_p, med_p, cfg)
        np.save(out_dir / "tis_model_pca.npy", tis_pca.astype(np.float32))
        pca_summary = summarize(tis_pca)
        save_json(out_dir / "tis_model_pca_summary.json", pca_summary)

    row = {
        "model": label,
        "layer": layer,
        "pooling": pooling,
        "token_value": token_value if token_value is not None else "",
        "n_obs": int(adata.n_obs),
        "n_has_row": int(has_row.sum()),
        "activation_dim": int(A_use.shape[1]),
        "tis_mean": tis_summary["mean"],
        "tis_median": tis_summary["median"],
        "tis_p90": tis_summary["p90"],
        "tis_n_valid": tis_summary["n_valid"],
        "shuffle_mean": shuf_summary["mean"],
        "shuffle_median": shuf_summary["median"],
        "shuffle_p90": shuf_summary["p90"],
    }
    if pca_summary:
        row.update(
            {
                "pca_mean": pca_summary["mean"],
                "pca_median": pca_summary["median"],
                "pca_p90": pca_summary["p90"],
            }
        )
    row.update({f"extract_{k}": v for k, v in extraction_summary.items() if isinstance(v, (int, float, str, bool))})

    save_json(out_dir / "summary_row.json", row)
    print("[OK]", row)
    return row


def plot_combined_summary(summary_df: pd.DataFrame, out_root: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print("[warn] matplotlib unavailable, skipping plots:", e)
        return

    plot_dir = ensure_dir(out_root / "plots_cross_model")
    x = np.arange(len(summary_df))
    width = 0.25

    plt.figure(figsize=(8, 4.5))
    plt.bar(x - width, summary_df["tis_mean"].astype(float), width, label="model")
    plt.bar(x, summary_df["shuffle_mean"].astype(float), width, label="shuffle")
    if "pca_mean" in summary_df.columns:
        plt.bar(x + width, summary_df["pca_mean"].astype(float), width, label="PCA")
    plt.xticks(x, summary_df["model"].astype(str).tolist())
    plt.ylabel("Mean TIS/MIS")
    plt.title("TIS validation across model activations")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "tis_summary_bar.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.bar(x, summary_df["tis_mean"].astype(float) - summary_df["shuffle_mean"].astype(float))
    plt.xticks(x, summary_df["model"].astype(str).tolist())
    plt.ylabel("Mean TIS - shuffled")
    plt.title("TIS improvement over shuffled baseline")
    plt.tight_layout()
    plt.savefig(plot_dir / "tis_delta_vs_shuffle.png", dpi=200)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Run 3-model TIS/MIS validation on model activations.")
    ap.add_argument("--labels", nargs=3, required=True)
    ap.add_argument("--store-roots", nargs=3, required=True)
    ap.add_argument("--layers", nargs=3, required=True)
    ap.add_argument("--pooling", nargs=3, required=True, choices=["token", "mean"])
    ap.add_argument("--token-values", nargs=3, required=True, help="Use NONE for mean pooling.")

    ap.add_argument("--adata-path", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--max-shards", default="NONE", help="Integer or NONE")
    ap.add_argument("--example-id-key", default="example_ids")
    ap.add_argument("--exclude-tokens", nargs="*", default=["<cls>", "<eos>"])

    ap.add_argument("--judge-mode", default="log1p_cp10k")

    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--n-trials", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--q-high", type=float, default=0.75)
    ap.add_argument("--q-low", type=float, default=0.25)
    ap.add_argument("--subsample-eval", type=int, default=None)

    ap.add_argument("--no-pca", action="store_true")
    ap.add_argument("--save-cell-activations", action="store_true")

    args = ap.parse_args()

    max_shards = parse_optional_int(args.max_shards)
    out_root = ensure_dir(args.out_root)

    print("[load] AnnData:", args.adata_path)
    adata = sc.read_h5ad(args.adata_path)
    print("  n_obs:", adata.n_obs, "n_vars:", adata.n_vars)

    cfg = TISConfig(
        K=int(args.K),
        n_trials=int(args.n_trials),
        seed=int(args.seed),
        q_high=float(args.q_high),
        q_low=float(args.q_low),
        exclude_query_from_pools=True,
        subsample_eval=args.subsample_eval,
    )

    rows = []
    for label, store_root, layer, pooling, token_value_raw in zip(
        args.labels,
        args.store_roots,
        args.layers,
        args.pooling,
        args.token_values,
    ):
        token_value = none_token(token_value_raw)
        model_out = out_root / label / layer.replace("/", "_")

        row = run_tis_one_model(
            label=label,
            store_root=store_root,
            layer=layer,
            pooling=pooling,
            token_value=token_value,
            adata=adata,
            out_dir=model_out,
            max_shards=max_shards,
            example_id_key=args.example_id_key,
            exclude_tokens=args.exclude_tokens,
            judge_mode=args.judge_mode,
            cfg=cfg,
            save_cell_activations=args.save_cell_activations,
            run_pca=not args.no_pca,
        )
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_root / "combined_summary.csv", index=False)
    plot_combined_summary(summary_df, out_root)

    print("\n[OK] wrote:", out_root)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()



# python test_tis_3models.py \
#   --labels scgpt c2sscale geneformer \
#   --store-roots \
#     /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/full_scgpt_cosmx \
#     /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/full_c2sscale_cosmx \
#     /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/full_geneformer_cosmx \
#   --layers layer_4.norm2 layer_17 layer_4 \
#   --pooling token mean token \
#   --token-values 60695 NONE '<cls>' \
#   --adata-path /maiziezhou_lab2/yunfei/Projects/FM_temp/interGFM/ge_shards/cosmx_human_lung_sec8.h5ad \
#   --out-root /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/tis_3models \
#   --max-shards NONE \
#   --K 32 \
#   --n-trials 800 \
#   --seed 42

# For scGPT and Geneformer, TIS was computed on cell-level special-token activations.
# For c2s-scale, which exposes gene-token activations without a CLS token in the ActivationStore, we used mean-pooled gene-token activations per cell.