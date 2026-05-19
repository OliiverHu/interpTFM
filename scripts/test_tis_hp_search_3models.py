#!/usr/bin/env python3
from __future__ import annotations

"""
3-model TIS seed reproducibility + light grid.

This adapts the original scGPT seed/grid TIS script to:
  - scGPT: use token_id == 60695 as the cell-level special token
  - c2s-scale: mean-pool gene-token activations by example_id
  - Geneformer: use token_id == <cls>

It validates base model activations, not SAE features.

Outputs per model:
  <out-root>/<label>/<layer>/seed_grid/
    extraction_summary.json
    has_row.npy
    tis_seed_repro_mat.npy
    tis_seed_repro_stats.json
    tis_grid_summary.csv
    tis_grid_agg.csv

Combined:
  <out-root>/combined_tis_seed_repro_stats.csv
  <out-root>/combined_tis_grid_agg.csv

Example smoke test:
python test_tis_seed_grid_3models.py \
  --labels scgpt c2sscale geneformer \
  --store-roots \
    /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/full_scgpt_cosmx \
    /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/full_c2sscale_cosmx \
    /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/full_geneformer_cosmx \
  --layers layer_4.norm2 layer_17 layer_4 \
  --pooling token mean token \
  --token-values 60695 NONE '<cls>' \
  --adata-path /maiziezhou_lab2/yunfei/Projects/FM_temp/interGFM/ge_shards/cosmx_human_lung_sec8.h5ad \
  --out-root /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/tis_seed_grid_3models_smoke \
  --max-shards 3 \
  --seeds 0 1 2 \
  --grid-seeds 0 1 \
  --Ks 15 32 \
  --n-trials-list 400 800
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

from interp_pipeline.tis.core import TISConfig
from interp_pipeline.tis.io import build_judge_matrix
from interp_pipeline.tis.gr_seed import (
    run_seed_reproducibility,
    run_light_grid,
    TISGridConfig,
    save_json,
)


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
        raise RuntimeError(f"No aligned token rows found for token={token_value} under {acts_root}")

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

        if token_ids is None:
            token_ids = [None] * len(example_ids)

        mapped_indices: List[int] = []
        used_rows: List[int] = []

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


def summarize_repro_stats(rep_stats: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in rep_stats.items():
        if isinstance(v, (int, float, str, bool)):
            out[k] = v
    return out


def run_one_model(
    *,
    label: str,
    store_root: str,
    layer: str,
    pooling: str,
    token_value: Optional[str],
    adata,
    J,
    out_dir: Path,
    max_shards: Optional[int],
    example_id_key: str,
    exclude_tokens: Optional[List[str]],
    base_cfg: TISConfig,
    seeds: List[int],
    grid: TISGridConfig,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    ensure_dir(out_dir)

    print("=" * 100)
    print(f"[TIS seed/grid] {label} | {layer}")
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
    save_json(os.path.join(out_dir, "extraction_summary.json"), extraction_summary)

    A_use = A_aligned[has_row].astype(np.float32)
    J_use = J[has_row]
    print("  A_use:", A_use.shape)
    print("  J_use:", getattr(J_use, "shape", None))

    print("[seed reproducibility]")
    tis_mat, rep_stats = run_seed_reproducibility(A_use, J_use, base_cfg=base_cfg, seeds=seeds)
    np.save(out_dir / "tis_seed_repro_mat.npy", tis_mat.astype(np.float32))
    save_json(os.path.join(out_dir, "tis_seed_repro_stats.json"), rep_stats)

    print("[light grid]")
    rows = run_light_grid(A_use, J_use, grid=grid)
    df_grid = pd.DataFrame(rows)
    df_grid.insert(0, "model", label)
    df_grid.insert(1, "layer", layer)
    df_grid.insert(2, "pooling", pooling)
    df_grid.insert(3, "token_value", token_value if token_value is not None else "")
    df_grid.to_csv(out_dir / "tis_grid_summary.csv", index=False)

    agg = (
        df_grid.groupby(["model", "layer", "pooling", "token_value", "K", "n_trials"], as_index=False)
        .agg(
            mean_mean=("mean", "mean"),
            mean_p90=("p90", "mean"),
            mean_p99=("p99", "mean"),
            mean_n_valid=("n_valid", "mean"),
            sd_mean=("mean", "std"),
        )
        .sort_values(["model", "layer", "K", "n_trials"])
    )
    agg.to_csv(out_dir / "tis_grid_agg.csv", index=False)

    summary = {
        "model": label,
        "layer": layer,
        "pooling": pooling,
        "token_value": token_value if token_value is not None else "",
        "n_obs": int(adata.n_obs),
        "n_has_row": int(has_row.sum()),
        "activation_dim": int(A_use.shape[1]),
    }
    summary.update({f"extract_{k}": v for k, v in extraction_summary.items() if isinstance(v, (int, float, str, bool))})
    summary.update({f"repro_{k}": v for k, v in summarize_repro_stats(rep_stats).items()})
    save_json(os.path.join(out_dir, "summary_row.json"), summary)

    print("[OK]", summary)
    return summary, df_grid, agg


def main() -> None:
    ap = argparse.ArgumentParser(description="Run 3-model TIS seed reproducibility and light grid.")
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

    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--grid-seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--Ks", nargs="+", type=int, default=[15, 24, 32])
    ap.add_argument("--n-trials-list", nargs="+", type=int, default=[400, 800, 1200, 2000])

    args = ap.parse_args()

    max_shards = parse_optional_int(args.max_shards)
    out_root = ensure_dir(args.out_root)

    print("[1] Load AnnData")
    adata = sc.read_h5ad(args.adata_path)
    print("  adata:", adata.shape)

    print("[2] Build judge matrix J")
    J = build_judge_matrix(adata, mode=args.judge_mode)
    print("  J shape:", J.shape, "mode:", args.judge_mode)

    base_cfg = TISConfig(
        K=int(args.K),
        n_trials=int(args.n_trials),
        seed=int(args.seed),
        q_high=float(args.q_high),
        q_low=float(args.q_low),
        exclude_query_from_pools=True,
        subsample_eval=args.subsample_eval,
    )

    grid = TISGridConfig(
        seeds=list(args.grid_seeds),
        Ks=list(args.Ks),
        n_trials_list=list(args.n_trials_list),
        q_low=float(args.q_low),
        q_high=float(args.q_high),
        exclude_query_from_pools=True,
        subsample_eval=args.subsample_eval,
    )

    summaries = []
    grid_frames = []
    agg_frames = []

    for label, store_root, layer, pooling, token_raw in zip(
        args.labels,
        args.store_roots,
        args.layers,
        args.pooling,
        args.token_values,
    ):
        token_value = none_token(token_raw)
        model_out = out_root / label / layer.replace("/", "_") / "seed_grid"

        summary, df_grid, agg = run_one_model(
            label=label,
            store_root=store_root,
            layer=layer,
            pooling=pooling,
            token_value=token_value,
            adata=adata,
            J=J,
            out_dir=model_out,
            max_shards=max_shards,
            example_id_key=args.example_id_key,
            exclude_tokens=args.exclude_tokens,
            base_cfg=base_cfg,
            seeds=list(args.seeds),
            grid=grid,
        )
        summaries.append(summary)
        grid_frames.append(df_grid)
        agg_frames.append(agg)

    pd.DataFrame(summaries).to_csv(out_root / "combined_tis_seed_repro_stats.csv", index=False)
    pd.concat(grid_frames, ignore_index=True).to_csv(out_root / "combined_tis_grid_summary.csv", index=False)
    pd.concat(agg_frames, ignore_index=True).to_csv(out_root / "combined_tis_grid_agg.csv", index=False)

    print("\n[OK] wrote:", out_root)


if __name__ == "__main__":
    main()


# python test_tis_hp_search_3models.py \
#   --labels scgpt c2sscale geneformer \
#   --store-roots \
#     /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/full_scgpt_cosmx \
#     /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/full_c2sscale_cosmx \
#     /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/full_geneformer_cosmx \
#   --layers layer_4.norm2 layer_17 layer_4 \
#   --pooling token mean token \
#   --token-values 60695 NONE '<cls>' \
#   --adata-path /maiziezhou_lab2/yunfei/Projects/FM_temp/interGFM/ge_shards/cosmx_human_lung_sec8.h5ad \
#   --out-root /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/tis_seed_grid_3models \
#   --max-shards 3 \
#   --seeds 0 1 2 \
#   --grid-seeds 0 1 2 \
#   --Ks 15 24 32 \
#   --n-trials-list 400 800 1200 2000