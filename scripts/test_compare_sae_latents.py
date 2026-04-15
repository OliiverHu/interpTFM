#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch


DEFAULT_THRESHOLDS = [0.0, 0.15, 0.3, 0.6]


def load_state_dict(ckpt_path: Path) -> Dict[str, torch.Tensor]:
    obj = torch.load(ckpt_path, map_location="cpu")
    if isinstance(obj, dict):
        for k in ["state_dict", "model_state_dict", "model"]:
            if k in obj and isinstance(obj[k], dict):
                inner = obj[k]
                if all(torch.is_tensor(v) for v in inner.values()):
                    return inner
        if all(torch.is_tensor(v) for v in obj.values()):
            return obj
    raise TypeError(f"Unsupported checkpoint format in {ckpt_path}")


def get_encoder_params(state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    w = None
    b_enc = None
    x_bias = None

    for k in ["encoder.weight", "W_enc", "enc.weight"]:
        if k in state:
            w = state[k].float()
            break
    for k in ["encoder.bias", "b_enc", "enc.bias"]:
        if k in state:
            b_enc = state[k].float()
            break
    for k in ["bias", "x_bias"]:
        if k in state:
            x_bias = state[k].float()
            break

    if w is None:
        raise KeyError(f"Could not find encoder weights in checkpoint. First keys: {list(state.keys())[:30]}")
    return w, b_enc, x_bias


def encode_relu(x: torch.Tensor, w: torch.Tensor, b_enc: torch.Tensor | None, x_bias: torch.Tensor | None) -> torch.Tensor:
    if x_bias is not None:
        x = x - x_bias
    z = x @ w.T
    if b_enc is not None:
        z = z + b_enc
    return torch.relu(z)


def iter_activation_shards(store_root: Path, layer: str, max_shards: int | None = None):
    shard_dirs = sorted(Path(p) for p in glob.glob(str(store_root / "activations" / layer / "shard_*")))
    if max_shards is not None:
        shard_dirs = shard_dirs[:max_shards]
    for shard_dir in shard_dirs:
        act_path = shard_dir / "activations.pt"
        if act_path.exists():
            yield shard_dir, act_path


def summarize_model(
    label: str,
    sae_ckpt: Path,
    store_root: Path,
    layer: str,
    thresholds: List[float],
    max_shards: int,
    token_chunk_size: int,
) -> tuple[pd.DataFrame, Dict]:
    state = load_state_dict(sae_ckpt)
    w, b_enc, x_bias = get_encoder_params(state)
    n_latents, d_in = w.shape

    total_rows = 0
    shard_count = 0

    max_per_feat = torch.zeros(n_latents)
    sum_per_feat = torch.zeros(n_latents)
    sq_sum_per_feat = torch.zeros(n_latents)
    gt_counts = {thr: torch.zeros(n_latents, dtype=torch.long) for thr in thresholds}

    for shard_dir, act_path in iter_activation_shards(store_root, layer, max_shards=max_shards):
        x = torch.load(act_path, map_location="cpu").float()
        if x.ndim != 2:
            raise ValueError(f"{act_path} is not 2D")
        if x.shape[1] != d_in:
            raise ValueError(f"Input dim mismatch for {label}: activations {x.shape[1]} vs encoder {d_in}")

        shard_count += 1
        total_rows += x.shape[0]

        for start in range(0, x.shape[0], token_chunk_size):
            xb = x[start:start + token_chunk_size]
            z = encode_relu(xb, w, b_enc, x_bias)

            max_per_feat = torch.maximum(max_per_feat, z.max(dim=0).values)
            sum_per_feat += z.sum(dim=0)
            sq_sum_per_feat += (z ** 2).sum(dim=0)
            for thr in thresholds:
                gt_counts[thr] += (z > thr).sum(dim=0)

    if total_rows == 0:
        raise ValueError(f"No activation rows found for {label}")

    mean_per_feat = sum_per_feat / total_rows
    var_per_feat = (sq_sum_per_feat / total_rows) - mean_per_feat ** 2
    var_per_feat = torch.clamp(var_per_feat, min=0)
    std_per_feat = torch.sqrt(var_per_feat)

    rows = []
    for thr in thresholds:
        density = gt_counts[thr].float() / total_rows
        rows.append({
            "model": label,
            "threshold": thr,
            "total_rows": total_rows,
            "shards_scanned": shard_count,
            "n_latents": n_latents,
            "global_active_fraction": float(gt_counts[thr].sum().item() / (total_rows * n_latents)),
            "mean_feature_density": float(density.mean().item()),
            "median_feature_density": float(density.median().item()),
            "mean_active_features_per_example": float(gt_counts[thr].sum().item() / total_rows),
            "median_feature_activation": float(mean_per_feat.median().item()),
            "median_feature_std": float(std_per_feat.median().item()),
            "median_feature_max": float(max_per_feat.median().item()),
            "active_features_any": int((gt_counts[thr] > 0).sum().item()),
            "active_features_1pct": int((density > 0.01).sum().item()),
            "active_features_10pct": int((density > 0.10).sum().item()),
            "active_features_50pct": int((density > 0.50).sum().item()),
        })

    summary = {
        "model": label,
        "sae_ckpt": str(sae_ckpt),
        "store_root": str(store_root),
        "layer": layer,
        "n_latents": n_latents,
        "input_dim": d_in,
        "total_rows": total_rows,
        "shards_scanned": shard_count,
        "thresholds": thresholds,
        "median_feature_max": float(max_per_feat.median().item()),
        "max_feature_max": float(max_per_feat.max().item()),
    }
    return pd.DataFrame(rows), summary


def plot_metric(df: pd.DataFrame, metric: str, out_path: Path, ylabel: str, title: str):
    plt.figure(figsize=(8, 5))
    for model_name, sub in df.groupby("model"):
        sub = sub.sort_values("threshold")
        plt.plot(sub["threshold"], sub[metric], marker="o", label=model_name)
    plt.xticks(sorted(df["threshold"].unique()))
    plt.xlabel("Threshold")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Compare post-normalization SAE latent filtering across 3 models.")
    ap.add_argument("--labels", nargs=3, required=True, help="Model labels, e.g. scgpt c2sscale geneformer")
    ap.add_argument("--sae-ckpts", nargs=3, required=True, help="Paths to normalized SAE checkpoints")
    ap.add_argument("--store-roots", nargs=3, required=True, help="ActivationStore roots for the 3 models")
    ap.add_argument("--layers", nargs=3, required=True, help="Layer name for each model")
    ap.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=DEFAULT_THRESHOLDS,
        help="Thresholds to compare (default: 0.0 0.15 0.3 0.6)",
    )
    ap.add_argument("--max-shards", type=int, default=5, help="How many shards per model to scan")
    ap.add_argument("--token-chunk-size", type=int, default=8192, help="Chunk size for latent computation")
    ap.add_argument("--out-dir", type=Path, default=Path("runs/compare_sae_latents"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_frames = []
    summaries = []

    for label, ckpt, store_root, layer in zip(args.labels, args.sae_ckpts, args.store_roots, args.layers):
        print(f"[compare] model={label} layer={layer}")
        df, summary = summarize_model(
            label=label,
            sae_ckpt=Path(ckpt),
            store_root=Path(store_root),
            layer=layer,
            thresholds=list(args.thresholds),
            max_shards=args.max_shards,
            token_chunk_size=args.token_chunk_size,
        )
        all_frames.append(df)
        summaries.append(summary)

    res = pd.concat(all_frames, ignore_index=True)
    res_csv = args.out_dir / "latent_threshold_comparison.csv"
    res.to_csv(res_csv, index=False)

    with open(args.out_dir / "run_summary.json", "w") as f:
        json.dump(summaries, f, indent=2)

    plot_metric(
        res,
        metric="global_active_fraction",
        out_path=args.out_dir / "global_active_fraction_vs_threshold.png",
        ylabel="Global active fraction",
        title="Post-normalization latent activity vs threshold",
    )
    plot_metric(
        res,
        metric="mean_active_features_per_example",
        out_path=args.out_dir / "mean_active_features_per_example_vs_threshold.png",
        ylabel="Mean active features / example",
        title="Active latents per example vs threshold",
    )
    plot_metric(
        res,
        metric="median_feature_density",
        out_path=args.out_dir / "median_feature_density_vs_threshold.png",
        ylabel="Median feature density",
        title="Median feature density vs threshold",
    )
    plot_metric(
        res,
        metric="active_features_any",
        out_path=args.out_dir / "active_features_any_vs_threshold.png",
        ylabel="# features active at least once",
        title="Alive features vs threshold",
    )

    print("\nSaved:")
    print(" ", res_csv)
    for name in [
        "global_active_fraction_vs_threshold.png",
        "mean_active_features_per_example_vs_threshold.png",
        "median_feature_density_vs_threshold.png",
        "active_features_any_vs_threshold.png",
        "run_summary.json",
    ]:
        print(" ", args.out_dir / name)

    print("\nPreview:")
    print(res[[
        "model", "threshold", "global_active_fraction",
        "mean_active_features_per_example", "median_feature_density",
        "active_features_any"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()



# python test_compare_sae_latents.py \
#   --labels scgpt c2sscale geneformer \
#   --sae-ckpts \
#     runs/full_scgpt_cosmx/sae/layer_4.norm2/sae_layer_4.norm2_best.pt \
#     runs/full_c2sscale_cosmx/sae/layer_17/sae_layer_17_best.pt \
#     runs/full_geneformer_cosmx/sae/layer_4/sae_layer_4_best.pt \
#   --store-roots \
#     runs/full_scgpt_cosmx \
#     runs/full_c2sscale_cosmx \
#     runs/full_geneformer_cosmx \
#   --layers layer_4.norm2 layer_17 layer_4 \
#   --max-shards 5 \
#   --out-dir runs/compare_sae_latents