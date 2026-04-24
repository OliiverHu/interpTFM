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


def list_activation_shards(store_root: Path, layer: str) -> List[Path]:
    return sorted(Path(p) for p in glob.glob(str(store_root / "activations" / layer / "shard_*")))


def choose_shards(shard_dirs: List[Path], split: str, val_fraction: float) -> List[Path]:
    if split == "all":
        return shard_dirs
    n = len(shard_dirs)
    if n == 0:
        return []
    n_val = max(1, int(round(n * val_fraction)))
    val_shards = shard_dirs[-n_val:]
    train_shards = shard_dirs[:-n_val]
    if split == "train":
        return train_shards
    if split == "val":
        return val_shards
    raise ValueError(f"Unsupported split: {split}")


def audit_model(
    label: str,
    ckpt_path: Path,
    store_root: Path,
    layer: str,
    split: str,
    val_fraction: float,
    max_shards: int | None,
    token_chunk_size: int,
    dead_eps: float,
    near_dead_rate: float,
) -> tuple[pd.DataFrame, Dict]:
    state = load_state_dict(ckpt_path)
    w, b_enc, x_bias = get_encoder_params(state)
    n_latents, d_in = w.shape
    encoder_l2_norm = torch.linalg.vector_norm(w, ord=2, dim=1)

    shard_dirs = list_activation_shards(store_root, layer)
    shard_dirs = choose_shards(shard_dirs, split=split, val_fraction=val_fraction)
    if max_shards is not None:
        shard_dirs = shard_dirs[:max_shards]

    if not shard_dirs:
        raise RuntimeError(f"No shard dirs found for model={label}, layer={layer}, split={split}")

    active_counts = torch.zeros(n_latents, dtype=torch.long)
    max_per_feat = torch.zeros(n_latents)
    sum_per_feat = torch.zeros(n_latents)
    total_rows = 0

    for shard_dir in shard_dirs:
        act_path = shard_dir / "activations.pt"
        if not act_path.exists():
            continue
        x = torch.load(act_path, map_location="cpu").float()
        if x.ndim != 2:
            raise ValueError(f"{act_path} is not 2D")
        if x.shape[1] != d_in:
            raise ValueError(f"Input dim mismatch for {label}: activations {x.shape[1]} vs encoder {d_in}")

        total_rows += x.shape[0]

        for start in range(0, x.shape[0], token_chunk_size):
            xb = x[start:start + token_chunk_size]
            z = encode_relu(xb, w, b_enc, x_bias)

            active_counts += (z > dead_eps).sum(dim=0)
            max_per_feat = torch.maximum(max_per_feat, z.max(dim=0).values)
            sum_per_feat += z.sum(dim=0)

    if total_rows == 0:
        raise RuntimeError(f"No activation rows found for model={label}, layer={layer}, split={split}")

    active_rate = active_counts.float() / total_rows
    mean_activation = sum_per_feat / total_rows

    strict_dead = active_counts == 0
    near_dead = active_rate < near_dead_rate

    feat_df = pd.DataFrame({
        "feature": list(range(n_latents)),
        "active_count": active_counts.tolist(),
        "active_rate": active_rate.tolist(),
        "max_activation": max_per_feat.tolist(),
        "mean_activation": mean_activation.tolist(),
        "encoder_l2_norm": encoder_l2_norm.tolist(),
        "strict_dead": strict_dead.tolist(),
        "near_dead": near_dead.tolist(),
    })

    summary = {
        "model": label,
        "layer": layer,
        "split": split,
        "ckpt_path": str(ckpt_path),
        "store_root": str(store_root),
        "d_in": d_in,
        "n_latents": n_latents,
        "total_rows": total_rows,
        "num_shards": len(shard_dirs),
        "dead_eps": dead_eps,
        "near_dead_rate": near_dead_rate,
        "strict_dead_count": int(strict_dead.sum().item()),
        "strict_dead_frac": float(strict_dead.float().mean().item()),
        "near_dead_count": int(near_dead.sum().item()),
        "near_dead_frac": float(near_dead.float().mean().item()),
        "median_active_rate": float(active_rate.median().item()),
        "mean_active_rate": float(active_rate.mean().item()),
        "median_max_activation": float(max_per_feat.median().item()),
        "max_activation_max": float(max_per_feat.max().item()),
        "median_encoder_l2_norm": float(encoder_l2_norm.median().item()),
        "mean_encoder_l2_norm": float(encoder_l2_norm.mean().item()),
        "min_encoder_l2_norm": float(encoder_l2_norm.min().item()),
        "max_encoder_l2_norm": float(encoder_l2_norm.max().item()),
    }
    return feat_df, summary


def plot_hist(values, title: str, xlabel: str, out_path: Path, bins: int = 50, log_x: bool = False):
    plt.figure(figsize=(6, 4))
    x = values
    if log_x:
        x = [v for v in x if v > 0]
    plt.hist(x, bins=bins)
    if log_x:
        plt.xscale("log")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()




def plot_scatter(
    x_values,
    y_values,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    log_x: bool = False,
    log_y: bool = False,
):
    plt.figure(figsize=(6, 4))
    x = pd.Series(x_values, dtype="float64")
    y = pd.Series(y_values, dtype="float64")
    mask = x.notna() & y.notna()
    if log_x:
        mask &= x > 0
    if log_y:
        mask &= y > 0
    x = x[mask]
    y = y[mask]

    plt.scatter(x, y, s=8, alpha=0.35)
    if log_x:
        plt.xscale("log")
    if log_y:
        plt.yscale("log")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Audit dead / near-dead SAE neurons from saved checkpoints.")
    ap.add_argument("--labels", nargs=3, required=True)
    ap.add_argument("--sae-ckpts", nargs=3, required=True)
    ap.add_argument("--store-roots", nargs=3, required=True)
    ap.add_argument("--layers", nargs=3, required=True)

    ap.add_argument("--split", choices=["all", "train", "val"], default="all")
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--token-chunk-size", type=int, default=8192)
    ap.add_argument("--dead-eps", type=float, default=1e-8)
    ap.add_argument("--near-dead-rate", type=float, default=1e-4)
    ap.add_argument("--out-dir", type=Path, default=Path("runs/audit_dead_neurons"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_summary = []
    for label, ckpt, store_root, layer in zip(args.labels, args.sae_ckpts, args.store_roots, args.layers):
        print(f"[audit] model={label} layer={layer} split={args.split}")
        feat_df, summary = audit_model(
            label=label,
            ckpt_path=Path(ckpt),
            store_root=Path(store_root),
            layer=layer,
            split=args.split,
            val_fraction=args.val_fraction,
            max_shards=args.max_shards,
            token_chunk_size=args.token_chunk_size,
            dead_eps=args.dead_eps,
            near_dead_rate=args.near_dead_rate,
        )
        all_summary.append(summary)

        model_dir = args.out_dir / label
        model_dir.mkdir(parents=True, exist_ok=True)

        feat_csv = model_dir / f"feature_stats_{args.split}.csv"
        feat_df.to_csv(feat_csv, index=False)

        plot_hist(
            feat_df["active_rate"].tolist(),
            title=f"{label} active-rate histogram ({args.split})",
            xlabel="Feature active rate",
            out_path=model_dir / f"active_rate_hist_{args.split}.png",
            bins=50,
            log_x=False,
        )
        plot_hist(
            feat_df["encoder_l2_norm"].tolist(),
            title=f"{label} encoder L2-norm histogram ({args.split})",
            xlabel="Encoder weight L2 norm",
            out_path=model_dir / f"encoder_l2_norm_hist_{args.split}.png",
            bins=50,
            log_x=False,
        )
        plot_scatter(
            feat_df["active_rate"].tolist(),
            feat_df["encoder_l2_norm"].tolist(),
            title=f"{label} active rate vs encoder L2 norm ({args.split})",
            xlabel="Feature active rate",
            ylabel="Encoder weight L2 norm",
            out_path=model_dir / f"active_rate_vs_encoder_l2_norm_{args.split}.png",
            log_x=True,
            log_y=False,
        )
        plot_hist(
            feat_df["max_activation"].tolist(),
            title=f"{label} max-activation histogram ({args.split})",
            xlabel="Feature max activation",
            out_path=model_dir / f"max_activation_hist_{args.split}.png",
            bins=50,
            log_x=True,
        )

    summary_df = pd.DataFrame(all_summary)
    summary_df.to_csv(args.out_dir / f"dead_neuron_summary_{args.split}.csv", index=False)

    with open(args.out_dir / f"dead_neuron_summary_{args.split}.json", "w") as f:
        json.dump(all_summary, f, indent=2)

    print("\nSummary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()


# python audit_dead_neurons_3models.py \
#   --labels scgpt c2sscale geneformer \
#   --sae-ckpts \
#     /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/full_scgpt_cosmx/sae/layer_4.norm2/nr_on__steps_6000__l1_3e-3/sae_layer_4.norm2_best.pt \
#     /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/full_c2sscale_cosmx/sae/layer_17/nr_on__steps_6000__l1_3e-3/sae_layer_17_best.pt \
#     /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/full_geneformer_cosmx/sae/layer_4/nr_on__steps_6000__l1_3e-3/sae_layer_4_best.pt \
#   --store-roots \
#     ../runs/full_scgpt_cosmx \
#     ../runs/full_c2sscale_cosmx \
#     ../runs/full_geneformer_cosmx \
#   --layers layer_4.norm2 layer_17 layer_4 \
#   --split all \
#   --max-shards 5 \
#   --near-dead-rate 1e-4 \
#   --out-dir /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/audit_dead_neurons_l1_3e-3