#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch


def load_state_dict(ckpt_path: Path) -> Dict[str, torch.Tensor]:
    obj = torch.load(ckpt_path, map_location="cpu")
    if isinstance(obj, dict):
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
        for k in ["state_dict", "model_state_dict", "model"]:
            if k in obj and isinstance(obj[k], dict):
                inner = obj[k]
                if all(isinstance(v, torch.Tensor) for v in inner.values()):
                    return inner
    raise TypeError(f"Unsupported checkpoint format in {ckpt_path}")


def get_encoder_params(state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor | None]:
    weight = None
    bias = None
    for k in ["encoder.weight", "W_enc", "enc.weight"]:
        if k in state:
            weight = state[k]
            break
    for k in ["encoder.bias", "b_enc", "enc.bias"]:
        if k in state:
            bias = state[k]
            break
    if weight is None:
        raise KeyError(f"Could not find encoder weights. Keys include: {list(state.keys())[:30]}")
    return weight.float(), None if bias is None else bias.float()


def encode_relu(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None) -> torch.Tensor:
    z = x @ w.T
    if b is not None:
        z = z + b
    return torch.relu(z)


def iter_activation_shards(store_root: Path, layer: str, max_shards: int | None = None):
    shard_dirs = sorted(Path(p) for p in glob.glob(str(store_root / "activations" / layer / "shard_*")))
    if max_shards is not None:
        shard_dirs = shard_dirs[:max_shards]
    for shard_dir in shard_dirs:
        act_path = shard_dir / "activations.pt"
        idx_path = shard_dir / "index.pt"
        if act_path.exists() and idx_path.exists():
            yield shard_dir, act_path, idx_path


def inspect_latents(
    sae_ckpt: Path,
    store_root: Path,
    layer: str,
    max_shards: int,
    token_chunk_size: int,
    topk: int,
):
    state = load_state_dict(sae_ckpt)
    w, b = get_encoder_params(state)

    n_latents, d_in = w.shape
    print("=== SAE CHECKPOINT ===")
    print(f"ckpt         : {sae_ckpt}")
    print(f"encoder shape: {tuple(w.shape)}")
    print(f"bias present : {b is not None}")

    total_rows = 0
    shard_count = 0

    max_per_feat = torch.zeros(n_latents)
    mean_acc = torch.zeros(n_latents)
    sq_acc = torch.zeros(n_latents)
    nonzero_count = torch.zeros(n_latents, dtype=torch.long)
    gt1_count = torch.zeros(n_latents, dtype=torch.long)
    gt01_count = torch.zeros(n_latents, dtype=torch.long)

    global_min = None
    global_max = None
    first_rows_examples = []

    for shard_dir, act_path, idx_path in iter_activation_shards(store_root, layer, max_shards=max_shards):
        x = torch.load(act_path, map_location="cpu").float()
        idx = torch.load(idx_path, map_location="cpu")
        if x.ndim != 2:
            raise ValueError(f"{act_path} is not 2D")
        if x.shape[1] != d_in:
            raise ValueError(f"Input dim mismatch: activations {x.shape[1]} vs encoder {d_in}")

        example_ids = idx.get("example_ids", [])
        token_ids = idx.get("token_ids", [])

        shard_count += 1
        total_rows += x.shape[0]

        for start in range(0, x.shape[0], token_chunk_size):
            xb = x[start : start + token_chunk_size]
            zb = encode_relu(xb, w, b)

            cur_min = float(zb.min().item())
            cur_max = float(zb.max().item())
            global_min = cur_min if global_min is None else min(global_min, cur_min)
            global_max = cur_max if global_max is None else max(global_max, cur_max)

            max_per_feat = torch.maximum(max_per_feat, zb.max(dim=0).values)
            mean_acc += zb.sum(dim=0)
            sq_acc += (zb ** 2).sum(dim=0)
            nonzero_count += (zb > 0).sum(dim=0)
            gt1_count += (zb > 1.0).sum(dim=0)
            gt01_count += (zb > 0.1).sum(dim=0)

            if len(first_rows_examples) < 10:
                take = min(10 - len(first_rows_examples), zb.shape[0])
                for i in range(take):
                    row_idx = start + i
                    vals, inds = torch.topk(zb[i], k=min(8, zb.shape[1]))
                    first_rows_examples.append(
                        {
                            "row_in_shard": row_idx,
                            "cell": str(example_ids[row_idx]) if row_idx < len(example_ids) else None,
                            "gene": str(token_ids[row_idx]) if row_idx < len(token_ids) else None,
                            "latent_max": float(zb[i].max().item()),
                            "latent_nnz": int((zb[i] > 0).sum().item()),
                            "top_latents": inds.tolist(),
                            "top_values": [float(v) for v in vals.tolist()],
                        }
                    )

    if total_rows == 0:
        raise ValueError("No activation rows found.")

    mean_per_feat = mean_acc / total_rows
    var_per_feat = (sq_acc / total_rows) - mean_per_feat ** 2
    var_per_feat = torch.clamp(var_per_feat, min=0)
    std_per_feat = torch.sqrt(var_per_feat)
    density_per_feat = nonzero_count.float() / total_rows

    best_max_vals, best_max_idx = torch.topk(max_per_feat, k=min(topk, n_latents))
    best_mean_vals, best_mean_idx = torch.topk(mean_per_feat, k=min(topk, n_latents))
    best_density_vals, best_density_idx = torch.topk(density_per_feat, k=min(topk, n_latents))

    summary = {
        "layer": layer,
        "total_rows": total_rows,
        "shards_scanned": shard_count,
        "n_latents": int(n_latents),
        "input_dim": int(d_in),
        "global_latent_min": global_min,
        "global_latent_max": global_max,
        "mean_of_feature_means": float(mean_per_feat.mean().item()),
        "median_feature_mean": float(mean_per_feat.median().item()),
        "mean_of_feature_stds": float(std_per_feat.mean().item()),
        "median_feature_std": float(std_per_feat.median().item()),
        "mean_density": float(density_per_feat.mean().item()),
        "median_density": float(density_per_feat.median().item()),
        "dead_features": int((nonzero_count == 0).sum().item()),
        "almost_dead_features_density_lt_1e-4": int((density_per_feat < 1e-4).sum().item()),
        "sparse_features_density_lt_1pct": int((density_per_feat < 0.01).sum().item()),
        "dense_features_density_gt_50pct": int((density_per_feat > 0.5).sum().item()),
        "top_features_by_max": [
            {"feature": int(i), "max": float(v)}
            for i, v in zip(best_max_idx.tolist(), best_max_vals.tolist())
        ],
        "top_features_by_mean": [
            {
                "feature": int(i),
                "mean": float(v),
                "std": float(std_per_feat[i].item()),
                "density": float(density_per_feat[i].item()),
            }
            for i, v in zip(best_mean_idx.tolist(), best_mean_vals.tolist())
        ],
        "top_features_by_density": [
            {
                "feature": int(i),
                "density": float(v),
                "mean": float(mean_per_feat[i].item()),
                "max": float(max_per_feat[i].item()),
            }
            for i, v in zip(best_density_idx.tolist(), best_density_vals.tolist())
        ],
        "first_row_examples": first_rows_examples,
    }

    print("\n=== LATENT SUMMARY ===")
    print(json.dumps(summary, indent=2))

    print("\n=== FEATURE DENSITY COUNTS ===")
    for p in [0.0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0]:
        cnt = int((density_per_feat <= p).sum().item())
        print(f"density <= {p:>5}: {cnt}")

    print("\n=== FEATURE VALUE COUNTS ===")
    print(f"features with max <= 0.1 : {int((max_per_feat <= 0.1).sum().item())}")
    print(f"features with max <= 1.0 : {int((max_per_feat <= 1.0).sum().item())}")
    print(f"features with max >  1.0 : {int((max_per_feat > 1.0).sum().item())}")
    print(f"features with any act >1 : {int((gt1_count > 0).sum().item())}")
    print(f"features with any act >.1: {int((gt01_count > 0).sum().item())}")

    print("\n=== VALUE QUANTILES ===")
    for name, t in [
        ("max_per_feat", max_per_feat),
        ("mean_per_feat", mean_per_feat),
        ("std_per_feat", std_per_feat),
        ("density_per_feat", density_per_feat),
    ]:
        qs = torch.quantile(t, torch.tensor([0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]))
        print(name, [float(x) for x in qs.tolist()])


def main():
    ap = argparse.ArgumentParser(description="Inspect SAE latent distributions on stored activations.")
    ap.add_argument("--sae-ckpt", type=Path, required=True)
    ap.add_argument("--store-root", type=Path, required=True)
    ap.add_argument("--layer", type=str, required=True)
    ap.add_argument("--max-shards", type=int, default=5)
    ap.add_argument("--token-chunk-size", type=int, default=8192)
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()

    inspect_latents(
        sae_ckpt=args.sae_ckpt,
        store_root=args.store_root,
        layer=args.layer,
        max_shards=args.max_shards,
        token_chunk_size=args.token_chunk_size,
        topk=args.topk,
    )


if __name__ == "__main__":
    main()
