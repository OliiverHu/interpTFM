from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch

from interp_pipeline.sae.sae_base import AutoEncoder


@torch.no_grad()
def _load_wrapped_sae(ckpt_path: Path, device: str) -> AutoEncoder:
    ae = AutoEncoder.from_pretrained(str(ckpt_path), device=device)
    ae.eval()
    return ae


@torch.no_grad()
def calculate_feature_maxima(
    sae: AutoEncoder,
    activations_root: Path,
    layer: str,
    max_shards: Optional[int] = None,
    token_chunk_size: int = 25_000,
) -> torch.Tensor:
    shard_dirs = sorted((activations_root / "activations" / layer).glob("shard_*"))
    if max_shards is not None:
        shard_dirs = shard_dirs[:max_shards]

    max_per_feat = torch.zeros(sae.dict_size, device=next(sae.parameters()).device)

    for shard_dir in shard_dirs:
        act_path = shard_dir / "activations.pt"
        if not act_path.exists():
            continue
        x = torch.load(act_path, map_location=max_per_feat.device).float()
        for start in range(0, x.shape[0], token_chunk_size):
            xb = x[start : start + token_chunk_size]
            z = sae.encode(xb)
            max_per_feat = torch.maximum(max_per_feat, z.max(dim=0).values)

    return max_per_feat


@torch.no_grad()
def create_normalized_state_dict(sae: AutoEncoder, max_per_feat: torch.Tensor):
    max_safe = max_per_feat.clone()
    max_safe[max_safe <= 0] = 1.0

    sae.encoder.weight.div_(max_safe.unsqueeze(1))
    if sae.encoder.bias is not None:
        sae.encoder.bias.div_(max_safe)
    sae.decoder.weight.mul_(max_safe.unsqueeze(0))
    return sae.state_dict()


def normalize_sae_features(
    sae_ckpt_path: str,
    activations_root: str,
    layer: str,
    out_path: Optional[str] = None,
    max_shards: Optional[int] = None,
    token_chunk_size: int = 25_000,
    device: str = "cuda",
) -> str:
    ckpt_path = Path(sae_ckpt_path)
    out_file = Path(out_path) if out_path else ckpt_path.with_name(f"{ckpt_path.stem}_normalized.pt")
    feat_stat_dir = ckpt_path.parent / "feature_stats"
    feat_stat_dir.mkdir(parents=True, exist_ok=True)

    ae = _load_wrapped_sae(ckpt_path, device=device)
    max_per_feat = calculate_feature_maxima(
        sae=ae,
        activations_root=Path(activations_root),
        layer=layer,
        max_shards=max_shards,
        token_chunk_size=token_chunk_size,
    )

    torch.save(max_per_feat.cpu(), feat_stat_dir / "max_per_feat.pt")

    norm_state = create_normalized_state_dict(ae, max_per_feat)

    original = torch.load(ckpt_path, map_location="cpu")
    if isinstance(original, dict) and "state_dict" in original:
        original["state_dict"] = {k: v.cpu() for k, v in norm_state.items()}
        payload = original
    else:
        payload = {k: v.cpu() for k, v in norm_state.items()}

    torch.save(payload, out_file)

    summary = {
        "layer": layer,
        "source_ckpt": str(ckpt_path),
        "normalized_ckpt": str(out_file),
        "max_per_feat_min": float(max_per_feat.min().item()),
        "max_per_feat_median": float(max_per_feat.median().item()),
        "max_per_feat_mean": float(max_per_feat.mean().item()),
        "max_per_feat_max": float(max_per_feat.max().item()),
        "num_nonpositive_maxima": int((max_per_feat <= 0).sum().item()),
    }
    with open(feat_stat_dir / "normalization_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return str(out_file)


def normalize_sae_checkpoint(
    ckpt_path: str,
    store_root: str,
    layer: str,
    output_path: Optional[str] = None,
    device: str = "cuda",
    max_shards: Optional[int] = None,
    token_chunk_size: int = 25_000,
    feature_chunk_size: int = 1024,
) -> str:
    return normalize_sae_features(
        sae_ckpt_path=ckpt_path,
        activations_root=store_root,
        layer=layer,
        out_path=output_path,
        max_shards=max_shards,
        token_chunk_size=token_chunk_size,
        device=device,
    )