from __future__ import annotations

import copy
import glob
import json
import os
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from interp_pipeline.sae.sae_base import AutoEncoder


# =============================================================================
# Legacy API preserved for existing scripts
# =============================================================================

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
    """
    Original max-activation scan used by scripts/test_normalize_sae_latents.py.
    Kept for backward compatibility.
    """
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
    """
    Original InterPLM-style checkpoint rewrite.

    This is intentionally unchanged for backward compatibility with old behavior:
      encoder.weight /= max
      encoder.bias   /= max
      decoder.weight *= max

    Note: this only guards max <= 0. For safer F1 sensitivity analysis, use
    normalize_sae_checkpoint_for_f1(...), which adds tiny-scale and active-rate
    guards.
    """
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
    """
    Original public helper. Kept so existing scripts continue to work.
    """
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
    """
    Original wrapper used by scripts/test_normalize_sae_latents.py.

    Do not change this default behavior; keeping it stable prevents older
    scripts from silently changing meaning.
    """
    return normalize_sae_features(
        sae_ckpt_path=ckpt_path,
        activations_root=store_root,
        layer=layer,
        out_path=output_path,
        max_shards=max_shards,
        token_chunk_size=token_chunk_size,
        device=device,
    )


# =============================================================================
# New F1-safe max1 normalization API
# =============================================================================

def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _get_state_dict_container(ckpt: dict) -> dict:
    if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]
    if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
        return ckpt["model_state_dict"]
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]
    if all(torch.is_tensor(v) for v in ckpt.values()):
        return ckpt
    raise RuntimeError(f"Could not find state dict. Top-level keys: {list(ckpt.keys())[:40]}")


def _first_tensor_key(state: dict, keys: Sequence[str]) -> Optional[str]:
    for k in keys:
        if k in state and torch.is_tensor(state[k]):
            return k
    return None


class EncoderOnlySAE(torch.nn.Module):
    """
    Minimal SAE encoder used for robust F1-normalization calibration.

    Supports both your newer checkpoints and older-style checkpoints:
      encoder.weight / W_enc / enc.weight
      encoder.bias / b_enc / enc.bias
      bias / x_bias
    """
    def __init__(
        self,
        w_enc: torch.Tensor,
        b_enc: Optional[torch.Tensor] = None,
        x_bias: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.register_buffer("w_enc", w_enc.float().contiguous())
        self.register_buffer("b_enc", None if b_enc is None else b_enc.float().contiguous())
        self.register_buffer("x_bias", None if x_bias is None else x_bias.float().contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if self.x_bias is not None:
            x = x - self.x_bias.to(device=x.device, dtype=x.dtype)
        z_pre = x @ self.w_enc.to(device=x.device, dtype=x.dtype).T
        if self.b_enc is not None:
            z_pre = z_pre + self.b_enc.to(device=x.device, dtype=x.dtype)
        return torch.relu(z_pre)


def load_sae_encoder_from_checkpoint(
    ckpt_path: str,
    device: torch.device,
) -> Tuple[torch.nn.Module, int, int, dict, dict, str, Optional[str], Optional[str]]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Expected dict checkpoint, got {type(ckpt)}")

    state = _get_state_dict_container(ckpt)

    w_key = _first_tensor_key(state, ["encoder.weight", "W_enc", "enc.weight"])
    if w_key is None:
        raise RuntimeError(f"Could not find encoder weight. First state keys: {list(state.keys())[:40]}")

    w_enc = state[w_key].detach().float()
    if w_enc.ndim != 2:
        raise RuntimeError(f"{w_key} must be 2D, got {tuple(w_enc.shape)}")

    n_lat, d_in_from_w = int(w_enc.shape[0]), int(w_enc.shape[1])
    d_in = int(ckpt.get("d_in", d_in_from_w))
    n_lat_ckpt = int(ckpt.get("n_latents", n_lat))

    if d_in != d_in_from_w:
        raise RuntimeError(f"ckpt d_in={d_in}, encoder input dim={d_in_from_w}")
    if n_lat_ckpt != n_lat:
        raise RuntimeError(f"ckpt n_latents={n_lat_ckpt}, encoder rows={n_lat}")

    b_key = _first_tensor_key(state, ["encoder.bias", "b_enc", "enc.bias"])
    x_bias_key = _first_tensor_key(state, ["bias", "x_bias"])

    b_enc = state[b_key].detach().float() if b_key is not None else None
    x_bias = state[x_bias_key].detach().float() if x_bias_key is not None else None

    if b_enc is not None and int(b_enc.numel()) != n_lat:
        raise RuntimeError(f"{b_key} shape={tuple(b_enc.shape)}, expected {n_lat}")
    if x_bias is not None and int(x_bias.numel()) != d_in:
        raise RuntimeError(f"{x_bias_key} shape={tuple(x_bias.shape)}, expected {d_in}")

    model = EncoderOnlySAE(w_enc=w_enc, b_enc=b_enc, x_bias=x_bias).to(device).eval()

    return model, d_in, n_lat, ckpt, state, w_key, b_key, x_bias_key


def _list_activation_shards(
    store_root: str,
    layer: str,
    max_shards: Optional[int] = None,
    shard_offset: int = 0,
) -> list[str]:
    acts_root = os.path.join(store_root, "activations", layer)
    shards = sorted([p for p in glob.glob(os.path.join(acts_root, "shard_*")) if os.path.isdir(p)])
    if shard_offset:
        shards = shards[int(shard_offset):]
    if max_shards is not None:
        shards = shards[: int(max_shards)]
    if not shards:
        raise RuntimeError(f"No shard dirs found under {acts_root}")
    return shards


@torch.no_grad()
def compute_sae_latent_max1_stats(
    *,
    sae: torch.nn.Module,
    d_in: int,
    n_latents: int,
    store_root: str,
    layer: str,
    device: torch.device,
    batch_size: int = 8192,
    max_shards: Optional[int] = None,
    shard_offset: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """
    Compute max activation and active rate for each SAE latent over activation shards.
    """
    shards = _list_activation_shards(
        store_root=store_root,
        layer=layer,
        max_shards=max_shards,
        shard_offset=shard_offset,
    )

    max_per_latent = torch.zeros(n_latents, dtype=torch.float32)
    active_count = torch.zeros(n_latents, dtype=torch.int64)
    n_rows = 0
    n_shards_used = 0

    for shard in tqdm(shards, desc=f"scan:{layer}"):
        acts_path = os.path.join(shard, "activations.pt")
        if not os.path.exists(acts_path):
            continue

        X = torch.load(acts_path, map_location="cpu")
        if not isinstance(X, torch.Tensor) or X.ndim != 2:
            continue
        if int(X.shape[1]) != int(d_in):
            raise RuntimeError(f"{acts_path}: H={X.shape[1]} but SAE d_in={d_in}")

        n_shards_used += 1
        n_rows += int(X.shape[0])

        for i0 in range(0, int(X.shape[0]), int(batch_size)):
            xb = X[i0:i0 + int(batch_size)].float().to(device, non_blocking=True)
            z = sae(xb).detach().cpu()
            max_per_latent = torch.maximum(max_per_latent, z.max(dim=0).values)
            active_count += (z > 0).sum(dim=0).to(torch.int64)

    if n_rows == 0:
        raise RuntimeError("No activation rows processed.")

    active_rate = active_count.float() / float(n_rows)
    return max_per_latent, active_rate, int(n_rows), int(n_shards_used)


def make_safe_max1_scale(
    max_per_latent: torch.Tensor,
    active_rate: torch.Tensor,
    *,
    min_scale: float = 1e-3,
    min_active_rate: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Latents with tiny max or tiny active rate are left unchanged.
    """
    was_normalized = (max_per_latent >= float(min_scale)) & (active_rate >= float(min_active_rate))
    safe_scale = max_per_latent.clone()
    safe_scale[~was_normalized] = 1.0
    return safe_scale, was_normalized


def scale_decoder_if_present(state: dict, safe_scale: torch.Tensor) -> list[str]:
    """
    Optionally preserve reconstruction equivalence.

    Common PyTorch Linear decoder:
      decoder.weight shape = (d_in, n_latents), multiply columns by scale.

    Alternate matrix:
      W_dec shape = (n_latents, d_in), multiply rows by scale.
    """
    changed = []

    for key in ["decoder.weight", "W_dec", "dec.weight"]:
        if key not in state or not torch.is_tensor(state[key]):
            continue

        W = state[key]
        scale = safe_scale.to(dtype=W.dtype, device=W.device)

        if W.ndim != 2:
            continue

        if W.shape[1] == scale.numel():
            state[key] = W * scale.view(1, -1)
            changed.append(key)
        elif W.shape[0] == scale.numel():
            state[key] = W * scale.view(-1, 1)
            changed.append(key)

    return changed


def normalize_sae_checkpoint_for_f1(
    *,
    ckpt_path: str,
    store_root: str,
    layer: str,
    output_path: str,
    label: str = "",
    device: str = "cuda",
    batch_size: int = 8192,
    max_shards: Optional[int] = None,
    shard_offset: int = 0,
    min_scale: float = 1e-3,
    min_active_rate: float = 1e-4,
    scale_decoder: bool = True,
    stats_dir: Optional[str] = None,
) -> str:
    """
    Create an F1-compatible max1-normalized checkpoint.

    The current F1 heldout script thresholds latent activations directly.
    This function bakes InterPLM-style post-activation scaling into the encoder:

      z_norm = ReLU(pre / safe_scale) = ReLU(pre) / safe_scale

    Since safe_scale is positive, dividing encoder.weight and encoder.bias by
    safe_scale is equivalent for latent thresholding.

    Input checkpoint is not modified.
    """
    output_path_p = Path(output_path)
    _ensure_dir(output_path_p.parent)

    if stats_dir is None:
        stats_dir_p = _ensure_dir(output_path_p.parent / f"{output_path_p.stem}_max1_stats")
    else:
        stats_dir_p = _ensure_dir(stats_dir)

    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    sae, d_in, n_latents, ckpt, _state, w_key, b_key, _x_bias_key = load_sae_encoder_from_checkpoint(
        ckpt_path=ckpt_path,
        device=dev,
    )

    max_per_latent, active_rate, n_rows, n_shards_used = compute_sae_latent_max1_stats(
        sae=sae,
        d_in=d_in,
        n_latents=n_latents,
        store_root=store_root,
        layer=layer,
        device=dev,
        batch_size=batch_size,
        max_shards=max_shards,
        shard_offset=shard_offset,
    )

    safe_scale, was_normalized = make_safe_max1_scale(
        max_per_latent=max_per_latent,
        active_rate=active_rate,
        min_scale=min_scale,
        min_active_rate=min_active_rate,
    )

    new_ckpt = copy.deepcopy(ckpt)
    new_state = _get_state_dict_container(new_ckpt)

    scale_for_w = safe_scale.to(dtype=new_state[w_key].dtype, device=new_state[w_key].device)
    new_state[w_key] = new_state[w_key] / scale_for_w.view(-1, 1)

    if b_key is not None and b_key in new_state and torch.is_tensor(new_state[b_key]):
        new_state[b_key] = new_state[b_key] / scale_for_w

    decoder_changed = []
    if scale_decoder:
        decoder_changed = scale_decoder_if_present(new_state, safe_scale)

    new_ckpt["max1_f1_normalized"] = True
    new_ckpt["max1_f1_normalization"] = {
        "method": "baked_encoder_divide_by_safe_max",
        "note": (
            "F1-compatible max1 checkpoint. Encoder produces z_raw / safe_scale. "
            "Input checkpoint was not modified."
        ),
        "label": label,
        "layer": layer,
        "source_ckpt": ckpt_path,
        "store_root": store_root,
        "n_rows_for_calibration": int(n_rows),
        "n_shards_used": int(n_shards_used),
        "min_scale": float(min_scale),
        "min_active_rate": float(min_active_rate),
        "n_latents": int(n_latents),
        "n_normalized": int(was_normalized.sum().item()),
        "n_skipped": int((~was_normalized).sum().item()),
        "encoder_weight_key": w_key,
        "encoder_bias_key": b_key,
        "decoder_scaled": bool(scale_decoder),
        "decoder_changed_keys": decoder_changed,
    }

    torch.save(new_ckpt, str(output_path_p))

    stats = pd.DataFrame(
        {
            "latent": np.arange(n_latents),
            "max_activation": max_per_latent.numpy(),
            "safe_scale": safe_scale.numpy(),
            "active_rate": active_rate.numpy(),
            "was_normalized": was_normalized.numpy(),
        }
    )
    stats.to_csv(stats_dir_p / "latent_max1_stats.csv", index=False)

    summary = dict(new_ckpt["max1_f1_normalization"])
    summary.update(
        {
            "max_min": float(max_per_latent.min().item()),
            "max_median": float(max_per_latent.median().item()),
            "max_p99": float(torch.quantile(max_per_latent, 0.99).item()),
            "max_max": float(max_per_latent.max().item()),
            "active_rate_min": float(active_rate.min().item()),
            "active_rate_median": float(active_rate.median().item()),
            "active_rate_p99": float(torch.quantile(active_rate, 0.99).item()),
            "active_rate_max": float(active_rate.max().item()),
            "output_ckpt": str(output_path_p),
            "stats_csv": str(stats_dir_p / "latent_max1_stats.csv"),
        }
    )
    with open(stats_dir_p / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return str(output_path_p)
