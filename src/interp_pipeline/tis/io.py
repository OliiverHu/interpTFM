from __future__ import annotations

import os
import glob
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from scipy import sparse


CLS_TOKENS = {"<cls>", "<CLS>", "CLS"}
DEFAULT_CLS_TOKEN_ID = 60695  # scGPT vocab id for <cls> in your checkpoint

def list_shard_dirs(root: str) -> List[str]:
    shards = sorted([p for p in glob.glob(os.path.join(root, "shard_*")) if os.path.isdir(p)])
    return shards


def _load_index_pt(path: str) -> Dict[str, Any]:
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise RuntimeError(f"{path}: expected dict, got {type(obj)}")
    return obj


def extract_cls_from_shards(
    acts_root: str,
    *,
    max_shards: Optional[int] = None,
    cls_token_id: int = DEFAULT_CLS_TOKEN_ID,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Reads activation shards and extracts rows that correspond to CLS tokens.

    Handles both cases:
      - token_ids contains literal "<cls>"
      - token_ids contains vocab ids as strings/ints (e.g. "60695")
    """
    shards = list_shard_dirs(acts_root)
    if max_shards is not None:
        shards = shards[: int(max_shards)]
    if not shards:
        raise RuntimeError(f"No shards under {acts_root}")

    A_list: List[np.ndarray] = []
    id_list: List[str] = []

    def is_cls(tok: Any) -> bool:
        s = str(tok).strip()
        if s in CLS_TOKENS:
            return True
        if s.isdigit():
            return int(s) == int(cls_token_id)
        return False

    for shard in shards:
        acts_path = os.path.join(shard, "activations.pt")
        idx_path = os.path.join(shard, "index.pt")
        if not (os.path.exists(acts_path) and os.path.exists(idx_path)):
            continue

        X = torch.load(acts_path, map_location="cpu")
        idx = _load_index_pt(idx_path)

        token_ids = idx.get("token_ids", None)
        if token_ids is None or len(token_ids) != X.shape[0]:
            continue

        cls_rows = [i for i, t in enumerate(token_ids) if is_cls(t)]
        if not cls_rows:
            continue

        Xcls = X[cls_rows].detach().cpu().numpy().astype(np.float32)
        A_list.append(Xcls)

        # optional cell ids if present
        cell_ids = idx.get("cell_ids", None) or idx.get("obs_ids", None) or idx.get("sample_ids", None)
        if cell_ids is not None and len(cell_ids) == len(token_ids):
            for r in cls_rows:
                id_list.append(str(cell_ids[r]))

    if not A_list:
        # debug hint: print a sample of token_ids from the first shard
        first_idx = os.path.join(shards[0], "index.pt")
        try:
            idx0 = _load_index_pt(first_idx)
            toks = idx0.get("token_ids", [])[:10]
            raise RuntimeError(
                "No CLS rows found across shards.\n"
                f"  Hint: token_ids sample from first shard: {toks}\n"
                f"  Current cls_token_id={cls_token_id} (edit if different)."
            )
        except Exception:
            raise RuntimeError("No CLS rows found across shards. Check token_ids content.")

    A_cls = np.concatenate(A_list, axis=0)
    ids = np.array(id_list, dtype=object) if id_list else None
    return A_cls, ids


@torch.no_grad()
def encode_sae_latents(
    A: np.ndarray,
    sae_ckpt_path: str,
    *,
    device: str = "cuda",
    batch_size: int = 8192,
) -> np.ndarray:
    """
    Encode cell-level activations A [N,H] into SAE latents Z [N,K].
    Expects SAE checkpoint format: dict with state_dict, d_in, n_latents.
    """
    from interp_pipeline.sae.sae_base import AutoEncoder

    ckpt = torch.load(sae_ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise RuntimeError(f"Bad SAE ckpt: {sae_ckpt_path}. expected dict with state_dict/d_in/n_latents")
    model = AutoEncoder(d_in=int(ckpt["d_in"]), n_latents=int(ckpt["n_latents"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)

    X = torch.from_numpy(A).float()
    Zs = []
    for i0 in range(0, X.shape[0], int(batch_size)):
        xb = X[i0:i0 + int(batch_size)].to(device, non_blocking=True)
        out = model(xb)
        if not (isinstance(out, (tuple, list)) and len(out) == 2):
            raise RuntimeError("SAE forward must return (x_hat, z)")
        zb = out[1].detach().cpu()
        Zs.append(zb)
    Z = torch.cat(Zs, dim=0).numpy().astype(np.float32)
    return Z


def load_expression_from_adata(adata, use_layer: Optional[str] = None):
    """
    Returns sparse-friendly expression matrix.
    """
    if use_layer is None:
        X = adata.X
    else:
        X = adata.layers[use_layer]
    return X if sparse.issparse(X) else X.astype(np.float32)


from typing import Optional, Tuple, Any, Dict, List
import numpy as np
import os, glob
import torch

CLS_TOKENS = {"<cls>", "<CLS>", "CLS"}
DEFAULT_CLS_TOKEN_ID = 60695


def extract_cls_from_shards_aligned(
    acts_root: str,
    *,
    obs_names: Sequence[str],
    max_shards: Optional[int] = None,
    cls_token_id: int = DEFAULT_CLS_TOKEN_ID,
    example_id_key: str = "example_ids",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract CLS activations and align them into adata.obs order.

    Alignment logic:
      - if example_ids look like integers -> treat as row indices
      - else treat as string IDs and match to adata.obs_names via dict

    Returns:
      A_aligned: (n_obs, H) float32
      has_row:   (n_obs,) bool
    """
    n_obs = len(obs_names)
    obs_index = {str(k): i for i, k in enumerate(obs_names)}

    shards = sorted([p for p in glob.glob(os.path.join(acts_root, "shard_*")) if os.path.isdir(p)])
    if max_shards is not None:
        shards = shards[: int(max_shards)]
    if not shards:
        raise RuntimeError(f"No shards under {acts_root}")

    def is_cls(tok: Any) -> bool:
        s = str(tok).strip()
        if s in CLS_TOKENS:
            return True
        if s.isdigit():
            return int(s) == int(cls_token_id)
        return False

    A_aligned = None
    has_row = np.zeros((n_obs,), dtype=bool)

    n_unmatched = 0
    n_total_cls = 0

    for shard in shards:
        acts_path = os.path.join(shard, "activations.pt")
        idx_path = os.path.join(shard, "index.pt")
        if not (os.path.exists(acts_path) and os.path.exists(idx_path)):
            continue

        X = torch.load(acts_path, map_location="cpu")
        idx: Dict[str, Any] = torch.load(idx_path, map_location="cpu")

        token_ids = idx.get("token_ids", None)
        example_ids = idx.get(example_id_key, None)

        if token_ids is None or example_ids is None:
            continue
        if len(token_ids) != X.shape[0] or len(example_ids) != X.shape[0]:
            continue

        cls_rows = [i for i, t in enumerate(token_ids) if is_cls(t)]
        if not cls_rows:
            continue

        if A_aligned is None:
            H = int(X.shape[1])
            A_aligned = np.zeros((n_obs, H), dtype=np.float32)

        for r in cls_rows:
            n_total_cls += 1
            ex_raw = example_ids[r]
            ex_str = str(ex_raw)

            # Case 1: numeric string -> row index
            if ex_str.strip().isdigit():
                ex_idx = int(ex_str)
            else:
                # Case 2: obs-name key
                ex_idx = obs_index.get(ex_str, None)

            if ex_idx is None or not (0 <= int(ex_idx) < n_obs):
                n_unmatched += 1
                continue

            ex_idx = int(ex_idx)
            A_aligned[ex_idx] = X[r].detach().cpu().numpy().astype(np.float32)
            has_row[ex_idx] = True

    if A_aligned is None or not has_row.any():
        # debug hint
        idx0 = torch.load(os.path.join(shards[0], "index.pt"), map_location="cpu")
        sample_tokens = idx0.get("token_ids", [])[:10]
        sample_ex = idx0.get(example_id_key, [])[:10]
        raise RuntimeError(
            "No CLS rows aligned.\n"
            f"token_ids sample={sample_tokens}\n"
            f"{example_id_key} sample={sample_ex}\n"
            f"cls_token_id={cls_token_id}\n"
            f"index keys={list(idx0.keys())}"
        )

    if n_unmatched > 0:
        print(f"[warn] CLS alignment: {n_unmatched}/{n_total_cls} CLS rows had example_ids not found in obs_names.")

    return A_aligned, has_row


def build_judge_matrix(
    adata,
    *,
    mode: str = "raw",     # "raw" | "log1p_cp10k" | "layer:<name>"
    target_sum: float = 1e4,
    eps: float = 1e-12,
):
    """
    Returns J in the same format as notebook typically uses.
    mode:
      - "raw": adata.X as-is
      - "log1p_cp10k": normalize_total to target_sum then log1p (sparse-safe)
      - "layer:<name>": adata.layers[name]
    """
    import numpy as np
    from scipy import sparse

    if mode.startswith("layer:"):
        name = mode.split(":", 1)[1]
        X = adata.layers[name]
        return X

    X = adata.X

    if mode == "raw":
        return X

    if mode == "log1p_cp10k":
        if sparse.issparse(X):
            rs = np.asarray(X.sum(axis=1)).ravel()
            rs = np.where(rs < eps, 1.0, rs)
            scale = (target_sum / rs).astype(np.float32)
            Y = X.multiply(scale[:, None])
            Y = Y.tocsr()
            Y.data = np.log1p(Y.data)
            return Y
        else:
            rs = X.sum(axis=1, keepdims=True)
            rs = np.where(rs < eps, 1.0, rs)
            Y = (X / rs) * target_sum
            return np.log1p(Y).astype(np.float32)

    raise ValueError(f"Unknown judge mode: {mode}")