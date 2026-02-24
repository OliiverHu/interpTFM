from __future__ import annotations

import os
import glob
import time
from typing import Any, Dict, List, Callable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import scanpy as sc
from tqdm import tqdm
from scipy import sparse

from interp_pipeline.adapters.models.scgpt import ScGPTAdapter
from interp_pipeline.adapters.model_base import ModelSpec

# NEW: import the library functions you pasted above
from interp_pipeline.get_annotation.f1_alignment import (
    accumulate_tp_sparse_into_dense,
    compute_f1_from_counts,
    top_hits_per_latent,
)


# -----------------------
# Config (EDIT THESE)
# -----------------------
ADATA_PATH = "/maiziezhou_lab2/yunfei/Projects/FM_temp/InterPLM/interplm/ge_shards/cosmx_human_lung_sec8.h5ad"
GT_PATH = "debug_acts/gprofiler_only/gprofiler_binary_gene_by_term.csv"

LAYER = "layer_0.norm2"
ACTS_ROOT = f"debug_acts/activations/{LAYER}"

SAE_CKPT = "/maiziezhou_lab2/yunfei/Projects/interpTFM/scripts/debug_acts/sae_layer_0.norm2.pt"
SCGPT_CKPT = "/maiziezhou_lab/zihang/SpatialFoundationModel/eval/scGPT/whole-human-pretrain"

OUT_DIR = "debug_acts/f1_align_real"
os.makedirs(OUT_DIR, exist_ok=True)

MAX_SHARDS = 20
SHARD_OFFSET = 0

BATCH_SIZE = 8192
TOPN_PER_LATENT = 10
LATENT_THRESHOLDS = [0.0, 0.15, 0.3, 0.6]
EXPECT_N_LATENTS: Optional[int] = 4096

DO_ACTIVATION_NORM = True
NORM_MAX_ROWS: Optional[int] = 2_000_000
DO_SAE_WEIGHT_NORM = True

# memory knob
TP_DTYPE = np.int32  # tp[K,T] fits in int32 for your scales


# -----------------------
# Utils
# -----------------------
def now() -> float:
    return time.time()


def fmt_dt(t0: float) -> str:
    return f"{(time.time() - t0):.1f}s"


def list_shards(root: str) -> List[str]:
    shards = sorted(glob.glob(os.path.join(root, "shard_*")))
    return [s for s in shards if os.path.isdir(s)]


def load_index_pt(path: str) -> Dict[str, Any]:
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise RuntimeError(f"{path}: expected dict, got {type(obj)}")
    return obj


def is_int_str(x: Any) -> bool:
    try:
        return str(x).strip().isdigit()
    except Exception:
        return False


def build_symbol_to_ensembl_from_adata(adata) -> Dict[str, str]:
    ens = adata.var.index.astype(str).tolist()
    if "index" not in adata.var.columns:
        raise ValueError("Expected adata.var['index'] to contain gene symbols.")
    sym = adata.var["index"].astype(str).tolist()
    return {s: e for e, s in zip(ens, sym)}


def build_scgpt_gene_decoder(tokenizer: Any) -> Callable[[int], str]:
    for parent_attr in ["gene_vocab", "vocab"]:
        if hasattr(tokenizer, parent_attr):
            v = getattr(tokenizer, parent_attr)
            for itos_attr in ["itos", "get_itos"]:
                if hasattr(v, itos_attr):
                    itos = getattr(v, itos_attr)
                    itos = itos() if callable(itos) else itos
                    if isinstance(itos, (list, tuple)) and len(itos) > 0:
                        def _fast(i: int, itos_ref=itos) -> str:
                            i = int(i)
                            if i < 0 or i >= len(itos_ref):
                                return ""
                            return str(itos_ref[i])
                        return _fast

    if hasattr(tokenizer, "decode") and callable(getattr(tokenizer, "decode")):
        def _decode_one(i: int) -> str:
            out = tokenizer.decode([[int(i)]])
            if not out:
                return ""
            if isinstance(out[0], list):
                return str(out[0][0]) if out[0] else ""
            return str(out[0]) if out else ""
        return _decode_one

    raise RuntimeError("Cannot build decoder for tokenizer.")


def load_sae_any(path: str, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, torch.nn.Module):
        return ckpt.to(device).eval()
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Unknown SAE checkpoint type: {type(ckpt)}")

    state = ckpt.get("state_dict")
    d_in = ckpt.get("d_in")
    n_lat = ckpt.get("n_latents")
    if state is None or d_in is None or n_lat is None:
        raise RuntimeError(f"Checkpoint missing required keys. Have: {list(ckpt.keys())}")

    from interp_pipeline.sae.sae_base import AutoEncoder
    model = AutoEncoder(d_in=int(d_in), n_latents=int(n_lat))
    model.load_state_dict(state)
    return model.to(device).eval()


@torch.no_grad()
def sae_encode(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    if isinstance(out, (tuple, list)) and len(out) == 2 and isinstance(out[1], torch.Tensor):
        return out[1]
    raise RuntimeError("Unexpected SAE forward output; expected (x_hat, z).")


@torch.no_grad()
def estimate_mean_std_streaming(
    shards: List[str],
    sym2ens: Dict[str, str],
    gt_row_for_ens: Dict[str, int],
    decode_token: Callable[[int], str],
    decode_cache: Dict[int, str],
    max_rows: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    n = 0
    mean = None
    M2 = None

    for shard in tqdm(shards, desc="norm-estimate", leave=False):
        acts_path = os.path.join(shard, "activations.pt")
        idx_path = os.path.join(shard, "index.pt")
        if not (os.path.exists(acts_path) and os.path.exists(idx_path)):
            continue

        X = torch.load(acts_path, map_location="cpu")
        if not isinstance(X, torch.Tensor) or X.ndim != 2:
            continue

        idx = load_index_pt(idx_path)
        token_unit = idx.get("token_unit", "unknown")
        token_ids = idx.get("token_ids", None)
        if token_unit != "gene" or token_ids is None or len(token_ids) != X.shape[0]:
            continue

        gt_rows = np.full((X.shape[0],), -1, dtype=np.int32)
        for i, tid in enumerate(token_ids):
            if not is_int_str(tid):
                sym = str(tid).strip()
            else:
                ii = int(tid)
                sym = decode_cache.get(ii)
                if sym is None:
                    sym = decode_token(ii)
                    decode_cache[ii] = sym

            if not sym or sym in {"<pad>", "<PAD>", "<cls>", "<CLS>", "PAD", "CLS"}:
                continue

            ens = sym2ens.get(sym)
            if ens is None and sym.startswith("ENSG"):
                ens = sym
            if ens is None:
                continue

            r = gt_row_for_ens.get(ens)
            if r is not None:
                gt_rows[i] = r

        mask = gt_rows >= 0
        if not np.any(mask):
            continue

        X_use = X[mask].float()
        if mean is None:
            H = X_use.shape[1]
            mean = torch.zeros((H,), dtype=torch.float64)
            M2 = torch.zeros((H,), dtype=torch.float64)

        Xb = X_use.to(torch.float64)
        for row in Xb:
            n += 1
            delta = row - mean
            mean += delta / n
            delta2 = row - mean
            M2 += delta * delta2
            if max_rows is not None and n >= max_rows:
                break
        if max_rows is not None and n >= max_rows:
            break

    if mean is None or M2 is None or n < 2:
        raise RuntimeError("Failed to estimate mean/std: not enough mapped rows.")

    var = M2 / (n - 1)
    std = torch.sqrt(torch.clamp(var, min=1e-12))
    return mean.to(torch.float32), std.to(torch.float32), n


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[1/7] Load CosMx AnnData + mappings")
    t0 = now()
    adata = sc.read_h5ad(ADATA_PATH)
    sym2ens = build_symbol_to_ensembl_from_adata(adata)
    print(f"  loaded adata: n_obs={adata.n_obs} n_vars={adata.n_vars} in {fmt_dt(t0)}")

    print("[2/7] Load gene×term GT (rows=Ensembl IDs)")
    t0 = now()
    gt_df = pd.read_csv(GT_PATH, index_col=0)
    gt_genes = gt_df.index.astype(str).tolist()
    term_ids = gt_df.columns.astype(str).tolist()
    GT = gt_df.values.astype(np.int8)  # keep small
    gt_row_for_ens = {g: i for i, g in enumerate(gt_genes)}
    print(f"  GT shape={GT.shape} in {fmt_dt(t0)}")

    print("[3/7] Load scGPT (for vocab decode)")
    t0 = now()
    scgpt_adapter = ScGPTAdapter()
    handle = scgpt_adapter.load(ModelSpec(name="scgpt", checkpoint=SCGPT_CKPT, device=str(device)))
    decode_token = build_scgpt_gene_decoder(handle.tokenizer)
    print(f"  scGPT loaded in {fmt_dt(t0)}")

    print("[4/7] List shards (with limiting)")
    shards = list_shards(ACTS_ROOT)
    if not shards:
        raise RuntimeError(f"No shards found under {ACTS_ROOT}")
    print(f"  found {len(shards)} shards total")
    shards = shards[SHARD_OFFSET: SHARD_OFFSET + MAX_SHARDS] if MAX_SHARDS is not None else shards[SHARD_OFFSET:]
    print(f"  using {len(shards)} shards (offset={SHARD_OFFSET}, max={MAX_SHARDS})")

    print("[5/7] Load SAE")
    sae = load_sae_any(SAE_CKPT, device=device)
    print(f"  SAE loaded on {device}")

    decode_cache: Dict[int, str] = {}

    mean = std = None
    if DO_ACTIVATION_NORM:
        print("[6/7] Estimate activation mean/std")
        mean, std, n_used = estimate_mean_std_streaming(
            shards=shards,
            sym2ens=sym2ens,
            gt_row_for_ens=gt_row_for_ens,
            decode_token=decode_token,
            decode_cache=decode_cache,
            max_rows=NORM_MAX_ROWS,
        )
        print(f"  estimated mean/std using n_rows={n_used}")

    print("[7/7] Stream shards + accumulate TP (SPARSE) for multiple thresholds")

    tp: Dict[float, Optional[np.ndarray]] = {thr: None for thr in LATENT_THRESHOLDS}
    pred_pos: Dict[float, Optional[np.ndarray]] = {thr: None for thr in LATENT_THRESHOLDS}
    true_pos: Optional[np.ndarray] = None

    observed_K: Optional[int] = None
    T_terms = GT.shape[1]

    n_rows_total = 0
    n_rows_mapped = 0

    for shard in tqdm(shards, desc="shards"):
        acts_path = os.path.join(shard, "activations.pt")
        idx_path = os.path.join(shard, "index.pt")
        if not (os.path.exists(acts_path) and os.path.exists(idx_path)):
            continue

        X = torch.load(acts_path, map_location="cpu")
        if not isinstance(X, torch.Tensor) or X.ndim != 2:
            raise RuntimeError(f"{acts_path}: expected Tensor [N,H], got {type(X)} shape={getattr(X,'shape',None)}")

        idx = load_index_pt(idx_path)
        token_unit = idx.get("token_unit", "unknown")
        token_ids = idx.get("token_ids", None)
        if token_unit != "gene" or token_ids is None or len(token_ids) != X.shape[0]:
            continue

        n_rows_total += X.shape[0]

        gt_rows = np.full((X.shape[0],), -1, dtype=np.int32)
        for i, tid in enumerate(token_ids):
            if not is_int_str(tid):
                sym = str(tid).strip()
            else:
                ii = int(tid)
                sym = decode_cache.get(ii)
                if sym is None:
                    sym = decode_token(ii)
                    decode_cache[ii] = sym

            if not sym or sym in {"<pad>", "<PAD>", "<cls>", "<CLS>", "PAD", "CLS"}:
                continue

            ens = sym2ens.get(sym)
            if ens is None and sym.startswith("ENSG"):
                ens = sym
            if ens is None:
                continue

            r = gt_row_for_ens.get(ens)
            if r is not None:
                gt_rows[i] = r

        mask = gt_rows >= 0
        if not np.any(mask):
            continue

        n_rows_mapped += int(mask.sum())

        X_use = X[mask].float()
        if DO_ACTIVATION_NORM and mean is not None and std is not None:
            X_use = (X_use - mean) / std

        G_use = GT[gt_rows[mask]]               # dense 0/1
        G_csr = sparse.csr_matrix(G_use)        # sparse (N_use, T)

        if true_pos is None:
            true_pos = np.zeros((T_terms,), dtype=np.int64)
        true_pos += np.asarray(G_csr.sum(axis=0)).ravel().astype(np.int64)

        # Encode -> Z on CPU
        Zs = []
        for i0 in range(0, X_use.shape[0], BATCH_SIZE):
            xb = X_use[i0:i0 + BATCH_SIZE].to(device, non_blocking=True)
            zb = sae_encode(sae, xb)
            Zs.append(zb.detach().cpu())
        Z = torch.cat(Zs, dim=0).numpy()  # (N_use, K)

        if observed_K is None:
            observed_K = int(Z.shape[1])
            print(f"\n  inferred SAE latents K={observed_K}")
            if EXPECT_N_LATENTS is not None and observed_K != EXPECT_N_LATENTS:
                raise RuntimeError(f"Expected K={EXPECT_N_LATENTS} but got K={observed_K}. Wrong SAE_CKPT?")

            for thr in LATENT_THRESHOLDS:
                tp[thr] = np.zeros((observed_K, T_terms), dtype=TP_DTYPE)
                pred_pos[thr] = np.zeros((observed_K,), dtype=np.int64)

        assert observed_K is not None

        for thr in LATENT_THRESHOLDS:
            A_bool = (Z > float(thr))  # (N_use, K) bool
            pred_pos[thr] += A_bool.sum(axis=0).astype(np.int64)
            accumulate_tp_sparse_into_dense(tp[thr], A_bool, G_csr)

    if observed_K is None or true_pos is None:
        raise RuntimeError("No usable rows mapped to GT across shards.")

    print(f"  processed rows total={n_rows_total} mapped={n_rows_mapped}")
    print(f"  decode cache size={len(decode_cache)}")

    # Save outputs per threshold
    for thr in LATENT_THRESHOLDS:
        precision, recall, f1 = compute_f1_from_counts(tp[thr], pred_pos[thr], true_pos)
        rows = top_hits_per_latent(
            f1=f1,
            tp=tp[thr],
            pred_pos=pred_pos[thr],
            true_pos=true_pos,
            term_ids=term_ids,
            topn=TOPN_PER_LATENT,
            threshold=thr,
        )

        out_path = os.path.join(OUT_DIR, f"top_hits_{LAYER}_thr{thr}_SPARSE.csv")
        pd.DataFrame(rows).to_csv(out_path, index=False)

        f1_path = os.path.join(OUT_DIR, f"f1_matrix_{LAYER}_thr{thr}_SPARSE.npy")
        np.save(f1_path, f1.astype(np.float32))

        print(f"  wrote: {out_path}")
        print(f"  wrote: {f1_path}")

    print("DONE.")


if __name__ == "__main__":
    main()