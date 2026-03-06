from __future__ import annotations

import os
import glob
from typing import Any, Dict, Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import scanpy as sc
from tqdm import tqdm
from scipy import sparse

from interp_pipeline.adapters.models.scgpt import ScGPTAdapter
from interp_pipeline.adapters.model_base import ModelSpec
from interp_pipeline.sae.sae_base import AutoEncoder

# -----------------------
# CONFIG (paste your config block here)
# -----------------------
ADATA_PATH = "/maiziezhou_lab2/yunfei/Projects/FM_temp/InterPLM/interplm/ge_shards/cosmx_human_lung_sec8.h5ad"
SCGPT_CKPT = "/maiziezhou_lab/zihang/SpatialFoundationModel/eval/scGPT/whole-human-pretrain"
OUT_ROOT = "runs/full_scgpt_cosmx"
GT_PATH = f"{OUT_ROOT}/gprofiler/gprofiler_binary_gene_by_term.csv"
DEBUG_LAYER = "layer_0.norm2"   # or None
MAX_SHARDS = 3
SHARD_OFFSET = 0
SAE_CKPT = f"{OUT_ROOT}/sae/{DEBUG_LAYER}/sae_{DEBUG_LAYER}.pt"
DEVICE = "cuda"
BATCH_SIZE = 8192
LATENT_THRESHOLDS = [0.0, 0.15, 0.3, 0.6]
ASSUME_INDEX_TOKENS_CAN_BE_VOCAB_IDS = True


# -----------------------
# Small utilities
# -----------------------
def list_shards(layer: str) -> List[str]:
    root = os.path.join(OUT_ROOT, "activations", layer)
    shards = sorted([p for p in glob.glob(os.path.join(root, "shard_*")) if os.path.isdir(p)])
    if SHARD_OFFSET:
        shards = shards[SHARD_OFFSET:]
    if MAX_SHARDS is not None:
        shards = shards[:MAX_SHARDS]
    return shards


def is_int_str(x: Any) -> bool:
    try:
        return str(x).strip().isdigit()
    except Exception:
        return False


def build_symbol_to_ensembl_from_adata(adata) -> Dict[str, str]:
    # adata.var.index: Ensembl; adata.var["index"]: symbol
    if "index" not in adata.var.columns:
        raise ValueError("Expected adata.var['index'] to contain gene symbols.")
    ens = adata.var.index.astype(str).tolist()
    sym = adata.var["index"].astype(str).tolist()
    return {s: e for e, s in zip(ens, sym)}  # fine for debug


def build_scgpt_gene_decoder(tokenizer: Any) -> Callable[[int], str]:
    # Prefer fast vocab itos
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

    # Fallback decode
    if hasattr(tokenizer, "decode") and callable(getattr(tokenizer, "decode")):
        def _decode_one(i: int) -> str:
            out = tokenizer.decode([[int(i)]])
            if not out:
                return ""
            if isinstance(out[0], list):
                return str(out[0][0]) if out[0] else ""
            return str(out[0]) if out else ""
        return _decode_one

    raise RuntimeError(f"Cannot build decoder for tokenizer={type(tokenizer)}")


def load_sae_checkpoint(path: str, device: torch.device) -> AutoEncoder:
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise RuntimeError(f"Expected ckpt dict with state_dict/d_in/n_latents. got={type(ckpt)} keys={getattr(ckpt,'keys',lambda:[])()}")
    model = AutoEncoder(d_in=int(ckpt["d_in"]), n_latents=int(ckpt["n_latents"]))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


@torch.no_grad()
def sae_encode(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    if isinstance(out, (tuple, list)) and len(out) == 2:
        return out[1]
    raise RuntimeError("SAE forward did not return (x_hat, z)")


def dense_bool_to_csr(A_bool: np.ndarray) -> sparse.csr_matrix:
    rows, cols = np.where(A_bool)
    data = np.ones(rows.shape[0], dtype=np.int8)
    return sparse.csr_matrix((data, (rows, cols)), shape=A_bool.shape)


def accumulate_tp_sparse_into_dense(tp_dense: np.ndarray, A_bool: np.ndarray, G_csr: sparse.csr_matrix) -> None:
    A_csr = dense_bool_to_csr(A_bool)
    TP = (A_csr.T @ G_csr).tocoo()
    tp_dense[TP.row, TP.col] += TP.data.astype(tp_dense.dtype, copy=False)


def compute_f1_from_counts(tp: np.ndarray, pred_pos: np.ndarray, true_pos: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    tp_f = tp.astype(np.float32)
    fp_f = (pred_pos[:, None] - tp).astype(np.float32)
    fn_f = (true_pos[None, :] - tp).astype(np.float32)
    precision = tp_f / (tp_f + fp_f + eps)
    recall = tp_f / (tp_f + fn_f + eps)
    f1 = (2 * precision * recall) / (precision + recall + eps)
    return f1


# -----------------------
# Core debug steps
# -----------------------
def debug_mapping_and_gt(
    shards: List[str],
    gt_df: pd.DataFrame,
    decode_token: Optional[Callable[[int], str]],
    sym2ens: Dict[str, str],
) -> Tuple[int, int, int]:
    """
    Returns:
      n_total_rows, n_mapped_rows, n_unique_mapped_ens
    """
    gt_row_for_ens = {g: i for i, g in enumerate(gt_df.index.astype(str).tolist())}
    decode_cache: Dict[int, str] = {}
    mapped_ens_set = set()

    n_total = 0
    n_mapped = 0

    for shard in shards:
        acts_path = os.path.join(shard, "activations.pt")
        idx_path = os.path.join(shard, "index.pt")
        if not (os.path.exists(acts_path) and os.path.exists(idx_path)):
            print(f"[warn] missing {shard}")
            continue

        X = torch.load(acts_path, map_location="cpu")
        idx = torch.load(idx_path, map_location="cpu")
        token_unit = idx.get("token_unit", "unknown")
        token_ids = idx.get("token_ids", None)

        if token_unit != "gene" or token_ids is None or len(token_ids) != X.shape[0]:
            print(f"[warn] skip shard={shard} token_unit={token_unit} token_ids={None if token_ids is None else len(token_ids)} X={tuple(X.shape)}")
            continue

        n_total += X.shape[0]

        # show a few raw ids
        raw_sample = token_ids[:5]
        print(f"\n[shard] {os.path.basename(shard)} X={tuple(X.shape)} token_ids sample={raw_sample}")

        gt_rows = np.full((X.shape[0],), -1, dtype=np.int32)

        for i, tid in enumerate(token_ids):
            s = str(tid).strip()

            # fast path: already ENSG
            if s.startswith("ENSG"):
                ens = s
            else:
                if not ASSUME_INDEX_TOKENS_CAN_BE_VOCAB_IDS:
                    continue
                if decode_token is None:
                    continue
                if not is_int_str(s):
                    # maybe it's symbol already
                    sym = s
                else:
                    ii = int(s)
                    sym = decode_cache.get(ii)
                    if sym is None:
                        sym = decode_token(ii)
                        decode_cache[ii] = sym

                if not sym or sym in {"<pad>", "<PAD>", "<cls>", "<CLS>", "PAD", "CLS"}:
                    continue
                ens = sym2ens.get(sym) if sym in sym2ens else (sym if sym.startswith("ENSG") else None)
                if ens is None:
                    continue

            r = gt_row_for_ens.get(ens)
            if r is not None:
                gt_rows[i] = r
                mapped_ens_set.add(ens)

        mask = gt_rows >= 0
        n_mapped += int(mask.sum())

        print(f"  mapped rows this shard: {int(mask.sum())}/{X.shape[0]} ({(mask.mean()*100):.2f}%)")
        if int(mask.sum()) == 0:
            # extra hint: print a few decoded tokens if possible
            if decode_token is not None and len(raw_sample) > 0 and is_int_str(raw_sample[0]):
                ex = [int(x) for x in raw_sample[:5] if is_int_str(x)]
                print("  decode(sample):", [(x, decode_token(x)) for x in ex])

    return n_total, n_mapped, len(mapped_ens_set)


def debug_alignment_one_layer(
    layer: str,
    shards: List[str],
    gt_df: pd.DataFrame,
    decode_token: Optional[Callable[[int], str]],
    sym2ens: Dict[str, str],
    sae_ckpt: Optional[str],
) -> None:
    print(f"\n==============================")
    print(f"[DEBUG] layer = {layer}")
    print(f"  shards used = {len(shards)} (offset={SHARD_OFFSET}, max={MAX_SHARDS})")
    print(f"  GT shape = {gt_df.shape} (genes x terms)")
    print(f"  GT nonzeros = {int((gt_df.values != 0).sum())}")
    print(f"  SAE_CKPT = {sae_ckpt}")
    print(f"==============================\n")

    # Step A: mapping sanity
    n_total, n_mapped, n_unique = debug_mapping_and_gt(
        shards=shards,
        gt_df=gt_df,
        decode_token=decode_token,
        sym2ens=sym2ens,
    )
    print(f"\n[mapping summary] total_rows={n_total} mapped_rows={n_mapped} unique_mapped_ens={n_unique}")
    if n_mapped == 0:
        print("\n*** ROOT CAUSE LIKELY: token_ids are not ENSG and decode/map failed OR GT rows don't match your ENSG universe. ***")
        print("Next actions:")
        print("  1) open one shard index.pt and confirm token_ids content (ENSG? vocab ids? symbols?)")
        print("  2) print decode(sample) above; if decode returns symbols not in adata.var['index'], mapping will drop them")
        return

    # If no SAE, stop here
    if sae_ckpt is None or not os.path.exists(sae_ckpt):
        print("\n[stop] SAE checkpoint missing; mapping+GT looks non-empty, but cannot test F1.")
        return

    # Step B: alignment sanity (tp/pred/true nonzero?)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    sae = load_sae_checkpoint(sae_ckpt, device=device)

    K = sae.n_latents if hasattr(sae, "n_latents") else None
    T = gt_df.shape[1]
    gt_row_for_ens = {g: i for i, g in enumerate(gt_df.index.astype(str).tolist())}
    GT = gt_df.values.astype(np.int8)
    term_ids = gt_df.columns.astype(str).tolist()

    # counters per threshold
    tp = {thr: None for thr in LATENT_THRESHOLDS}
    pred_pos = {thr: None for thr in LATENT_THRESHOLDS}
    true_pos = np.zeros((T,), dtype=np.int64)

    decode_cache: Dict[int, str] = {}

    for shard in tqdm(shards, desc="align-sanity"):
        acts_path = os.path.join(shard, "activations.pt")
        idx_path = os.path.join(shard, "index.pt")
        if not (os.path.exists(acts_path) and os.path.exists(idx_path)):
            continue

        X = torch.load(acts_path, map_location="cpu")
        idx = torch.load(idx_path, map_location="cpu")
        token_unit = idx.get("token_unit", "unknown")
        token_ids = idx.get("token_ids", None)
        if token_unit != "gene" or token_ids is None or len(token_ids) != X.shape[0]:
            continue

        gt_rows = np.full((X.shape[0],), -1, dtype=np.int32)
        for i, tid in enumerate(token_ids):
            s = str(tid).strip()
            if s.startswith("ENSG"):
                ens = s
            else:
                if not ASSUME_INDEX_TOKENS_CAN_BE_VOCAB_IDS or decode_token is None:
                    continue
                if not is_int_str(s):
                    sym = s
                else:
                    ii = int(s)
                    sym = decode_cache.get(ii)
                    if sym is None:
                        sym = decode_token(ii)
                        decode_cache[ii] = sym
                if not sym or sym in {"<pad>", "<PAD>", "<cls>", "<CLS>", "PAD", "CLS"}:
                    continue
                ens = sym2ens.get(sym) if sym in sym2ens else (sym if sym.startswith("ENSG") else None)
                if ens is None:
                    continue

            r = gt_row_for_ens.get(ens)
            if r is not None:
                gt_rows[i] = r

        mask = gt_rows >= 0
        if not np.any(mask):
            continue

        X_use = X[mask].float()
        G_use = GT[gt_rows[mask]]
        G_csr = sparse.csr_matrix(G_use)

        true_pos += np.asarray(G_csr.sum(axis=0)).ravel().astype(np.int64)

        # encode -> Z
        Zs = []
        for i0 in range(0, X_use.shape[0], BATCH_SIZE):
            xb = X_use[i0:i0 + BATCH_SIZE].to(device, non_blocking=True)
            zb = sae_encode(sae, xb).detach().cpu()
            Zs.append(zb)
        Z = torch.cat(Zs, dim=0).numpy()  # (N, K)
        # Z stats (catch dead SAE / NaNs)
        if not hasattr(debug_alignment_one_layer, "_printed_z_stats"):
            debug_alignment_one_layer._printed_z_stats = True
            z_min = float(np.nanmin(Z))
            z_max = float(np.nanmax(Z))
            z_mean = float(np.nanmean(Z))
            z_nan = int(np.isnan(Z).sum())
            print(f"\n  [Z stats] min={z_min:.6f} max={z_max:.6f} mean={z_mean:.6f} n_nan={z_nan}")
            for thr0 in LATENT_THRESHOLDS:
                frac = float((Z > float(thr0)).mean())
                print(f"  [Z>thr] thr={thr0} frac_active={frac:.6f}")

        if K is None:
            K = int(Z.shape[1])

        # init counters once
        for thr in LATENT_THRESHOLDS:
            if tp[thr] is None:
                tp[thr] = np.zeros((K, T), dtype=np.int32)
                pred_pos[thr] = np.zeros((K,), dtype=np.int64)

        # accumulate
        for thr in LATENT_THRESHOLDS:
            A_bool = (Z > float(thr))
            pred_pos[thr] += A_bool.sum(axis=0).astype(np.int64)
            accumulate_tp_sparse_into_dense(tp[thr], A_bool, G_csr)

    print("\n[count sanity]")
    print(f"  true_pos sum = {int(true_pos.sum())}  (should be >0)")
    for thr in LATENT_THRESHOLDS:
        tp_thr = tp[thr]
        pred_thr = pred_pos[thr]
        if tp_thr is None or pred_thr is None:
            print(f"  thr={thr}: no data")
            continue
        print(f"  thr={thr}: pred_pos sum={int(pred_thr.sum())}, tp sum={int(tp_thr.sum())}")

        f1 = compute_f1_from_counts(tp_thr, pred_thr, true_pos)
        print(f"           f1 max={float(np.max(f1)):.6f}  f1 mean={float(np.mean(f1)):.6f}")

        # print a couple top hits globally
        flat = np.argmax(f1)
        k_best, t_best = np.unravel_index(flat, f1.shape)
        print(f"           best pair: latent={k_best} term={term_ids[t_best]} f1={float(f1[k_best, t_best]):.6f} "
              f"tp={int(tp_thr[k_best, t_best])} pred={int(pred_thr[k_best])} true={int(true_pos[t_best])}")

    # If everything is still zero:
    if int(true_pos.sum()) == 0:
        print("\n*** ROOT CAUSE LIKELY: your GT matrix has no positive labels for mapped rows (or mapping is inconsistent). ***")
    else:
        all_tp_zero = all(tp[thr] is not None and int(tp[thr].sum()) == 0 for thr in LATENT_THRESHOLDS)
        if all_tp_zero:
            print("\n*** ROOT CAUSE LIKELY: SAE outputs are all <= thresholds (or Z is NaN/zero). ***")
            print("Next actions: print Z stats on a batch (min/max/mean, pct>0) and confirm SAE checkpoint matches activations layer.")


def main():
    # Load adata and build symbol->ens map (for vocab decode path)
    print("[1] Load AnnData + symbol<->Ensembl mapping")
    adata = sc.read_h5ad(ADATA_PATH)
    sym2ens = build_symbol_to_ensembl_from_adata(adata)
    print(f"  adata n_obs={adata.n_obs} n_vars={adata.n_vars}")
    print(f"  sym2ens size={len(sym2ens)} sample={list(sym2ens.items())[:3]}")

    # Load GT
    print("\n[2] Load GT gene×term matrix")
    if not os.path.exists(GT_PATH):
        raise FileNotFoundError(f"GT_PATH not found: {GT_PATH}")
    gt_df = pd.read_csv(GT_PATH, index_col=0)
    print(f"  GT shape={gt_df.shape}")
    print(f"  GT genes sample={gt_df.index[:3].tolist()}")
    print(f"  GT terms sample={gt_df.columns[:3].tolist()}")

    # Load scGPT (for decoding vocab ids)
    print("\n[3] Load scGPT tokenizer for decoding token_ids (if needed)")
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    scgpt_adapter = ScGPTAdapter()
    handle = scgpt_adapter.load(ModelSpec(name="scgpt", checkpoint=SCGPT_CKPT, device=str(device)))
    decode_token = build_scgpt_gene_decoder(handle.tokenizer)
    print("  decode sanity:", [(x, decode_token(int(x))) for x in [2439, 2441, 60695]])

    # Pick layer
    print("\n[4] Pick debug layer")
    layer = DEBUG_LAYER
    if layer is None:
        layers = scgpt_adapter.list_layers(handle)
        if not layers:
            raise RuntimeError("No layers returned by adapter.list_layers(handle)")
        layer = layers[0]
    print(f"  layer={layer}")

    # List shards
    print("\n[5] List shards")
    shards = list_shards(layer)
    print(f"  shards={len(shards)}")
    if len(shards) == 0:
        raise RuntimeError(f"No shards found for layer={layer} under {OUT_ROOT}/activations/{layer}")

    # SAE ckpt path: allow formatting if you used DEBUG_LAYER in path
    sae_ckpt = SAE_CKPT
    if isinstance(sae_ckpt, str) and "{DEBUG_LAYER}" in sae_ckpt:
        sae_ckpt = sae_ckpt.format(DEBUG_LAYER=layer)

    # Run debug
    debug_alignment_one_layer(
        layer=layer,
        shards=shards,
        gt_df=gt_df,
        decode_token=decode_token,
        sym2ens=sym2ens,
        sae_ckpt=sae_ckpt if (sae_ckpt is not None and os.path.exists(sae_ckpt)) else None,
    )


if __name__ == "__main__":
    main()