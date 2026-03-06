from __future__ import annotations

import os
import glob
from typing import Any, Dict, Callable, List, Optional

import numpy as np
import pandas as pd
import torch
import scanpy as sc
from tqdm import tqdm
from scipy import sparse

from interp_pipeline.adapters.models.scgpt import ScGPTAdapter
from interp_pipeline.adapters.model_base import ModelSpec


# ============================================================
# EDIT HERE
# ============================================================
ADATA_PATH = "/maiziezhou_lab2/yunfei/Projects/FM_temp/InterPLM/interplm/ge_shards/cosmx_human_lung_sec8.h5ad"
SCGPT_CKPT = "/maiziezhou_lab/zihang/SpatialFoundationModel/eval/scGPT/whole-human-pretrain"

RUNS_ROOT = "runs/full_scgpt_cosmx"
GT_PATH = os.path.join(RUNS_ROOT, "gprofiler", "gprofiler_binary_gene_by_term.csv")

LAYER = "layer_4.norm2"              # baseline per-layer; loop it later if you want
ACTS_ROOT = os.path.join(RUNS_ROOT, "activations", LAYER)

OUT_DIR = os.path.join(RUNS_ROOT, "activation_f1_baseline", LAYER)
os.makedirs(OUT_DIR, exist_ok=True)

MAX_SHARDS = 20
SHARD_OFFSET = 0

BATCH_SIZE = 8192  # only used if we chunk ops later
THRESHOLDS = [0.0, 0.15, 0.3, 0.6]   # NOTE: raw activations can be negative; see USE_RELU
USE_RELU = True  # if True, use max(X,0) before thresholding (often makes thresholds meaningful)
DO_ZSCORE = True  # normalize each dim with mean/std over mapped rows (streaming would be better for full run)
# ============================================================


def list_shards(root: str) -> List[str]:
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
    if "index" not in adata.var.columns:
        raise ValueError("Expected adata.var['index'] to contain gene symbols.")
    ens = adata.var.index.astype(str).tolist()
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[1] Load AnnData + mappings")
    adata = sc.read_h5ad(ADATA_PATH)
    sym2ens = build_symbol_to_ensembl_from_adata(adata)

    print("[2] Load GT")
    gt_df = pd.read_csv(GT_PATH, index_col=0)
    GT = gt_df.values.astype(np.int8)
    gt_row_for_ens = {g: i for i, g in enumerate(gt_df.index.astype(str).tolist())}
    term_ids = gt_df.columns.astype(str).tolist()
    T = GT.shape[1]

    print("[3] Load scGPT tokenizer (decode vocab ids)")
    scgpt_adapter = ScGPTAdapter()
    handle = scgpt_adapter.load(ModelSpec(name="scgpt", checkpoint=SCGPT_CKPT, device=str(device)))
    decode_token = build_scgpt_gene_decoder(handle.tokenizer)
    decode_cache: Dict[int, str] = {}

    print("[4] List shards")
    shards = list_shards(ACTS_ROOT)
    print(f"  shards={len(shards)}  root={ACTS_ROOT}")
    if not shards:
        raise RuntimeError("No shards found.")

    # First pass: gather mapped activations for z-score stats (debug-scale; OK for limited shards)
    # If DO_ZSCORE=False, we skip.
    mean = std = None
    if DO_ZSCORE:
        print("[5] Estimate mean/std (cheap pass; limited shards recommended)")
        sums = None
        sums2 = None
        n = 0
        for shard in tqdm(shards, desc="meanstd"):
            X = torch.load(os.path.join(shard, "activations.pt"), map_location="cpu").float()
            idx = torch.load(os.path.join(shard, "index.pt"), map_location="cpu")
            token_ids = idx.get("token_ids", None)
            if token_ids is None or len(token_ids) != X.shape[0]:
                continue
            gt_rows = np.full((X.shape[0],), -1, dtype=np.int32)
            for i, tid in enumerate(token_ids):
                s = str(tid).strip()
                if s.startswith("ENSG"):
                    ens = s
                else:
                    if is_int_str(s):
                        sym = decode_cache.get(int(s))
                        if sym is None:
                            sym = decode_token(int(s))
                            decode_cache[int(s)] = sym
                    else:
                        sym = s
                    ens = sym2ens.get(sym) if sym in sym2ens else (sym if sym.startswith("ENSG") else None)
                if ens is None:
                    continue
                r = gt_row_for_ens.get(ens)
                if r is not None:
                    gt_rows[i] = r
            m = gt_rows >= 0
            if not np.any(m):
                continue
            Xu = X[m]
            if USE_RELU:
                Xu = torch.relu(Xu)
            if sums is None:
                H = Xu.shape[1]
                sums = torch.zeros(H)
                sums2 = torch.zeros(H)
            sums += Xu.sum(dim=0)
            sums2 += (Xu * Xu).sum(dim=0)
            n += Xu.shape[0]
        mean = (sums / max(n, 1)).numpy()
        var = (sums2 / max(n, 1)).numpy() - mean * mean
        std = np.sqrt(np.clip(var, 1e-12, None))
        print(f"  mean/std ready using n={n} rows")

    print("[6] Accumulate F1 counts for activation dims")
    tp = {thr: None for thr in THRESHOLDS}
    pred_pos = {thr: None for thr in THRESHOLDS}
    true_pos = np.zeros((T,), dtype=np.int64)

    H_observed = None

    for shard in tqdm(shards, desc="shards"):
        X = torch.load(os.path.join(shard, "activations.pt"), map_location="cpu").float()
        idx = torch.load(os.path.join(shard, "index.pt"), map_location="cpu")
        token_ids = idx.get("token_ids", None)
        if token_ids is None or len(token_ids) != X.shape[0]:
            continue

        gt_rows = np.full((X.shape[0],), -1, dtype=np.int32)
        for i, tid in enumerate(token_ids):
            s = str(tid).strip()
            if s.startswith("ENSG"):
                ens = s
            else:
                if is_int_str(s):
                    sym = decode_cache.get(int(s))
                    if sym is None:
                        sym = decode_token(int(s))
                        decode_cache[int(s)] = sym
                else:
                    sym = s
                ens = sym2ens.get(sym) if sym in sym2ens else (sym if sym.startswith("ENSG") else None)
            if ens is None:
                continue
            r = gt_row_for_ens.get(ens)
            if r is not None:
                gt_rows[i] = r

        m = gt_rows >= 0
        if not np.any(m):
            continue

        Xu = X[m].numpy()
        if USE_RELU:
            Xu = np.maximum(Xu, 0.0)
        if DO_ZSCORE and mean is not None and std is not None:
            Xu = (Xu - mean[None, :]) / std[None, :]

        G = GT[gt_rows[m]]
        G_csr = sparse.csr_matrix(G)
        true_pos += np.asarray(G_csr.sum(axis=0)).ravel().astype(np.int64)

        if H_observed is None:
            H_observed = int(Xu.shape[1])
            for thr in THRESHOLDS:
                tp[thr] = np.zeros((H_observed, T), dtype=np.int32)
                pred_pos[thr] = np.zeros((H_observed,), dtype=np.int64)

        for thr in THRESHOLDS:
            A_bool = (Xu > float(thr))
            pred_pos[thr] += A_bool.sum(axis=0).astype(np.int64)
            accumulate_tp_sparse_into_dense(tp[thr], A_bool, G_csr)

    if H_observed is None:
        raise RuntimeError("No usable data.")

    print("[7] Write best-term-per-activation-dim outputs")
    rows = []
    for thr in THRESHOLDS:
        f1 = compute_f1_from_counts(tp[thr], pred_pos[thr], true_pos)
        best_term_idx = np.argmax(f1, axis=1)  # per dim
        best_f1 = f1[np.arange(H_observed), best_term_idx]
        for h in range(H_observed):
            rows.append(
                {
                    "layer": LAYER,
                    "threshold": float(thr),
                    "feature": int(h),
                    "best_term_id": term_ids[int(best_term_idx[h])],
                    "best_f1": float(best_f1[h]),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "activation_per_feature_best.csv"), index=False)
    print("  wrote:", os.path.join(OUT_DIR, "activation_per_feature_best.csv"))


if __name__ == "__main__":
    main()