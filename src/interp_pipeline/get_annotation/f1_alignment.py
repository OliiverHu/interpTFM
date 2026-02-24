from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Literal

import numpy as np
from scipy import sparse
import numpy as np

try:
    from scipy import sparse
except Exception:
    sparse = None


BinarizeMethod = Literal["topk", "percentile", "threshold"]


@dataclass(frozen=True)
class AlignSpec:
    binarize: BinarizeMethod = "topk"
    topk: int = 30                 # for topk
    percentile: float = 99.0       # for percentile (per-latent)
    threshold: float = 0.0         # for threshold

    min_tp: int = 3
    min_f1: float = 0.1


@dataclass(frozen=True)
class AlignResult:
    f1: np.ndarray          # [n_latents, n_terms]
    precision: np.ndarray   # [n_latents, n_terms]
    recall: np.ndarray      # [n_latents, n_terms]
    tp: np.ndarray          # [n_latents, n_terms]
    fp: np.ndarray          # [n_latents, n_terms]
    fn: np.ndarray          # [n_latents, n_terms]


def _to_bool_matrix(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x.astype(bool)
    if sparse is not None and sparse.issparse(x):
        return x.astype(bool)
    return np.asarray(x).astype(bool)


def binarize_latents(
    gene_by_latent_scores: np.ndarray,
    method: BinarizeMethod,
    topk: int = 30,
    percentile: float = 99.0,
    threshold: float = 0.0,
) -> np.ndarray:
    """
    Convert gene×latent scores into a boolean selection matrix gene×latent.

    - topk: per latent, select top-k genes by score
    - percentile: per latent, select genes above percentile
    - threshold: select genes with score >= threshold
    """
    S = np.asarray(gene_by_latent_scores)
    if S.ndim != 2:
        raise ValueError(f"expected 2D scores [n_genes, n_latents], got {S.shape}")

    n_genes, n_latents = S.shape
    out = np.zeros((n_genes, n_latents), dtype=bool)

    if method == "threshold":
        out = S >= float(threshold)
        return out

    if method == "percentile":
        p = float(percentile)
        for j in range(n_latents):
            thr = np.percentile(S[:, j], p)
            out[:, j] = S[:, j] >= thr
        return out

    if method == "topk":
        k = int(topk)
        if k <= 0:
            raise ValueError("topk must be > 0")
        k = min(k, n_genes)
        # argsort is fine at 960 genes; very fast
        for j in range(n_latents):
            idx = np.argpartition(S[:, j], -k)[-k:]
            out[idx, j] = True
        return out

    raise ValueError(f"unknown binarize method: {method}")


def compute_prf1(
    gene_by_latent_bin: np.ndarray,
    gene_by_term_bin: np.ndarray,
    eps: float = 1e-12,
) -> AlignResult:
    """
    Compute TP/FP/FN and precision/recall/F1 for every latent×term.

    Inputs:
      gene_by_latent_bin: [n_genes, n_latents] boolean
      gene_by_term_bin:   [n_genes, n_terms] boolean
    """
    A = _to_bool_matrix(gene_by_latent_bin)  # genes×latents
    G = _to_bool_matrix(gene_by_term_bin)    # genes×terms

    if A.shape[0] != G.shape[0]:
        raise ValueError(f"gene dimension mismatch: A={A.shape}, G={G.shape}")

    # Use int dot products (fast at these sizes)
    A_i = A.astype(np.int32)
    G_i = G.astype(np.int32)

    # TP: latents×terms = (genes×latents)^T @ (genes×terms)
    tp = (A_i.T @ G_i).astype(np.int32)

    # Support sizes
    pred_pos = A_i.sum(axis=0).reshape(-1, 1).astype(np.int32)   # latents×1
    true_pos = G_i.sum(axis=0).reshape(1, -1).astype(np.int32)   # 1×terms

    fp = (pred_pos - tp).astype(np.int32)
    fn = (true_pos - tp).astype(np.int32)

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = (2 * precision * recall) / (precision + recall + eps)

    return AlignResult(
        f1=f1,
        precision=precision,
        recall=recall,
        tp=tp,
        fp=fp,
        fn=fn,
    )


def rank_terms_for_latents(
    res: AlignResult,
    term_ids: List[str],
    spec: AlignSpec,
    topn: int = 10,
) -> Dict[int, List[Tuple[str, float, int]]]:
    """
    Returns for each latent index:
      list of (term_id, f1, tp) sorted by descending f1 then tp,
      filtered by min_tp and min_f1.
    """
    n_latents, n_terms = res.f1.shape
    if len(term_ids) != n_terms:
        raise ValueError(f"term_ids length mismatch: {len(term_ids)} vs {n_terms}")

    out: Dict[int, List[Tuple[str, float, int]]] = {}
    for i in range(n_latents):
        f1_row = res.f1[i]
        tp_row = res.tp[i]

        # candidate indices passing filters
        cand = np.where((tp_row >= spec.min_tp) & (f1_row >= spec.min_f1))[0]
        if cand.size == 0:
            out[i] = []
            continue

        # sort by (f1 desc, tp desc)
        cand_sorted = sorted(
            cand.tolist(),
            key=lambda j: (float(f1_row[j]), int(tp_row[j])),
            reverse=True,
        )[:topn]

        out[i] = [(term_ids[j], float(f1_row[j]), int(tp_row[j])) for j in cand_sorted]
    return out


def dense_bool_to_csr(X_bool: np.ndarray) -> sparse.csr_matrix:
    """
    Convert dense boolean array (N, K) to CSR with int8 data.
    """
    if X_bool.dtype != np.bool_:
        X_bool = X_bool.astype(bool, copy=False)
    rows, cols = np.where(X_bool)
    data = np.ones(rows.shape[0], dtype=np.int8)
    return sparse.csr_matrix((data, (rows, cols)), shape=X_bool.shape)


def accumulate_tp_sparse_into_dense(
    tp_dense: np.ndarray,
    A_bool: np.ndarray,
    G_csr: sparse.csr_matrix,
) -> None:
    """
    Given:
      A_bool: (N_use, K) dense boolean activations for latents
      G_csr:  (N_use, T) CSR sparse labels (gene×term duplicated to token rows)
    Computes TP = A^T @ G and accumulates into tp_dense (K, T) in-place.

    tp_dense should be an integer ndarray (e.g. int32).
    """
    A_csr = dense_bool_to_csr(A_bool)
    TP_csr = (A_csr.T @ G_csr)  # (K, T) sparse
    TP = TP_csr.tocoo()
    tp_dense[TP.row, TP.col] += TP.data.astype(tp_dense.dtype, copy=False)


def compute_f1_from_counts(
    tp: np.ndarray,
    pred_pos: np.ndarray,
    true_pos: np.ndarray,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute precision/recall/F1 from:
      tp: (K, T)
      pred_pos: (K,)
      true_pos: (T,)
    Returns:
      precision: (K, T)
      recall:    (K, T)
      f1:        (K, T)
    """
    tp_f = tp.astype(np.float32)
    fp_f = (pred_pos[:, None] - tp).astype(np.float32)
    fn_f = (true_pos[None, :] - tp).astype(np.float32)

    precision = tp_f / (tp_f + fp_f + eps)
    recall = tp_f / (tp_f + fn_f + eps)
    f1 = (2 * precision * recall) / (precision + recall + eps)
    return precision, recall, f1


def top_hits_per_latent(
    f1: np.ndarray,
    tp: np.ndarray,
    pred_pos: np.ndarray,
    true_pos: np.ndarray,
    term_ids: List[str],
    topn: int = 10,
    threshold: Optional[float] = None,
) -> List[dict]:
    """
    Produce a long-form list of top hits per latent, suitable for CSV.

    Returns list of dicts:
      latent, term_id, f1, tp, pred_pos, true_pos, (optional threshold)
    """
    K, T = f1.shape
    rows: List[dict] = []
    for k in range(K):
        best = np.argsort(-f1[k])[:topn]
        for j in best:
            d = {
                "latent": int(k),
                "term_id": term_ids[j],
                "f1": float(f1[k, j]),
                "tp": int(tp[k, j]),
                "pred_pos": int(pred_pos[k]),
                "true_pos": int(true_pos[j]),
            }
            if threshold is not None:
                d["threshold"] = float(threshold)
            rows.append(d)
    return rows