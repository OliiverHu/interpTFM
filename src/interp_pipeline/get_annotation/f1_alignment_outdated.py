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


from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd


def counts_to_long_df(
    concept_ids: List[str],
    tp: np.ndarray,          # (K, C) or (C, K) - we will assume (K, C) below
    pred_pos: np.ndarray,    # (K,)
    true_pos: np.ndarray,    # (C,)
    threshold: float,
    split: str,
    feature_axis_is_latent: bool = True,
) -> pd.DataFrame:
    """
    Build the long-format table that your old reporting expects, per split+threshold:
      concept, feature, threshold_pct, precision, recall, f1, tp, fp, fn

    We standardize:
      - feature == latent index (int)
      - concept == term_id (string)
      - threshold_pct == threshold value (legacy name)
    """
    if not feature_axis_is_latent:
        # convert (C,K) -> (K,C)
        tp = tp.T

    K, C = tp.shape
    assert len(concept_ids) == C

    fp = pred_pos[:, None] - tp
    fn = true_pos[None, :] - tp

    tp_f = tp.astype(np.float32)
    fp_f = fp.astype(np.float32)
    fn_f = fn.astype(np.float32)
    eps = 1e-12

    precision = tp_f / (tp_f + fp_f + eps)
    recall = tp_f / (tp_f + fn_f + eps)
    f1 = (2 * precision * recall) / (precision + recall + eps)

    # Long table
    # This is large (K*C). For big C you might want to write per-concept topM only,
    # but to match your old logic, we keep full long.
    concept_col = np.repeat(np.array(concept_ids, dtype=object), K)
    feature_col = np.tile(np.arange(K, dtype=np.int32), C)

    tp_long = tp.reshape(-1, order="F")        # concept-major, then features
    fp_long = fp.reshape(-1, order="F")
    fn_long = fn.reshape(-1, order="F")
    prec_long = precision.reshape(-1, order="F")
    rec_long = recall.reshape(-1, order="F")
    f1_long = f1.reshape(-1, order="F")

    df = pd.DataFrame(
        {
            "concept": concept_col,
            "feature": feature_col,
            "threshold_pct": float(threshold),
            "precision": prec_long,
            "recall": rec_long,
            "f1": f1_long,
            "tp": tp_long.astype(np.int64),
            "fp": fp_long.astype(np.int64),
            "fn": fn_long.astype(np.int64),
            "split": split,
        }
    )
    return df


def select_top_feature_per_concept(
    df_valid: pd.DataFrame,
) -> pd.DataFrame:
    """
    Mirror your notebook logic:
      for each concept, pick the single best row by f1
    Tie-breakers: higher tp, then lower fp.
    """
    # sort so best is first
    df = df_valid.sort_values(["concept", "f1", "tp", "fp"], ascending=[True, False, False, True])
    top = df.drop_duplicates(subset=["concept"], keep="first").copy()
    return top[["concept", "feature", "threshold_pct"]]


def evaluate_selected_pairs_on_test(
    df_test: pd.DataFrame,
    selected: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join selected (concept, feature, threshold_pct) onto test results.
    Returns a dataframe like heldout_top_pairings.csv in your notebook.
    """
    key = ["concept", "feature", "threshold_pct"]
    merged = pd.merge(selected, df_test, on=key, how="left", suffixes=("", "_test"))
    return merged


import os
import json
import glob
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from scipy import sparse


def _list_shards(layer_dir: str) -> List[str]:
    shards = sorted([p for p in glob.glob(os.path.join(layer_dir, "shard_*")) if os.path.isdir(p)])
    return shards


def _split_shards(
    shards: List[str],
    valid_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 0,
) -> Dict[str, List[str]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(shards))
    rng.shuffle(idx)

    n = len(shards)
    n_test = int(round(test_frac * n))
    n_valid = int(round(valid_frac * n))

    test_idx = idx[:n_test]
    valid_idx = idx[n_test:n_test + n_valid]
    train_idx = idx[n_test + n_valid:]

    return {
        "train": [shards[i] for i in train_idx],
        "valid": [shards[i] for i in valid_idx],
        "test":  [shards[i] for i in test_idx],
    }


def _counts_to_long_df(
    concept_ids: List[str],
    tp: np.ndarray,          # (K, C)
    pred_pos: np.ndarray,    # (K,)
    true_pos: np.ndarray,    # (C,)
    threshold: float,
) -> pd.DataFrame:
    fp = pred_pos[:, None] - tp
    fn = true_pos[None, :] - tp

    tp_f = tp.astype(np.float32)
    fp_f = fp.astype(np.float32)
    fn_f = fn.astype(np.float32)
    eps = 1e-12

    precision = tp_f / (tp_f + fp_f + eps)
    recall = tp_f / (tp_f + fn_f + eps)
    f1 = (2 * precision * recall) / (precision + recall + eps)

    K, C = tp.shape
    assert len(concept_ids) == C

    # concept-major ordering: for each concept, list all features
    concept_col = np.repeat(np.array(concept_ids, dtype=object), K)
    feature_col = np.tile(np.arange(K, dtype=np.int32), C)

    return pd.DataFrame(
        {
            "concept": concept_col,
            "feature": feature_col,
            "threshold_pct": float(threshold),  # legacy name (it's a threshold value)
            "precision": precision.reshape(-1, order="F"),
            "recall": recall.reshape(-1, order="F"),
            "f1": f1.reshape(-1, order="F"),
            "tp": tp.reshape(-1, order="F").astype(np.int64),
            "fp": fp.reshape(-1, order="F").astype(np.int64),
            "fn": fn.reshape(-1, order="F").astype(np.int64),
        }
    )


def _filter_topM_per_concept_per_threshold(df: pd.DataFrame, M: int) -> pd.DataFrame:
    """
    Keep only top M rows per (concept, threshold_pct) by f1.
    Tie-break: higher tp, then lower fp.
    """
    if M is None or M <= 0:
        return df

    sort_cols = ["concept", "threshold_pct", "f1", "tp", "fp"]
    ascending = [True, True, False, False, True]
    df2 = df.sort_values(sort_cols, ascending=ascending, kind="mergesort")

    return (
        df2.groupby(["concept", "threshold_pct"], sort=False, group_keys=False)
           .head(M)
           .reset_index(drop=True)
    )


def _select_top_feature_per_concept(df_valid: pd.DataFrame) -> pd.DataFrame:
    """
    For each concept, choose the single best row by f1 (tie-break: tp desc, fp asc).
    Returns columns: concept, feature, threshold_pct
    """
    df = df_valid.sort_values(["concept", "f1", "tp", "fp"], ascending=[True, False, False, True], kind="mergesort")
    top = df.drop_duplicates(subset=["concept"], keep="first").copy()
    return top[["concept", "feature", "threshold_pct"]]


def heldout_report_for_layer(
    *,
    layer: str,
    store_root: str,
    gt_csv: str,
    sae_ckpt_path: str,
    out_dir: str,
    latent_thresholds: List[float],
    valid_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 0,
    topM_valid_per_concept_per_threshold: int = 200,
    batch_size: int = 8192,
    device: str = "cuda",
    tp_dtype=np.int32,
) -> None:
    """
    Heldout reporting:
      - split shards into valid/test
      - compute concept×feature metrics for each threshold
      - on VALID: optionally keep only top M features per (concept, threshold) before writing CSV
      - select best (feature, threshold) per concept on VALID
      - evaluate those same pairs on TEST
      - write:
          out_dir/valid/concept_f1_scores.csv
          out_dir/test/concept_f1_scores.csv
          out_dir/heldout_top_pairings.csv
          out_dir/heldout_summary.json

    Assumptions (per your decision):
      - gene-token only
      - token_ids in index.pt are Ensembl IDs (ENSG...)
    """
    os.makedirs(out_dir, exist_ok=True)

    # --- load GT ---
    gt_df = pd.read_csv(gt_csv, index_col=0)
    concept_ids = gt_df.columns.astype(str).tolist()
    GT = gt_df.values.astype(np.int8)           # (G, C)
    gt_row_for_ens = {g: i for i, g in enumerate(gt_df.index.astype(str).tolist())}
    C = GT.shape[1]

    # --- load SAE ---
    ckpt = torch.load(sae_ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise RuntimeError("SAE checkpoint must be dict with state_dict/d_in/n_latents.")
    from interp_pipeline.sae.sae_base import AutoEncoder
    sae = AutoEncoder(d_in=int(ckpt["d_in"]), n_latents=int(ckpt["n_latents"]))
    sae.load_state_dict(ckpt["state_dict"])
    dev = torch.device(device)
    sae = sae.to(dev).eval()
    K = int(ckpt["n_latents"])

    @torch.no_grad()
    def encode(x: torch.Tensor) -> torch.Tensor:
        out = sae(x)
        return out[1]

    # --- shards + split ---
    layer_dir = os.path.join(store_root, "activations", layer)
    shards = _list_shards(layer_dir)
    if not shards:
        raise RuntimeError(f"No shards under {layer_dir}")
    splits = _split_shards(shards, valid_frac=valid_frac, test_frac=test_frac, seed=seed)

    def run_counts(split_shards: List[str]) -> Dict[float, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        tp = {thr: np.zeros((K, C), dtype=tp_dtype) for thr in latent_thresholds}
        pred_pos = {thr: np.zeros((K,), dtype=np.int64) for thr in latent_thresholds}
        true_pos = np.zeros((C,), dtype=np.int64)

        for shard in tqdm(split_shards, desc="count", leave=False):
            acts_path = os.path.join(shard, "activations.pt")
            idx_path = os.path.join(shard, "index.pt")
            if not (os.path.exists(acts_path) and os.path.exists(idx_path)):
                continue

            X = torch.load(acts_path, map_location="cpu")
            idx = torch.load(idx_path, map_location="cpu")
            if idx.get("token_unit", "unknown") != "gene":
                continue
            token_ids = idx.get("token_ids", None)
            if token_ids is None or len(token_ids) != X.shape[0]:
                continue

            gt_rows = np.full((X.shape[0],), -1, dtype=np.int32)
            for i, tid in enumerate(token_ids):
                r = gt_row_for_ens.get(str(tid))
                if r is not None:
                    gt_rows[i] = r

            mask = gt_rows >= 0
            if not np.any(mask):
                continue

            X_use = X[mask].float()
            G_use = GT[gt_rows[mask]]  # (N_use, C)
            G_csr = sparse.csr_matrix(G_use)
            true_pos += np.asarray(G_csr.sum(axis=0)).ravel().astype(np.int64)

            # encode
            Zs = []
            for i0 in range(0, X_use.shape[0], batch_size):
                xb = X_use[i0:i0 + batch_size].to(dev, non_blocking=True)
                zb = encode(xb).detach().cpu()
                Zs.append(zb)
            Z = torch.cat(Zs, dim=0).numpy()  # (N_use, K)

            for thr in latent_thresholds:
                A_bool = (Z > float(thr))
                pred_pos[thr] += A_bool.sum(axis=0).astype(np.int64)
                # uses your existing sparse accumulator:
                accumulate_tp_sparse_into_dense(tp[thr], A_bool, G_csr)

        return {thr: (tp[thr], pred_pos[thr], true_pos) for thr in latent_thresholds}

    # --- VALID counts -> long table -> topM filter -> write ---
    valid_counts = run_counts(splits["valid"])
    valid_tables = []
    for thr, (tp_thr, pred_thr, true_thr) in valid_counts.items():
        valid_tables.append(_counts_to_long_df(concept_ids, tp_thr, pred_thr, true_thr, threshold=float(thr)))
    df_valid = pd.concat(valid_tables, ignore_index=True)

    df_valid = _filter_topM_per_concept_per_threshold(df_valid, topM_valid_per_concept_per_threshold)

    valid_dir = os.path.join(out_dir, "valid")
    os.makedirs(valid_dir, exist_ok=True)
    df_valid.to_csv(os.path.join(valid_dir, "concept_f1_scores.csv"), index=False)

    # --- TEST counts -> long table -> write (NO filtering) ---
    test_counts = run_counts(splits["test"])
    test_tables = []
    for thr, (tp_thr, pred_thr, true_thr) in test_counts.items():
        test_tables.append(_counts_to_long_df(concept_ids, tp_thr, pred_thr, true_thr, threshold=float(thr)))
    df_test = pd.concat(test_tables, ignore_index=True)

    test_dir = os.path.join(out_dir, "test")
    os.makedirs(test_dir, exist_ok=True)
    df_test.to_csv(os.path.join(test_dir, "concept_f1_scores.csv"), index=False)

    # --- heldout selection + eval ---
    selected = _select_top_feature_per_concept(df_valid)
    heldout = pd.merge(selected, df_test, on=["concept", "feature", "threshold_pct"], how="left")
    heldout.to_csv(os.path.join(out_dir, "heldout_top_pairings.csv"), index=False)

    mean_test_f1 = float(np.nanmean(heldout["f1"].to_numpy()))

    with open(os.path.join(out_dir, "heldout_summary.json"), "w") as f:
        json.dump(
            {
                "layer": layer,
                "valid_frac": valid_frac,
                "test_frac": test_frac,
                "seed": seed,
                "thresholds": [float(x) for x in latent_thresholds],
                "topM_valid_per_concept_per_threshold": int(topM_valid_per_concept_per_threshold),
                "mean_test_f1_over_concepts": mean_test_f1,
                "n_concepts": int(len(concept_ids)),
                "n_latents": int(K),
                "n_valid_shards": int(len(splits["valid"])),
                "n_test_shards": int(len(splits["test"])),
            },
            f,
            indent=2,
        )