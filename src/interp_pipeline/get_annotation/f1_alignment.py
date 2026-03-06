
from __future__ import annotations

"""
interp_pipeline.get_annotation.f1_alignment

Library module for F1 alignment + heldout reporting.

Key guarantees (avoids the "all zeros" failure mode):
- Activation shard index.pt token_ids may be:
    1) Ensembl IDs ("ENSG...")
    2) scGPT vocab indices (often strings of ints, e.g. "60695")
    3) gene symbols
  We robustly map all three to Ensembl IDs before GT lookup.

- Uses sparse accumulation for TP:
    TP = A^T @ G
  where A is (N,K) latent-active boolean, G is (N,T) gene×term CSR.

- Heldout reporting:
    * Split by gene (Ensembl) into train/valid/test.
    * VALID: keep only top-M features per concept per threshold before writing.
    * TEST: evaluate only those selected features (from VALID) per concept per threshold.

- DEV MODE:
    * dev_max_shards
    * dev_max_rows_per_split_per_shard
    * dev_only_valid
"""

import os
import json
import glob
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from tqdm import tqdm


# -----------------------
# Types
# -----------------------
BinarizeMethod = Literal["threshold"]


@dataclass(frozen=True)
class TokenMapper:
    """
    Maps token_ids stored in activation shards -> Ensembl gene IDs.

    decode_token:
      int(vocab_id) -> symbol or ENSG or special token
    sym2ens:
      symbol -> ENSG
    """
    decode_token: Optional[Callable[[int], str]]
    sym2ens: Dict[str, str]

    _special: Set[str] = None  # type: ignore

    def __post_init__(self) -> None:
        if self._special is None:
            object.__setattr__(self, "_special", {"<pad>", "<PAD>", "<cls>", "<CLS>", "PAD", "CLS"})

    @staticmethod
    def _is_int_str(x: Any) -> bool:
        try:
            return str(x).strip().isdigit()
        except Exception:
            return False

    def token_to_ens(self, tid: Any, decode_cache: Dict[int, str]) -> Optional[str]:
        s = str(tid).strip()
        if not s or s in self._special:
            return None

        # Case 1: already Ensembl
        if s.startswith("ENSG"):
            return s

        # Case 2: vocab id -> decode -> symbol -> ens
        if self._is_int_str(s):
            if self.decode_token is None:
                return None
            ii = int(s)
            sym = decode_cache.get(ii)
            if sym is None:
                sym = self.decode_token(ii)
                decode_cache[ii] = sym

            if not sym or sym in self._special:
                return None
            if sym.startswith("ENSG"):
                return sym
            return self.sym2ens.get(sym)

        # Case 3: treat as symbol
        if s.startswith("ENSG"):
            return s
        return self.sym2ens.get(s)


# -----------------------
# Sparse math
# -----------------------
def dense_bool_to_csr(A_bool: np.ndarray) -> sparse.csr_matrix:
    """
    Convert dense bool array (N,K) to CSR with 1s at True positions.
    """
    if A_bool.dtype != np.bool_:
        A_bool = A_bool.astype(bool, copy=False)
    rows, cols = np.where(A_bool)
    data = np.ones(rows.shape[0], dtype=np.int8)
    return sparse.csr_matrix((data, (rows, cols)), shape=A_bool.shape)


def accumulate_tp_sparse_into_dense(tp_dense: np.ndarray, A_bool: np.ndarray, G_csr: sparse.csr_matrix) -> None:
    """
    tp_dense: (K,T) int array, modified in-place
    A_bool:   (N,K) dense bool
    G_csr:    (N,T) csr, 0/1
    """
    A_csr = dense_bool_to_csr(A_bool)
    TP = (A_csr.T @ G_csr).tocoo()  # (K,T)
    tp_dense[TP.row, TP.col] += TP.data.astype(tp_dense.dtype, copy=False)


def compute_f1_from_counts(
    tp: np.ndarray,
    pred_pos: np.ndarray,
    true_pos: np.ndarray,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns precision, recall, f1, fp, fn with shapes (K,T).
    """
    tp_f = tp.astype(np.float32, copy=False)
    fp = (pred_pos[:, None] - tp).astype(np.int64, copy=False)
    fn = (true_pos[None, :] - tp).astype(np.int64, copy=False)
    fp_f = fp.astype(np.float32, copy=False)
    fn_f = fn.astype(np.float32, copy=False)

    precision = tp_f / (tp_f + fp_f + eps)
    recall = tp_f / (tp_f + fn_f + eps)
    f1 = (2.0 * precision * recall) / (precision + recall + eps)
    return precision, recall, f1, fp, fn


def top_hits_per_latent(
    *,
    f1: np.ndarray,
    tp: np.ndarray,
    pred_pos: np.ndarray,
    true_pos: np.ndarray,
    term_ids: Sequence[str],
    topn: int,
    threshold: float,
) -> List[Dict[str, Any]]:
    """
    Convenience utility: topN concepts per latent (for quick debugging).
    """
    rows: List[Dict[str, Any]] = []
    K, _T = f1.shape
    topn = int(topn)

    for k in range(K):
        best = np.argsort(-f1[k])[:topn]
        for j in best:
            rows.append(
                {
                    "latent": int(k),
                    "threshold": float(threshold),
                    "term_id": str(term_ids[j]),
                    "f1": float(f1[k, j]),
                    "tp": int(tp[k, j]),
                    "pred_pos": int(pred_pos[k]),
                    "true_pos": int(true_pos[j]),
                }
            )
    return rows


# -----------------------
# Heldout reporting
# -----------------------
def _split_genes(genes: Sequence[str], valid_frac: float, test_frac: float, seed: int) -> Tuple[Set[str], Set[str], Set[str]]:
    rng = np.random.default_rng(int(seed))
    uniq = np.array(sorted(set(genes)), dtype=object)
    rng.shuffle(uniq)
    n = len(uniq)
    n_test = int(round(n * float(test_frac)))
    n_valid = int(round(n * float(valid_frac)))

    test = set(uniq[:n_test].tolist())
    valid = set(uniq[n_test:n_test + n_valid].tolist())
    train = set(uniq[n_test + n_valid:].tolist())
    return train, valid, test


def _load_sym2ens_from_cosmx_adata(adata_path: str, symbol_col: str = "index") -> Dict[str, str]:
    import scanpy as sc  # local import
    adata = sc.read_h5ad(adata_path)
    if symbol_col not in adata.var.columns:
        raise ValueError(f"Expected adata.var['{symbol_col}'] to contain gene symbols.")
    ens = adata.var.index.astype(str).tolist()
    sym = adata.var[symbol_col].astype(str).tolist()
    return {s: e for e, s in zip(ens, sym)}


def _load_scgpt_decoder(scgpt_ckpt: str) -> Callable[[int], str]:
    from interp_pipeline.adapters.models.scgpt import ScGPTAdapter
    from interp_pipeline.adapters.model_base import ModelSpec

    adapter = ScGPTAdapter()
    handle = adapter.load(ModelSpec(name="scgpt", checkpoint=scgpt_ckpt, device="cpu", options={}))
    tok = handle.tokenizer

    for parent_attr in ["gene_vocab", "vocab"]:
        if hasattr(tok, parent_attr):
            v = getattr(tok, parent_attr)
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

    if hasattr(tok, "decode") and callable(getattr(tok, "decode")):
        def _decode_one(i: int) -> str:
            out = tok.decode([[int(i)]])
            if not out:
                return ""
            if isinstance(out[0], list):
                return str(out[0][0]) if out[0] else ""
            return str(out[0]) if out else ""
        return _decode_one

    raise RuntimeError("Cannot build scGPT decoder (no vocab itos / decode).")


def _load_sae(ckpt_path: str, device: torch.device) -> Tuple[torch.nn.Module, int, int]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise RuntimeError(f"SAE checkpoint must be dict with state_dict/d_in/n_latents. Got {type(ckpt)}")
    d_in = int(ckpt["d_in"])
    n_lat = int(ckpt["n_latents"])

    from interp_pipeline.sae.sae_base import AutoEncoder
    model = AutoEncoder(d_in=d_in, n_latents=n_lat)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).eval()
    return model, d_in, n_lat


@torch.no_grad()
def _sae_encode(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    if isinstance(out, (tuple, list)) and len(out) == 2 and isinstance(out[1], torch.Tensor):
        return out[1]
    raise RuntimeError("Unexpected SAE forward output; expected (x_hat, z).")


def heldout_report_for_layer(
    *,
    layer: str,
    store_root: str,
    gt_csv: str,
    sae_ckpt_path: str,
    out_dir: str,
    latent_thresholds: Sequence[float],
    # splits
    valid_frac: float = 0.10,
    test_frac: float = 0.10,
    seed: int = 0,
    # selection
    topM_valid_per_concept_per_threshold: int = 200,
    # runtime
    batch_size: int = 8192,
    device: str = "cuda",
    tp_dtype: Any = np.int32,
    # mapping requirements
    adata_path: Optional[str] = None,
    scgpt_ckpt: Optional[str] = None,
    adata_symbol_col: str = "index",
    # -------- DEV MODE --------
    dev_mode: bool = False,
    dev_max_shards: Optional[int] = 50,
    dev_max_rows_per_split_per_shard: Optional[int] = 4096,
    dev_only_valid: bool = False,
) -> None:
    """
    Heldout evaluation for one layer. Writes:
      - valid_concept_f1_scores.csv   (topM per concept per threshold)
      - test_concept_f1_scores.csv    (restricted to valid-selected features)
      - selected_features_valid.json
      - counts_summary.json

    DEV MODE:
      - limits #shards and #rows per shard per split to speed iteration
      - optionally runs only VALID

    IMPORTANT: If token_ids in shards are vocab ids, you must pass adata_path and scgpt_ckpt.
    """
    os.makedirs(out_dir, exist_ok=True)

    # ---- GT ----
    gt_df = pd.read_csv(gt_csv, index_col=0)
    gt_genes = gt_df.index.astype(str).tolist()
    term_ids = gt_df.columns.astype(str).tolist()
    GT = gt_df.values.astype(np.int8, copy=False)  # (G,T)
    gt_row_for_ens = {g: i for i, g in enumerate(gt_genes)}
    T_terms = GT.shape[1]

    # ---- token mapping ----
    sym2ens: Dict[str, str] = {}
    decode_token: Optional[Callable[[int], str]] = None

    if adata_path is not None:
        sym2ens = _load_sym2ens_from_cosmx_adata(adata_path, symbol_col=adata_symbol_col)
    if scgpt_ckpt is not None:
        decode_token = _load_scgpt_decoder(scgpt_ckpt)

    mapper = TokenMapper(decode_token=decode_token, sym2ens=sym2ens)
    decode_cache: Dict[int, str] = {}

    # ---- locate shards ----
    acts_root = os.path.join(store_root, "activations", layer)
    shards = sorted([p for p in glob.glob(os.path.join(acts_root, "shard_*")) if os.path.isdir(p)])
    if not shards:
        raise RuntimeError(f"No activation shards found under {acts_root}")

    if dev_mode and dev_max_shards is not None:
        shards = shards[: int(dev_max_shards)]

    # ---- discover genes present (for split) ----
    observed_genes: List[str] = []
    for shard in shards:
        idx_path = os.path.join(shard, "index.pt")
        acts_path = os.path.join(shard, "activations.pt")
        if not (os.path.exists(idx_path) and os.path.exists(acts_path)):
            continue
        idx = torch.load(idx_path, map_location="cpu")
        if idx.get("token_unit", "unknown") != "gene":
            continue
        token_ids = idx.get("token_ids", None)
        if token_ids is None:
            continue
        for tid in token_ids[:2000]:
            ens = mapper.token_to_ens(tid, decode_cache)
            if ens is not None and ens in gt_row_for_ens:
                observed_genes.append(ens)

    if not observed_genes:
        raise RuntimeError(
            "No genes observed in shards that map to GT rows. "
            "If your token_ids are vocab ids, pass adata_path+scgpt_ckpt."
        )

    _, valid_genes, test_genes = _split_genes(observed_genes, valid_frac=valid_frac, test_frac=test_frac, seed=seed)

    # ---- SAE ----
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    sae, d_in, K = _load_sae(sae_ckpt_path, device=dev)

    # ---- counters (valid/test only) ----
    thresholds = [float(x) for x in latent_thresholds]
    splits = ("valid",) if (dev_mode and dev_only_valid) else ("valid", "test")

    tp: Dict[str, Dict[float, np.ndarray]] = {
        sp: {thr: np.zeros((K, T_terms), dtype=tp_dtype) for thr in thresholds}
        for sp in splits
    }
    pred_pos: Dict[str, Dict[float, np.ndarray]] = {
        sp: {thr: np.zeros((K,), dtype=np.int64) for thr in thresholds}
        for sp in splits
    }
    true_pos: Dict[str, np.ndarray] = {sp: np.zeros((T_terms,), dtype=np.int64) for sp in splits}

    rows_total = 0
    rows_mapped_to_gt = 0

    split_iter = (("valid", 0),) if (dev_mode and dev_only_valid) else (("valid", 0), ("test", 1))

    # ---- stream shards ----
    for shard in tqdm(shards, desc=f"heldout:{layer}"):
        acts_path = os.path.join(shard, "activations.pt")
        idx_path = os.path.join(shard, "index.pt")
        if not (os.path.exists(acts_path) and os.path.exists(idx_path)):
            continue

        X = torch.load(acts_path, map_location="cpu")
        if not isinstance(X, torch.Tensor) or X.ndim != 2:
            continue
        idx = torch.load(idx_path, map_location="cpu")
        if idx.get("token_unit", "unknown") != "gene":
            continue
        token_ids = idx.get("token_ids", None)
        if token_ids is None or len(token_ids) != X.shape[0]:
            continue

        rows_total += int(X.shape[0])

        gt_rows = np.full((X.shape[0],), -1, dtype=np.int32)
        row_split = np.full((X.shape[0],), -1, dtype=np.int8)  # 0=valid, 1=test, -1=ignore

        for i, tid in enumerate(token_ids):
            ens = mapper.token_to_ens(tid, decode_cache)
            if ens is None:
                continue
            r = gt_row_for_ens.get(ens)
            if r is None:
                continue
            gt_rows[i] = r
            if ens in valid_genes:
                row_split[i] = 0
            elif ens in test_genes:
                row_split[i] = 1

        mask_any = gt_rows >= 0
        if not np.any(mask_any):
            continue
        rows_mapped_to_gt += int(mask_any.sum())

        for sp, sp_id in split_iter:
            m = (row_split == sp_id) & mask_any
            if not np.any(m):
                continue

            if dev_mode and dev_max_rows_per_split_per_shard is not None:
                idxs = np.flatnonzero(m)
                if idxs.size > int(dev_max_rows_per_split_per_shard):
                    idxs = idxs[: int(dev_max_rows_per_split_per_shard)]
                    m2 = np.zeros_like(m, dtype=bool)
                    m2[idxs] = True
                    m = m2

            X_use = X[m].float()
            if X_use.shape[1] != d_in:
                raise RuntimeError(f"SAE d_in={d_in} but activations have H={X_use.shape[1]} for layer={layer}")

            G_use = GT[gt_rows[m]]  # (N,T) dense
            G_csr = sparse.csr_matrix(G_use)
            true_pos[sp] += np.asarray(G_csr.sum(axis=0)).ravel().astype(np.int64)

            # encode in batches
            Z_chunks: List[torch.Tensor] = []
            for i0 in range(0, X_use.shape[0], int(batch_size)):
                xb = X_use[i0:i0 + int(batch_size)].to(dev, non_blocking=True)
                zb = _sae_encode(sae, xb).detach().cpu()
                Z_chunks.append(zb)
            Z = torch.cat(Z_chunks, dim=0).numpy()  # (N,K)

            for thr in thresholds:
                A_bool = (Z > float(thr))
                pred_pos[sp][thr] += A_bool.sum(axis=0).astype(np.int64)
                accumulate_tp_sparse_into_dense(tp[sp][thr], A_bool, G_csr)

    # ---- build VALID table + selection ----
    def _make_rows_for_split(
        split_name: str,
        restrict: Optional[Dict[str, Dict[str, List[int]]]] = None,
    ) -> pd.DataFrame:
        all_rows: List[Dict[str, Any]] = []

        for thr in thresholds:
            precision, recall, f1, fp, fn = compute_f1_from_counts(
                tp=tp[split_name][thr],
                pred_pos=pred_pos[split_name][thr],
                true_pos=true_pos[split_name],
            )

            if restrict is not None:
                term_to_feats = restrict.get(str(thr), {})
            else:
                term_to_feats = None

            for j, term_id in enumerate(term_ids):
                if term_to_feats is not None:
                    feats = term_to_feats.get(term_id, [])
                    if not feats:
                        continue
                else:
                    feats = list(range(K))

                feats_sorted = sorted(
                    feats,
                    key=lambda k: (float(f1[k, j]), int(tp[split_name][thr][k, j])),
                    reverse=True,
                )
                if split_name == "valid":
                    feats_sorted = feats_sorted[: int(topM_valid_per_concept_per_threshold)]

                for k in feats_sorted:
                    all_rows.append(
                        {
                            "concept": term_id,
                            "feature": int(k),
                            "threshold_pct": float(thr),
                            "precision": float(precision[k, j]),
                            "recall": float(recall[k, j]),
                            "f1": float(f1[k, j]),
                            "tp": int(tp[split_name][thr][k, j]),
                            "fp": int(fp[k, j]),
                            "fn": int(fn[k, j]),
                        }
                    )

        return pd.DataFrame(all_rows)

    df_valid = _make_rows_for_split("valid", restrict=None)

    selected: Dict[str, Dict[str, List[int]]] = {}
    for thr in thresholds:
        thr_key = str(thr)
        selected[thr_key] = {}
        df_thr = df_valid[df_valid["threshold_pct"] == float(thr)]
        if df_thr.empty:
            continue
        for term_id, g in df_thr.groupby("concept"):
            gg = g.sort_values(["f1", "tp"], ascending=False)
            selected[thr_key][term_id] = gg["feature"].astype(int).tolist()

    if dev_mode and dev_only_valid:
        df_test = pd.DataFrame(columns=df_valid.columns)
    else:
        df_test = _make_rows_for_split("test", restrict=selected)

    # ---- write outputs ----
    df_valid.to_csv(os.path.join(out_dir, "valid_concept_f1_scores.csv"), index=False)
    df_test.to_csv(os.path.join(out_dir, "test_concept_f1_scores.csv"), index=False)
    with open(os.path.join(out_dir, "selected_features_valid.json"), "w") as f:
        json.dump(selected, f, indent=2)

    with open(os.path.join(out_dir, "counts_summary.json"), "w") as f:
        json.dump(
            {
                "layer": layer,
                "n_terms": int(T_terms),
                "n_latents": int(K),
                "thresholds": thresholds,
                "valid_frac": float(valid_frac),
                "test_frac": float(test_frac),
                "seed": int(seed),
                "rows_total": int(rows_total),
                "rows_mapped_to_gt": int(rows_mapped_to_gt),
                "topM_valid_per_concept_per_threshold": int(topM_valid_per_concept_per_threshold),
                "dev_mode": bool(dev_mode),
                "dev_max_shards": None if dev_max_shards is None else int(dev_max_shards),
                "dev_max_rows_per_split_per_shard": None if dev_max_rows_per_split_per_shard is None else int(dev_max_rows_per_split_per_shard),
                "dev_only_valid": bool(dev_only_valid),
                "note": "token_ids may be ENSG, vocab-id, or symbol; mapping handled by TokenMapper",
            },
            f,
            indent=2,
        )
