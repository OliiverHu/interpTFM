from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm


@dataclass(frozen=True)
class CellHeldoutF1Config:
    model: str
    layer: str
    store_root: str
    sae_ckpt_path: str
    gt_csv: str
    adata_path: str
    out_dir: str
    token_value: Optional[str] = None
    activation_pooling: str = "token"  # token or mean
    c2s_cell_pooling: str = "mean"     # mean or max for token-to-cell latent pooling
    split_seed: int = 0
    train_frac: float = 0.7
    valid_frac: float = 0.2
    test_frac: float = 0.1
    concept_score_quantile: float = 0.75
    latent_thresholds: Tuple[float, ...] = (0.0, 0.15, 0.3, 0.6)
    batch_size: int = 8192
    max_shards: Optional[int] = None
    device: str = "cuda"
    min_concept_genes: int = 3
    min_pos_valid: int = 20
    min_pos_test: int = 20
    min_neg_valid: int = 20
    min_neg_test: int = 20
    max_concepts: Optional[int] = None
    concept_chunk_size: int = 25
    dedupe_identical_concepts: bool = True


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def _load_state_dict(ckpt_path: Path) -> Dict[str, torch.Tensor]:
    obj = torch.load(ckpt_path, map_location="cpu")
    if isinstance(obj, dict):
        for k in ["state_dict", "model_state_dict", "model"]:
            if k in obj and isinstance(obj[k], dict):
                inner = obj[k]
                if all(torch.is_tensor(v) for v in inner.values()):
                    return inner
        if all(torch.is_tensor(v) for v in obj.values()):
            return obj
    raise TypeError(f"Unsupported checkpoint format in {ckpt_path}")


def _get_encoder_params(state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    w = None
    b_enc = None
    x_bias = None
    for k in ["encoder.weight", "W_enc", "enc.weight"]:
        if k in state:
            w = state[k].float()
            break
    for k in ["encoder.bias", "b_enc", "enc.bias"]:
        if k in state:
            b_enc = state[k].float()
            break
    for k in ["bias", "x_bias"]:
        if k in state:
            x_bias = state[k].float()
            break
    if w is None:
        raise KeyError(f"Could not find encoder weights. First keys: {list(state.keys())[:30]}")
    return w, b_enc, x_bias


def _encode_relu(x: torch.Tensor, w: torch.Tensor, b_enc: Optional[torch.Tensor], x_bias: Optional[torch.Tensor]) -> torch.Tensor:
    if x_bias is not None:
        x = x - x_bias
    z = x @ w.T
    if b_enc is not None:
        z = z + b_enc
    return torch.relu(z)


def _normalize_index(v: Any) -> str:
    return str(v)


def _build_cell_sae_matrix(cfg: CellHeldoutF1Config, obs_names: Sequence[str]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Build one cell-level SAE latent matrix [n_cells, n_latents].

    token pooling: select rows whose token_id matches cfg.token_value.
    mean pooling: encode all token rows, aggregate SAE latents by example_id.
    """
    from interp_pipeline.io.activation_store import ActivationStore, ActivationStoreSpec

    store = ActivationStore(ActivationStoreSpec(root=cfg.store_root))
    shards = store.list_shards(cfg.layer)
    if cfg.max_shards is not None:
        shards = shards[: int(cfg.max_shards)]
    if not shards:
        raise RuntimeError(f"No activation shards for {cfg.model} {cfg.layer} under {cfg.store_root}")

    state = _load_state_dict(Path(cfg.sae_ckpt_path))
    w_cpu, b_cpu, x_bias_cpu = _get_encoder_params(state)
    n_latents, d_in = int(w_cpu.shape[0]), int(w_cpu.shape[1])
    device = torch.device(cfg.device if cfg.device and torch.cuda.is_available() and str(cfg.device).startswith("cuda") else "cpu")
    w = w_cpu.to(device)
    b_enc = b_cpu.to(device) if b_cpu is not None else None
    x_bias = x_bias_cpu.to(device) if x_bias_cpu is not None else None

    obs_to_i = {_normalize_index(x): i for i, x in enumerate(obs_names)}
    n_obs = len(obs_names)
    Z_sum = np.zeros((n_obs, n_latents), dtype=np.float32)
    counts = np.zeros(n_obs, dtype=np.int32)
    if cfg.activation_pooling == "mean":
        if cfg.c2s_cell_pooling == "max":
            Z_sum[:] = -np.inf
    else:
        if cfg.token_value is None:
            raise ValueError("token activation_pooling requires token_value")

    rows_seen = 0
    rows_used = 0
    rows_unmatched = 0
    token_rows_seen = 0

    batch_iter = store.iter_token_batches(cfg.layer, batch_size=int(cfg.batch_size), shuffle_shards=False)
    batch_iter = tqdm(
        batch_iter,
        desc=f"cell-f1 encode:{cfg.model}:{cfg.layer}",
        unit="batch",
        dynamic_ncols=True,
    )

    for xb, idx in batch_iter:
        if xb.ndim != 2:
            raise ValueError(f"Expected 2D activation batch, got {tuple(xb.shape)}")
        if int(xb.shape[1]) != d_in:
            raise ValueError(f"SAE input dim mismatch: activations {xb.shape[1]} vs encoder {d_in}")
        ex_ids = [_normalize_index(x) for x in idx["example_ids"]]
        token_ids_raw = idx.get("token_ids")
        token_ids = None if token_ids_raw is None else [_normalize_index(x) for x in token_ids_raw]
        rows_seen += int(xb.shape[0])

        if cfg.activation_pooling == "token":
            if token_ids is None:
                raise ValueError("token pooling requested but index has token_ids=None")
            mask = [t == str(cfg.token_value) for t in token_ids]
            token_rows_seen += int(sum(mask))
            if not any(mask):
                continue
            keep = [i for i, m in enumerate(mask) if m]
            xb_use = xb[keep]
            ex_use = [ex_ids[i] for i in keep]
        elif cfg.activation_pooling == "mean":
            xb_use = xb
            ex_use = ex_ids
        else:
            raise ValueError(f"Unsupported activation_pooling={cfg.activation_pooling!r}")

        with torch.no_grad():
            z = _encode_relu(xb_use.float().to(device), w, b_enc, x_bias).detach().cpu().numpy().astype(np.float32)
        for row_z, ex in zip(z, ex_use):
            j = obs_to_i.get(ex)
            if j is None:
                rows_unmatched += 1
                continue
            if cfg.activation_pooling == "mean" and cfg.c2s_cell_pooling == "max":
                Z_sum[j] = np.maximum(Z_sum[j], row_z)
            else:
                Z_sum[j] += row_z
            counts[j] += 1
            rows_used += 1

    has_row = counts > 0
    if cfg.activation_pooling == "mean":
        if cfg.c2s_cell_pooling == "mean":
            Z_sum[has_row] /= counts[has_row, None].astype(np.float32)
        elif cfg.c2s_cell_pooling == "max":
            Z_sum[~has_row] = 0.0
        else:
            raise ValueError("c2s_cell_pooling must be mean or max")
    else:
        # token pooling should have one selected row per cell.  If duplicates exist, average them.
        dup = counts > 1
        if dup.any():
            Z_sum[dup] /= counts[dup, None].astype(np.float32)

    Z_sum[~has_row] = np.nan
    summary = {
        "model": cfg.model,
        "layer": cfg.layer,
        "activation_pooling": cfg.activation_pooling,
        "token_value": cfg.token_value,
        "c2s_cell_pooling": cfg.c2s_cell_pooling,
        "n_obs": int(n_obs),
        "n_has_row": int(has_row.sum()),
        "n_latents": int(n_latents),
        "d_in": int(d_in),
        "n_shards": int(len(shards)),
        "rows_seen": int(rows_seen),
        "rows_used": int(rows_used),
        "rows_unmatched": int(rows_unmatched),
        "token_rows_seen": int(token_rows_seen),
        "counts_min": int(counts[has_row].min()) if has_row.any() else 0,
        "counts_median": float(np.median(counts[has_row])) if has_row.any() else 0.0,
        "counts_max": int(counts[has_row].max()) if has_row.any() else 0,
    }
    return Z_sum, summary


def _prepare_expression_and_gt(cfg: CellHeldoutF1Config) -> Tuple[Any, pd.DataFrame, np.ndarray, List[str], Dict[str, Any]]:
    import scanpy as sc
    import scipy.sparse as sp

    adata = sc.read_h5ad(cfg.adata_path)
    gt = pd.read_csv(cfg.gt_csv, index_col=0)
    gt.index = gt.index.astype(str)
    gt.columns = gt.columns.astype(str)

    gene_to_i = {str(g): i for i, g in enumerate(adata.var_names.astype(str))}
    common_genes = [g for g in gt.index.astype(str) if g in gene_to_i]
    if not common_genes:
        raise RuntimeError("No overlap between GT genes and adata.var_names")
    gt_sub = gt.loc[common_genes].copy()
    gt_sub = gt_sub.apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.int8)

    concept_sizes = gt_sub.sum(axis=0).astype(int)
    keep_cols = concept_sizes[concept_sizes >= int(cfg.min_concept_genes)].index.tolist()
    gt_sub = gt_sub[keep_cols]
    duplicate_map: Dict[str, str] = {}
    if cfg.dedupe_identical_concepts and gt_sub.shape[1] > 0:
        # Exact duplicate concept signatures after panel restriction.
        seen: Dict[bytes, str] = {}
        keep = []
        for col in gt_sub.columns:
            sig = gt_sub[col].to_numpy(dtype=np.int8).tobytes()
            if sig in seen:
                duplicate_map[col] = seen[sig]
            else:
                seen[sig] = col
                keep.append(col)
        gt_sub = gt_sub[keep]
    if cfg.max_concepts is not None and gt_sub.shape[1] > int(cfg.max_concepts):
        # Keep largest panel-supported concepts for quick debugging if requested.
        sizes = gt_sub.sum(axis=0).sort_values(ascending=False)
        gt_sub = gt_sub[sizes.index[: int(cfg.max_concepts)].tolist()]

    gene_idx = [gene_to_i[g] for g in gt_sub.index.astype(str)]
    X = adata.X[:, gene_idx]
    if sp.issparse(X):
        X = X.astype(np.float32).tocsr()
        totals = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
        scale = np.divide(10000.0, totals, out=np.zeros_like(totals), where=totals > 0)
        X = X.multiply(scale[:, None])
        X.data = np.log1p(X.data)
        X = X.toarray().astype(np.float32)
    else:
        X = np.asarray(X, dtype=np.float32)
        totals = X.sum(axis=1).astype(np.float32)
        scale = np.divide(10000.0, totals, out=np.zeros_like(totals), where=totals > 0)
        X = np.log1p(X * scale[:, None]).astype(np.float32)

    meta = {
        "adata_path": cfg.adata_path,
        "gt_csv": cfg.gt_csv,
        "n_adata_genes": int(adata.n_vars),
        "n_gt_genes": int(gt.shape[0]),
        "n_common_genes": int(len(common_genes)),
        "n_concepts_after_min_genes": int(len(keep_cols)),
        "n_concepts_after_dedupe": int(gt_sub.shape[1]),
        "n_duplicate_concepts_removed": int(len(duplicate_map)),
        "duplicate_concepts_removed": duplicate_map,
    }
    return adata, gt_sub, X, list(gt_sub.columns), meta


def _split_cells(n: int, train_frac: float, valid_frac: float, test_frac: float, seed: int) -> Dict[str, np.ndarray]:
    if train_frac <= 0 or valid_frac <= 0 or test_frac <= 0:
        raise ValueError("train/valid/test fractions must be positive")
    total = train_frac + valid_frac + test_frac
    train_frac, valid_frac, test_frac = train_frac / total, valid_frac / total, test_frac / total
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(round(n * train_frac))
    n_valid = int(round(n * valid_frac))
    train = perm[:n_train]
    valid = perm[n_train:n_train + n_valid]
    test = perm[n_train + n_valid:]
    return {"train": train, "valid": valid, "test": test}


def _concept_scores(X: np.ndarray, gt_sub: pd.DataFrame) -> np.ndarray:
    G = gt_sub.to_numpy(dtype=np.float32)  # genes x concepts
    sizes = G.sum(axis=0)
    sizes[sizes <= 0] = np.nan
    scores = (X @ G) / sizes[None, :]
    return scores.astype(np.float32)


def _f1_from_counts(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
    denom = 2 * tp + fp + fn
    return np.divide(2 * tp, denom, out=np.zeros_like(tp, dtype=np.float32), where=denom > 0)


def _evaluate_split(Z: np.ndarray, y: np.ndarray, features: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    rows = []
    for c in range(y.shape[1]):
        feat = int(features[c])
        thr = float(thresholds[c])
        pred = Z[:, feat] > thr
        yy = y[:, c]
        tp = int(np.logical_and(pred, yy).sum())
        fp = int(np.logical_and(pred, ~yy).sum())
        fn = int(np.logical_and(~pred, yy).sum())
        tn = int(np.logical_and(~pred, ~yy).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        rows.append({"concept_idx": c, "feature": feat, "threshold": thr, "f1": f1, "precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn, "tn": tn})
    return pd.DataFrame(rows)


def run_cell_heldout_f1(cfg: CellHeldoutF1Config) -> Dict[str, Any]:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "cell_heldout_config.json", asdict(cfg))

    print("[cell-f1] preparing expression and GT", flush=True)
    adata, gt_sub, X, concepts, meta = _prepare_expression_and_gt(cfg)
    print(f"[cell-f1] prepared X={X.shape} concepts={len(concepts)} gt_sub={gt_sub.shape}", flush=True)
    obs_names = [str(x) for x in adata.obs_names]
    print("[cell-f1] building cell-level SAE matrix", flush=True)
    Z, extract_summary = _build_cell_sae_matrix(cfg, obs_names)
    print(f"[cell-f1] encode done n_has_row={extract_summary.get('n_has_row')} rows_used={extract_summary.get('rows_used')} token_rows_seen={extract_summary.get('token_rows_seen')}", flush=True)

    has_z = np.isfinite(Z).all(axis=1)
    if not has_z.any():
        raise RuntimeError("No cells with SAE latent rows after alignment")

    splits = _split_cells(len(obs_names), cfg.train_frac, cfg.valid_frac, cfg.test_frac, cfg.split_seed)
    _write_json(out_dir / "heldout_cell_split.json", {k: [str(obs_names[i]) for i in v] for k, v in splits.items()})

    print("[cell-f1] computing concept scores", flush=True)
    scores = _concept_scores(X, gt_sub)
    train_idx = splits["train"]
    # Positive threshold is fixed from train cells only, then applied to valid/test.
    concept_thresholds = np.nanquantile(scores[train_idx], float(cfg.concept_score_quantile), axis=0).astype(np.float32)
    Y = scores >= concept_thresholds[None, :]

    # Filter concepts with enough support on valid/test and at least one cell with latent rows.
    valid_idx = splits["valid"]
    test_idx = splits["test"]
    valid_mask_cells = has_z[valid_idx]
    test_mask_cells = has_z[test_idx]
    valid_idx2 = valid_idx[valid_mask_cells]
    test_idx2 = test_idx[test_mask_cells]
    train_idx2 = train_idx[has_z[train_idx]]

    support_rows = []
    keep_concepts = []
    concept_gene_counts = gt_sub.sum(axis=0).astype(int).to_dict()
    for c, concept in enumerate(concepts):
        v_pos = int(Y[valid_idx2, c].sum())
        t_pos = int(Y[test_idx2, c].sum())
        v_neg = int(len(valid_idx2) - v_pos)
        t_neg = int(len(test_idx2) - t_pos)
        tr_pos = int(Y[train_idx2, c].sum())
        keep = (v_pos >= cfg.min_pos_valid and t_pos >= cfg.min_pos_test and v_neg >= cfg.min_neg_valid and t_neg >= cfg.min_neg_test)
        support_rows.append({
            "concept": concept,
            "concept_idx": c,
            "n_genes": int(concept_gene_counts.get(concept, 0)),
            "score_threshold_train_q": float(concept_thresholds[c]),
            "train_pos": tr_pos,
            "train_n": int(len(train_idx2)),
            "valid_pos": v_pos,
            "valid_neg": v_neg,
            "valid_n": int(len(valid_idx2)),
            "test_pos": t_pos,
            "test_neg": t_neg,
            "test_n": int(len(test_idx2)),
            "keep": bool(keep),
        })
        if keep:
            keep_concepts.append(c)

    support_df = pd.DataFrame(support_rows)
    support_df.to_csv(out_dir / "concept_support_by_cell_split.csv", index=False)
    print(f"[cell-f1] support filtering kept {len(keep_concepts)}/{len(concepts)} concepts", flush=True)
    if not keep_concepts:
        raise RuntimeError("No concepts passed cell-heldout support filters")

    keep_concepts = np.asarray(keep_concepts, dtype=np.int64)
    concept_names_kept = [concepts[i] for i in keep_concepts.tolist()]
    Z_valid = Z[valid_idx2].astype(np.float32)
    Z_test = Z[test_idx2].astype(np.float32)
    Y_valid = Y[valid_idx2][:, keep_concepts].astype(bool)
    Y_test = Y[test_idx2][:, keep_concepts].astype(bool)

    n_latents = Z.shape[1]
    best_f1 = np.full(len(keep_concepts), -1.0, dtype=np.float32)
    best_feature = np.zeros(len(keep_concepts), dtype=np.int64)
    best_threshold = np.zeros(len(keep_concepts), dtype=np.float32)

    Yv_u8 = Y_valid.astype(np.uint8)
    y_pos = Yv_u8.sum(axis=0).astype(np.int64)
    concept_chunk_size = max(1, int(cfg.concept_chunk_size))
    counts_dir = out_dir / "counts"
    counts_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[cell-f1] threshold search Z_valid={Z_valid.shape} Y_valid={Y_valid.shape} "
        f"thresholds={list(cfg.latent_thresholds)} concept_chunk_size={concept_chunk_size}",
        flush=True,
    )

    for thr in cfg.latent_thresholds:
        thr_f = float(thr)
        print(f"[cell-f1] threshold={thr_f}", flush=True)

        B = (Z_valid > thr_f).astype(np.uint8)
        pred_pos = B.sum(axis=0).astype(np.int64)  # latent
        # Use int32 to avoid uint8 overflow while keeping memory below int64.
        B_T = B.T.astype(np.int32, copy=False)  # latent x cells

        n_concepts = Yv_u8.shape[1]
        tp_all = np.zeros((n_latents, n_concepts), dtype=np.uint32)

        for c0 in tqdm(
            range(0, n_concepts, concept_chunk_size),
            desc=f"cell-f1 thr={thr_f}",
            unit="chunk",
            dynamic_ncols=True,
        ):
            c1 = min(c0 + concept_chunk_size, n_concepts)
            Y_chunk = Yv_u8[:, c0:c1].astype(np.int32, copy=False)

            # latent x concept_chunk
            tp = B_T @ Y_chunk
            tp_all[:, c0:c1] = tp.astype(np.uint32, copy=False)

            fp = pred_pos[:, None] - tp
            fn = y_pos[None, c0:c1] - tp
            f1 = _f1_from_counts(
                tp.astype(np.float32, copy=False),
                fp.astype(np.float32, copy=False),
                fn.astype(np.float32, copy=False),
            )

            feat_for_chunk = f1.argmax(axis=0)
            f1_for_chunk = f1[feat_for_chunk, np.arange(f1.shape[1])]

            global_cols = np.arange(c0, c1)
            improve = f1_for_chunk > best_f1[global_cols]
            if improve.any():
                best_f1[global_cols[improve]] = f1_for_chunk[improve]
                best_feature[global_cols[improve]] = feat_for_chunk[improve]
                best_threshold[global_cols[improve]] = thr_f

        thr_tag = str(thr_f).replace(".", "p").replace("-", "m")
        np.savez_compressed(
            counts_dir / f"valid_counts_thr_{thr_tag}.npz",
            tp=tp_all,
            pred_pos=pred_pos.astype(np.uint32, copy=False),
            y_pos=y_pos.astype(np.uint32, copy=False),
            threshold=np.asarray([thr_f], dtype=np.float32),
            n_valid_cells=np.asarray([Z_valid.shape[0]], dtype=np.uint32),
        )

    valid_df = _evaluate_split(Z_valid, Y_valid, best_feature, best_threshold)
    test_df = _evaluate_split(Z_test, Y_test, best_feature, best_threshold)
    for df in (valid_df, test_df):
        df["concept"] = [concept_names_kept[i] for i in df["concept_idx"].tolist()]
        df["concept_global_idx"] = keep_concepts[df["concept_idx"].to_numpy(dtype=np.int64)]
        df["n_genes"] = [int(concept_gene_counts.get(c, 0)) for c in df["concept"].tolist()]
        # Put concept first for readability.
        cols = ["concept", "concept_global_idx", "n_genes", "feature", "threshold", "f1", "precision", "recall", "tp", "fp", "fn", "tn", "concept_idx"]
        df = df[cols]

    valid_df = valid_df[["concept", "concept_global_idx", "n_genes", "feature", "threshold", "f1", "precision", "recall", "tp", "fp", "fn", "tn", "concept_idx"]]
    test_df = test_df[["concept", "concept_global_idx", "n_genes", "feature", "threshold", "f1", "precision", "recall", "tp", "fp", "fn", "tn", "concept_idx"]]
    valid_df.to_csv(out_dir / "valid_cell_concept_f1_scores.csv", index=False)
    test_df.to_csv(out_dir / "test_cell_concept_f1_scores.csv", index=False)

    # Selected features manifest.
    selected = valid_df[["concept", "feature", "threshold", "f1", "precision", "recall"]].copy()
    selected = selected.sort_values("f1", ascending=False)
    selected.to_json(out_dir / "selected_features_valid_cell.json", orient="records", indent=2)

    summary = {
        "mode": "cell_heldout_f1",
        "model": cfg.model,
        "layer": cfg.layer,
        "n_obs": int(len(obs_names)),
        "n_cells_with_sae": int(has_z.sum()),
        "n_concepts_input": int(len(concepts)),
        "n_concepts_kept": int(len(keep_concepts)),
        "valid_max_f1": float(valid_df["f1"].max()) if len(valid_df) else 0.0,
        "valid_mean_f1": float(valid_df["f1"].mean()) if len(valid_df) else 0.0,
        "test_max_f1": float(test_df["f1"].max()) if len(test_df) else 0.0,
        "test_mean_f1": float(test_df["f1"].mean()) if len(test_df) else 0.0,
        "test_n_f1_ge_0p3": int((test_df["f1"] >= 0.3).sum()) if len(test_df) else 0,
        "test_n_f1_ge_0p5": int((test_df["f1"] >= 0.5).sum()) if len(test_df) else 0,
        "concept_chunk_size": int(cfg.concept_chunk_size),
        "counts_dir": str(out_dir / "counts"),
        "extract_summary": extract_summary,
        "gt_meta": meta,
    }
    _write_json(out_dir / "counts_summary_cell.json", summary)
    return summary
