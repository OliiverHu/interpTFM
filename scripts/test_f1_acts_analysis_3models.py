#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
import scanpy as sc
from tqdm import tqdm
from scipy import sparse

from interp_pipeline.adapters.model_base import ModelSpec
from interp_pipeline.adapters.models.scgpt import ScGPTAdapter


DEFAULT_THRESHOLDS = [0.0, 0.15, 0.3, 0.6]


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def is_int_str(x: Any) -> bool:
    try:
        return str(x).strip().isdigit()
    except Exception:
        return False


def build_symbol_to_ensembl_from_adata(adata, symbol_col: str = "index") -> Dict[str, str]:
    if symbol_col not in adata.var.columns:
        raise ValueError(f"Expected adata.var['{symbol_col}'] to contain gene symbols.")
    ens = adata.var.index.astype(str).tolist()
    sym = adata.var[symbol_col].astype(str).tolist()
    return {s: e for e, s in zip(ens, sym)}


def build_scgpt_gene_decoder(scgpt_ckpt: str, device: str = "cpu") -> Callable[[int], str]:
    adapter = ScGPTAdapter()
    handle = adapter.load(ModelSpec(name="scgpt", checkpoint=scgpt_ckpt, device=device, options={}))
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

    raise RuntimeError("Cannot build decoder for scGPT tokenizer.")


def token_to_ens(
    tid: Any,
    *,
    decode_token: Optional[Callable[[int], str]],
    sym2ens: Dict[str, str],
    decode_cache: Dict[int, str],
) -> Optional[str]:
    s = str(tid).strip()
    if not s or s in {"<pad>", "<PAD>", "<cls>", "<CLS>", "PAD", "CLS"}:
        return None

    if s.startswith("ENSG"):
        return s

    if is_int_str(s):
        if decode_token is None:
            return None
        ii = int(s)
        sym = decode_cache.get(ii)
        if sym is None:
            sym = decode_token(ii)
            decode_cache[ii] = sym
        if not sym or sym in {"<pad>", "<PAD>", "<cls>", "<CLS>", "PAD", "CLS"}:
            return None
        if sym.startswith("ENSG"):
            return sym
        return sym2ens.get(sym)

    return sym2ens.get(s)


def list_shards(acts_root: str, max_shards: Optional[int], shard_offset: int) -> List[str]:
    shards = sorted([p for p in glob.glob(os.path.join(acts_root, "shard_*")) if os.path.isdir(p)])
    if shard_offset:
        shards = shards[int(shard_offset):]
    if max_shards is not None:
        shards = shards[: int(max_shards)]
    return shards


def dense_bool_to_csr(A_bool: np.ndarray) -> sparse.csr_matrix:
    rows, cols = np.where(A_bool)
    data = np.ones(rows.shape[0], dtype=np.int8)
    return sparse.csr_matrix((data, (rows, cols)), shape=A_bool.shape)


def accumulate_tp_sparse_into_dense(tp_dense: np.ndarray, A_bool: np.ndarray, G_csr: sparse.csr_matrix) -> None:
    A_csr = dense_bool_to_csr(A_bool)
    TP = (A_csr.T @ G_csr).tocoo()
    tp_dense[TP.row, TP.col] += TP.data.astype(tp_dense.dtype, copy=False)


def compute_metrics_from_counts(
    tp: np.ndarray,
    pred_pos: np.ndarray,
    true_pos: np.ndarray,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tp_f = tp.astype(np.float32)
    fp = (pred_pos[:, None] - tp).astype(np.int64)
    fn = (true_pos[None, :] - tp).astype(np.int64)
    fp_f = fp.astype(np.float32)
    fn_f = fn.astype(np.float32)

    precision = tp_f / (tp_f + fp_f + eps)
    recall = tp_f / (tp_f + fn_f + eps)
    f1 = (2 * precision * recall) / (precision + recall + eps)
    return precision, recall, f1, fp, fn


def load_split_genes(split_json: Optional[str], split_name: str) -> Optional[Set[str]]:
    if split_json is None:
        return None
    with open(split_json) as f:
        obj = json.load(f)
    key = f"{split_name}_genes"
    if key not in obj:
        raise KeyError(f"{split_json} missing key {key}. Keys: {list(obj.keys())}")
    return set(map(str, obj[key]))


def map_rows_to_gt_and_split(
    token_ids: Sequence[Any],
    *,
    gt_row_for_ens: Dict[str, int],
    eval_genes: Optional[Set[str]],
    decode_token: Optional[Callable[[int], str]],
    sym2ens: Dict[str, str],
    decode_cache: Dict[int, str],
) -> Tuple[np.ndarray, np.ndarray]:
    gt_rows = np.full((len(token_ids),), -1, dtype=np.int32)
    keep = np.zeros((len(token_ids),), dtype=bool)

    for i, tid in enumerate(token_ids):
        ens = token_to_ens(
            tid,
            decode_token=decode_token,
            sym2ens=sym2ens,
            decode_cache=decode_cache,
        )
        if ens is None:
            continue
        r = gt_row_for_ens.get(ens)
        if r is None:
            continue
        if eval_genes is not None and ens not in eval_genes:
            continue
        gt_rows[i] = r
        keep[i] = True

    return gt_rows, keep


def load_term_meta(term_meta_path: Optional[str]) -> Dict[str, str]:
    if term_meta_path is None or not os.path.exists(term_meta_path):
        return {}
    meta = pd.read_csv(term_meta_path, sep="\t")
    if "term_id" not in meta.columns:
        return {}
    if "term_name" not in meta.columns and "name" in meta.columns:
        meta = meta.rename(columns={"name": "term_name"})
    if "term_name" not in meta.columns:
        return {}
    return meta.drop_duplicates("term_id").set_index("term_id")["term_name"].astype(str).to_dict()


def analyze_one_model(
    *,
    label: str,
    store_root: str,
    layer: str,
    gt_csv: str,
    adata_path: str,
    out_dir: str,
    scgpt_ckpt: Optional[str],
    term_meta_path: Optional[str],
    split_json: Optional[str],
    eval_split: str,
    thresholds: Sequence[float],
    max_shards: Optional[int],
    shard_offset: int,
    use_relu: bool,
    do_zscore: bool,
    zscore_eps: float,
    symbol_col: str,
) -> None:
    out_dir_p = ensure_dir(out_dir)
    acts_root = os.path.join(store_root, "activations", layer)

    print("=" * 100)
    print(f"[activation F1 baseline] {label}")
    print(f"  store_root={store_root}")
    print(f"  layer={layer}")
    print(f"  acts_root={acts_root}")
    print(f"  gt_csv={gt_csv}")
    print(f"  out_dir={out_dir_p}")
    print(f"  eval_split={eval_split}")
    print("=" * 100)

    adata = sc.read_h5ad(adata_path)
    sym2ens = build_symbol_to_ensembl_from_adata(adata, symbol_col=symbol_col)

    decode_token = None
    if label.lower() == "scgpt":
        if not scgpt_ckpt:
            raise ValueError("scGPT activation baseline requires --scgpt-foundation-ckpt")
        decode_token = build_scgpt_gene_decoder(scgpt_ckpt, device="cpu")
    decode_cache: Dict[int, str] = {}

    gt_df = pd.read_csv(gt_csv, index_col=0)
    GT = gt_df.values.astype(np.int8)
    gt_row_for_ens = {g: i for i, g in enumerate(gt_df.index.astype(str).tolist())}
    term_ids = gt_df.columns.astype(str).tolist()
    term_name_map = load_term_meta(term_meta_path)
    term_names = [term_name_map.get(t, t) for t in term_ids]
    T = GT.shape[1]

    eval_genes = None
    if split_json is not None and eval_split != "all":
        eval_genes = load_split_genes(split_json, split_name=eval_split)
        print(f"[split] loaded {len(eval_genes):,} {eval_split} genes from {split_json}")

    shards = list_shards(acts_root, max_shards=max_shards, shard_offset=shard_offset)
    print(f"[shards] n={len(shards)}")
    if not shards:
        raise RuntimeError(f"No shards found under {acts_root}")

    # Pass 1: z-score stats on mapped/eval rows.
    mean = std = None
    H_observed = None
    n_stats = 0
    if do_zscore:
        print("[pass 1] estimate mean/std over mapped eval rows")
        sums = None
        sums2 = None
        for shard in tqdm(shards, desc=f"{label}:meanstd"):
            acts_path = os.path.join(shard, "activations.pt")
            idx_path = os.path.join(shard, "index.pt")
            if not (os.path.exists(acts_path) and os.path.exists(idx_path)):
                continue
            X = torch.load(acts_path, map_location="cpu").float()
            idx = torch.load(idx_path, map_location="cpu")
            if idx.get("token_unit", "unknown") != "gene":
                continue
            token_ids = idx.get("token_ids", None)
            if token_ids is None or len(token_ids) != X.shape[0]:
                continue

            gt_rows, keep = map_rows_to_gt_and_split(
                token_ids,
                gt_row_for_ens=gt_row_for_ens,
                eval_genes=eval_genes,
                decode_token=decode_token,
                sym2ens=sym2ens,
                decode_cache=decode_cache,
            )
            if not np.any(keep):
                continue
            Xu = X[keep]
            if use_relu:
                Xu = torch.relu(Xu)
            if sums is None:
                H_observed = int(Xu.shape[1])
                sums = torch.zeros(H_observed, dtype=torch.float64)
                sums2 = torch.zeros(H_observed, dtype=torch.float64)
            sums += Xu.double().sum(dim=0)
            sums2 += (Xu.double() * Xu.double()).sum(dim=0)
            n_stats += int(Xu.shape[0])

        if sums is None or n_stats == 0:
            raise RuntimeError("No usable mapped rows for z-score stats.")
        mean = (sums / max(n_stats, 1)).numpy()
        var = (sums2 / max(n_stats, 1)).numpy() - mean * mean
        std = np.sqrt(np.clip(var, zscore_eps, None))
        print(f"[zscore] n_stats={n_stats:,}; H={H_observed}")

    print("[pass 2] accumulate activation-dim F1 counts")
    tp: Dict[float, Optional[np.ndarray]] = {float(thr): None for thr in thresholds}
    pred_pos: Dict[float, Optional[np.ndarray]] = {float(thr): None for thr in thresholds}
    true_pos = np.zeros((T,), dtype=np.int64)

    rows_total = 0
    rows_kept = 0

    for shard in tqdm(shards, desc=f"{label}:counts"):
        acts_path = os.path.join(shard, "activations.pt")
        idx_path = os.path.join(shard, "index.pt")
        if not (os.path.exists(acts_path) and os.path.exists(idx_path)):
            continue
        X = torch.load(acts_path, map_location="cpu").float()
        idx = torch.load(idx_path, map_location="cpu")
        if idx.get("token_unit", "unknown") != "gene":
            continue
        token_ids = idx.get("token_ids", None)
        if token_ids is None or len(token_ids) != X.shape[0]:
            continue

        rows_total += int(X.shape[0])
        gt_rows, keep = map_rows_to_gt_and_split(
            token_ids,
            gt_row_for_ens=gt_row_for_ens,
            eval_genes=eval_genes,
            decode_token=decode_token,
            sym2ens=sym2ens,
            decode_cache=decode_cache,
        )
        if not np.any(keep):
            continue

        Xu = X[keep].numpy()
        if use_relu:
            Xu = np.maximum(Xu, 0.0)
        if do_zscore and mean is not None and std is not None:
            Xu = (Xu - mean[None, :]) / std[None, :]

        rows_kept += int(Xu.shape[0])
        G = GT[gt_rows[keep]]
        G_csr = sparse.csr_matrix(G)
        true_pos += np.asarray(G_csr.sum(axis=0)).ravel().astype(np.int64)

        if H_observed is None:
            H_observed = int(Xu.shape[1])
        for thr in thresholds:
            thr = float(thr)
            if tp[thr] is None:
                tp[thr] = np.zeros((H_observed, T), dtype=np.int32)
                pred_pos[thr] = np.zeros((H_observed,), dtype=np.int64)

            A_bool = Xu > thr
            pred_pos[thr] += A_bool.sum(axis=0).astype(np.int64)  # type: ignore[operator]
            accumulate_tp_sparse_into_dense(tp[thr], A_bool, G_csr)  # type: ignore[arg-type]

    if H_observed is None:
        raise RuntimeError("No usable activation rows.")

    print("[write] activation_per_feature_best.csv and activation_f1_long.csv")
    best_rows = []
    long_rows = []

    for thr in thresholds:
        thr = float(thr)
        precision, recall, f1, fp, fn = compute_metrics_from_counts(
            tp=tp[thr],  # type: ignore[arg-type]
            pred_pos=pred_pos[thr],  # type: ignore[arg-type]
            true_pos=true_pos,
        )
        best_term_idx = np.argmax(f1, axis=1)
        best_f1 = f1[np.arange(H_observed), best_term_idx]

        for h in range(H_observed):
            j = int(best_term_idx[h])
            best_rows.append(
                {
                    "layer": layer,
                    "threshold": float(thr),
                    "feature": int(h),
                    "best_term_id": term_ids[j],
                    "best_term_name": term_names[j],
                    "best_f1": float(best_f1[h]),
                    "tp": int(tp[thr][h, j]),  # type: ignore[index]
                    "pred_pos": int(pred_pos[thr][h]),  # type: ignore[index]
                    "true_pos": int(true_pos[j]),
                }
            )

        # Long output can be large, but useful for debugging.
        for h in range(H_observed):
            for j, term_id in enumerate(term_ids):
                long_rows.append(
                    {
                        "layer": layer,
                        "threshold": float(thr),
                        "feature": int(h),
                        "concept": str(term_id),
                        "term_id": str(term_id),
                        "term_name": term_names[j],
                        "f1": float(f1[h, j]),
                        "precision": float(precision[h, j]),
                        "recall": float(recall[h, j]),
                        "tp": int(tp[thr][h, j]),  # type: ignore[index]
                        "fp": int(fp[h, j]),
                        "fn": int(fn[h, j]),
                        "pred_pos": int(pred_pos[thr][h]),  # type: ignore[index]
                        "true_pos": int(true_pos[j]),
                    }
                )

    best_df = pd.DataFrame(best_rows)
    best_df.to_csv(out_dir_p / "activation_per_feature_best.csv", index=False)

    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(out_dir_p / "activation_f1_long.csv", index=False)

    with open(out_dir_p / "activation_baseline_summary.json", "w") as f:
        json.dump(
            {
                "label": label,
                "store_root": store_root,
                "layer": layer,
                "acts_root": acts_root,
                "gt_csv": gt_csv,
                "adata_path": adata_path,
                "split_json": split_json,
                "eval_split": eval_split,
                "thresholds": [float(x) for x in thresholds],
                "use_relu": bool(use_relu),
                "do_zscore": bool(do_zscore),
                "max_shards": max_shards,
                "shard_offset": int(shard_offset),
                "rows_total": int(rows_total),
                "rows_kept": int(rows_kept),
                "n_stats": int(n_stats),
                "n_features": int(H_observed),
                "n_terms": int(T),
            },
            f,
            indent=2,
        )

    print(f"[OK] wrote {out_dir_p / 'activation_per_feature_best.csv'}")
    print(f"     rows_total={rows_total:,} rows_kept={rows_kept:,} H={H_observed} T={T}")


def none_if_str_none(x: str | None) -> str | None:
    if x is None:
        return None
    if str(x).lower() in {"none", "null", "na", ""}:
        return None
    return x


def main() -> None:
    ap = argparse.ArgumentParser(description="3-model raw activation/model-embedding F1 baseline.")
    ap.add_argument("--labels", nargs=3, required=True)
    ap.add_argument("--store-roots", nargs=3, required=True)
    ap.add_argument("--layers", nargs=3, required=True)
    ap.add_argument("--out-dirs", nargs=3, required=True)

    ap.add_argument("--gt-csv", required=True)
    ap.add_argument("--adata-path", required=True)
    ap.add_argument("--term-meta-path", default=None)
    ap.add_argument("--split-jsons", nargs=3, default=[None, None, None])
    ap.add_argument("--eval-split", choices=["all", "train", "valid", "test"], default="test")
    ap.add_argument("--scgpt-foundation-ckpt", default=None)
    ap.add_argument("--adata-symbol-col", default="index")

    ap.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS)
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--shard-offset", type=int, default=0)
    ap.add_argument("--use-relu", dest="use_relu", action="store_true", default=True)
    ap.add_argument("--no-use-relu", dest="use_relu", action="store_false")
    ap.add_argument("--do-zscore", dest="do_zscore", action="store_true", default=True)
    ap.add_argument("--no-do-zscore", dest="do_zscore", action="store_false")
    ap.add_argument("--zscore-eps", type=float, default=1e-12)

    args = ap.parse_args()

    split_jsons = [none_if_str_none(x) for x in args.split_jsons]

    for label, store_root, layer, out_dir, split_json in zip(
        args.labels,
        args.store_roots,
        args.layers,
        args.out_dirs,
        split_jsons,
    ):
        analyze_one_model(
            label=label,
            store_root=store_root,
            layer=layer,
            gt_csv=args.gt_csv,
            adata_path=args.adata_path,
            out_dir=out_dir,
            scgpt_ckpt=args.scgpt_foundation_ckpt,
            term_meta_path=args.term_meta_path,
            split_json=split_json,
            eval_split=args.eval_split,
            thresholds=args.thresholds,
            max_shards=args.max_shards,
            shard_offset=args.shard_offset,
            use_relu=args.use_relu,
            do_zscore=args.do_zscore,
            zscore_eps=args.zscore_eps,
            symbol_col=args.adata_symbol_col,
        )


if __name__ == "__main__":
    main()


# python test_f1_acts_analysis_3models.py \
#   --labels scgpt c2sscale geneformer \
#   --store-roots \
#     /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/full_scgpt_cosmx \
#     /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/full_c2sscale_cosmx \
#     /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/full_geneformer_cosmx \
#   --layers layer_4.norm2 layer_17 layer_4 \
#   --out-dirs \
#     /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/full_scgpt_cosmx/activation_f1_baseline/layer_4.norm2 \
#     /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/full_c2sscale_cosmx/activation_f1_baseline/layer_17 \
#     /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/full_geneformer_cosmx/activation_f1_baseline/layer_4 \
#   --gt-csv /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/heldout_3models_l1_3e-3_geneheldout/gprofiler/gprofiler_binary_gene_by_term.csv \
#   --adata-path /maiziezhou_lab2/yunfei/Projects/FM_temp/interGFM/ge_shards/cosmx_human_lung_sec8.h5ad \
#   --term-meta-path /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/heldout_3models_l1_3e-3_geneheldout/gprofiler/gprofiler_terms.tsv \
#   --split-jsons \
#     /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/heldout_3models_l1_3e-3_geneheldout/scgpt/layer_4.norm2/heldout_gene_split.json \
#     /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/heldout_3models_l1_3e-3_geneheldout/c2sscale/layer_17/heldout_gene_split.json \
#     /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/heldout_3models_l1_3e-3_geneheldout/geneformer/layer_4/heldout_gene_split.json \
#   --eval-split test \
#   --thresholds 0.0 0.15 0.3 0.6 \
#   --scgpt-foundation-ckpt /maiziezhou_lab/zihang/SpatialFoundationModel/eval/scGPT/whole-human-pretrain