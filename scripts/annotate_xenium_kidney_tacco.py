#!/usr/bin/env python3
"""
Annotate Xenium kidney cells with TACCO using GSE183273 reference labels.

Final behavior:
  - Reference: GSE183273 / ref2
  - Label key: obs['subclass.l1']
  - TACCO runs on shared genes between reference and query only
  - Final written full-panel AnnData keeps ALL original Xenium genes
  - No low-confidence cells are removed; tacco_celltype is always top1
  - Confidence scores/flags are still written for QC
  - A second model-rerun AnnData drops ONLY the precomputed model-union genes:
      CCDC39, CD45RA, CD45RO, CLECL1, MTRNR2L11, TRAC

Expected outputs:
  1. xenium_prepared_annotated_ref2_subclass_l1.h5ad        # 465228 x 405
  2. xenium_prepared_annotated_ref2_subclass_l1_union_filtered.h5ad  # 465228 x 399
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import tacco as tc


# ---------------------------------------------------------------------
# Defaults for this Xenium run
# ---------------------------------------------------------------------
DEFAULT_QUERY_PATH = "/maiziezhou_lab2/yunfei/datasets/xenium/xenium_prepared.h5ad"
DEFAULT_REF_PATH = "/maiziezhou_lab2/yunfei/Projects/kidney_omics/sc_ref2/GSE183273/GSE183273_kidney.h5ad"
DEFAULT_GTF_PATH = "/maiziezhou_lab2/yunfei/Projects/kidney_omics/ensembl/Homo_sapiens.GRCh38.115.gtf.gz"
DEFAULT_RUNS_ROOT = "/maiziezhou_lab4/yunfei/Projects/interpTFM/tacco_output_runs"
DEFAULT_FULL_OUT = "/maiziezhou_lab2/yunfei/datasets/xenium/xenium_prepared_annotated_ref2_subclass_l1.h5ad"
DEFAULT_MODEL_OUT = "/maiziezhou_lab2/yunfei/datasets/xenium/xenium_prepared_annotated_ref2_subclass_l1_union_filtered.h5ad"

MODEL_UNION_DROP_GENES = {"CCDC39", "CD45RA", "CD45RO", "CLECL1", "MTRNR2L11", "TRAC"}

METHOD = "OT"
PLATFORM_ITERATIONS = 1
NORMALIZE_TO = "adata"
MAX_ANNOTATION = None
N_HVG = None

STRICT_COUNT_CHECK = True
COUNT_INTEGER_ATOL = 1e-6

MIN_TOP_SCORE = 0.25
MIN_MARGIN = 0.05

# Ref2 labels are coarse kidney classes. Merge selected TAL subclasses if present.
REFERENCE_LABEL_RENAME_MAP = {
    "TAL1": "TAL",
    "TAL2": "TAL",
}


# ---------------------------------------------------------------------
# Logging and basic helpers
# ---------------------------------------------------------------------
def log(msg: str) -> None:
    print(f"[TACCO-XENIUM-L1] {msg}", flush=True)


def sanitize_var_names(adata: ad.AnnData) -> ad.AnnData:
    adata.var_names = pd.Index(adata.var_names.astype(str))
    adata.var_names_make_unique()
    return adata


def ensure_unique_obs_names(adata: ad.AnnData, prefix: Optional[str] = None) -> ad.AnnData:
    if prefix is None:
        adata.obs_names = pd.Index(adata.obs_names.astype(str))
    else:
        adata.obs_names = pd.Index([f"{prefix}::{x}" for x in adata.obs_names.astype(str)])
    adata.obs_names_make_unique()
    return adata


def move_counts_to_X(adata: ad.AnnData, counts_source: str) -> ad.AnnData:
    if counts_source == "X":
        return adata
    if counts_source == "raw":
        if adata.raw is None:
            raise ValueError("counts_source='raw' but adata.raw is None")
        tmp = adata.raw.to_adata()
        tmp.obs = adata.obs.copy()
        for k in adata.obsm.keys():
            tmp.obsm[k] = adata.obsm[k]
        for k in adata.uns.keys():
            tmp.uns[k] = adata.uns[k]
        return tmp
    if counts_source.startswith("layer:"):
        layer_name = counts_source.split("layer:", 1)[1]
        if layer_name not in adata.layers:
            raise ValueError(f"Requested layer {layer_name!r} not found")
        adata = adata.copy()
        adata.X = adata.layers[layer_name].copy()
        return adata
    raise ValueError(f"Unsupported counts_source: {counts_source}")


def _finite_matrix_values(X):
    vals = X.data if sp.issparse(X) else np.asarray(X).ravel()
    vals = np.asarray(vals)
    return vals[np.isfinite(vals)]


def assert_no_nan_in_X(adata: ad.AnnData, name: str) -> None:
    X = adata.X
    has_nan = np.isnan(X.data).any() if sp.issparse(X) else np.isnan(np.asarray(X)).any()
    if has_nan:
        raise ValueError(f"{name}: X contains NaN")


def inspect_matrix_kind(adata: ad.AnnData, name: str) -> dict:
    X = adata.X
    vals = _finite_matrix_values(X)
    if vals.size == 0:
        kind = "empty"
        min_val = max_val = None
    else:
        has_negative = bool(np.any(vals < 0))
        is_integer_like = bool(np.allclose(vals, np.round(vals), rtol=0, atol=COUNT_INTEGER_ATOL))
        min_val = float(np.min(vals))
        max_val = float(np.max(vals))
        if has_negative:
            kind = "contains_negative_values"
        elif is_integer_like:
            kind = "integer_like"
        else:
            kind = "non_integer_like"
    return {
        "name": name,
        "shape": [int(adata.n_obs), int(adata.n_vars)],
        "sparse": bool(sp.issparse(X)),
        "matrix_kind": kind,
        "min_value": min_val,
        "max_value": max_val,
    }


def assert_count_like(adata: ad.AnnData, name: str, strict: bool = STRICT_COUNT_CHECK) -> None:
    if not strict:
        return
    assert_no_nan_in_X(adata, name)
    vals = _finite_matrix_values(adata.X)
    if vals.size == 0:
        raise ValueError(f"{name}: X/source is empty")
    if np.any(vals < 0):
        raise ValueError(f"{name}: X/source contains negative values")
    if not np.allclose(vals, np.round(vals), rtol=0, atol=COUNT_INTEGER_ATOL):
        raise ValueError(
            f"{name}: X/source is non-integer-like; TACCO OT should receive raw counts. "
            f"First finite values: {vals[:10]}"
        )


def drop_empty_cells_and_genes(adata: ad.AnnData) -> ad.AnnData:
    X = adata.X
    if sp.issparse(X):
        cell_sums = np.asarray(X.sum(axis=1)).ravel()
        gene_sums = np.asarray(X.sum(axis=0)).ravel()
    else:
        cell_sums = np.asarray(X.sum(axis=1)).ravel()
        gene_sums = np.asarray(X.sum(axis=0)).ravel()
    keep_cells = cell_sums > 0
    keep_genes = gene_sums > 0
    if keep_cells.sum() == 0:
        raise ValueError("All cells have zero counts")
    if keep_genes.sum() == 0:
        raise ValueError("All genes have zero counts")
    return adata[keep_cells, keep_genes].copy()


def detect_gene_id_mode(var_names: Iterable[str], auto_sample_size: int = 1000) -> str:
    vals = pd.Index(pd.Index(var_names).astype(str))
    if len(vals) == 0:
        return "symbol"
    sample = vals[: min(len(vals), auto_sample_size)]
    n_ensembl_like = sum(bool(re.search(r"ENSG\d{11,}", x)) for x in sample)
    frac = n_ensembl_like / max(len(sample), 1)
    return "ensembl_to_symbol" if frac >= 0.5 else "symbol"


def strip_ensembl_version(x: str) -> str:
    return str(x).split(".")[0]


def normalize_possible_ensembl_id(x: str) -> str:
    s = str(x).strip()
    m = re.search(r"(ENSG\d{11,}(?:\.\d+)?)", s)
    if m is not None:
        return m.group(1).split(".")[0]
    return s.split(".")[0]


def parse_gtf_attributes(attr_string: str) -> dict:
    attrs = {}
    for m in re.finditer(r'(\S+)\s+"([^"]*)"', attr_string):
        attrs[m.group(1)] = m.group(2)
    return attrs


def build_ensembl_to_symbol_map_from_gtf(gtf_path: str) -> Dict[str, str]:
    log(f"Building ENSG->symbol map from GTF: {gtf_path}")
    opener = gzip.open if str(gtf_path).endswith(".gz") else open
    rows = []
    with opener(gtf_path, "rt") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "gene":
                continue
            attrs = parse_gtf_attributes(parts[8])
            gene_id = attrs.get("gene_id")
            gene_name = attrs.get("gene_name")
            if gene_id and gene_name:
                rows.append((strip_ensembl_version(gene_id), gene_name))
    if not rows:
        raise ValueError("No gene_id/gene_name pairs found in GTF")
    df = pd.DataFrame(rows, columns=["ensembl_gene_id", "gene_symbol"]).drop_duplicates()
    df = df.drop_duplicates(subset="ensembl_gene_id", keep="first")
    mapping = dict(zip(df["ensembl_gene_id"], df["gene_symbol"]))
    log(f"Built mapping for {len(mapping)} Ensembl gene IDs")
    return mapping


def collapse_duplicate_var_names_sum(adata: ad.AnnData) -> ad.AnnData:
    var_names = pd.Index(adata.var_names.astype(str))
    if var_names.is_unique:
        return adata
    log("Collapsing duplicated gene symbols by summing counts")
    X = adata.X.tocsr() if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
    codes, uniques = pd.factorize(var_names, sort=False)
    group_mat = sp.csr_matrix(
        (np.ones(len(codes), dtype=np.float32), (np.arange(len(codes)), codes)),
        shape=(len(codes), len(uniques)),
    )
    X_new = X @ group_mat
    out = ad.AnnData(
        X=X_new,
        obs=adata.obs.copy(),
        var=pd.DataFrame(index=pd.Index(uniques.astype(str))),
    )
    for k in adata.obsm.keys():
        out.obsm[k] = adata.obsm[k]
    for k in adata.uns.keys():
        out.uns[k] = adata.uns[k]
    out.var_names_make_unique()
    return out


def convert_ref_ensembl_to_symbols(
    adata: ad.AnnData,
    ensembl_to_symbol: Dict[str, str],
    object_name: str = "reference",
) -> ad.AnnData:
    adata = adata.copy()
    original_ids = pd.Index(adata.var_names.astype(str))
    normalized_ids = pd.Index([normalize_possible_ensembl_id(x) for x in original_ids])
    mapped_symbols = pd.Index([ensembl_to_symbol.get(x, np.nan) for x in normalized_ids])
    keep = pd.notna(mapped_symbols)
    log(f"{object_name}: mapped {int(keep.sum())}/{len(original_ids)} features from Ensembl IDs to symbols")
    if int(keep.sum()) == 0:
        log(f"{object_name}: first 20 feature ids = {list(original_ids[:20])}")
        raise ValueError(f"{object_name}: 0 features mapped from Ensembl IDs")
    adata = adata[:, keep].copy()
    mapped_symbols = pd.Index(mapped_symbols[keep].astype(str))
    original_ids = pd.Index(original_ids[keep].astype(str))
    normalized_ids = pd.Index(normalized_ids[keep].astype(str))
    bad = mapped_symbols.isin(["", "nan", "None"])
    if bad.sum() > 0:
        adata = adata[:, ~bad].copy()
        mapped_symbols = mapped_symbols[~bad]
        original_ids = original_ids[~bad]
        normalized_ids = normalized_ids[~bad]
    adata.var["original_feature_id"] = original_ids
    adata.var["normalized_ensembl_id"] = normalized_ids
    adata.var["gene_symbol"] = mapped_symbols
    adata.var_names = mapped_symbols
    adata = collapse_duplicate_var_names_sum(adata)
    log(f"{object_name}: features after symbol collapse = {adata.n_vars}")
    return adata


# ---------------------------------------------------------------------
# Reading reference/query
# ---------------------------------------------------------------------
def read_reference(
    ref_path: str,
    label_key: str,
    counts_source: str,
    gene_id_mode: str,
    ensembl_to_symbol: Dict[str, str],
) -> ad.AnnData:
    ref_path = str(Path(ref_path).resolve())
    ref_name = Path(ref_path).stem
    log(f"Reading reference: {ref_path}")
    ref = sc.read_h5ad(ref_path)
    ref = move_counts_to_X(ref, counts_source)
    ref = sanitize_var_names(ref)
    ref = ensure_unique_obs_names(ref, prefix=ref_name)
    assert_no_nan_in_X(ref, ref_name)

    resolved_gene_id_mode = gene_id_mode
    if gene_id_mode == "auto":
        resolved_gene_id_mode = detect_gene_id_mode(ref.var_names)

    log(
        f"{ref_name}: gene_id_mode requested={gene_id_mode}, resolved={resolved_gene_id_mode}, "
        f"first_10_features={list(ref.var_names[:10])}"
    )

    if resolved_gene_id_mode == "ensembl_to_symbol":
        ref = convert_ref_ensembl_to_symbols(ref, ensembl_to_symbol, object_name=ref_name)
    elif resolved_gene_id_mode == "symbol":
        ref = collapse_duplicate_var_names_sum(ref)
    else:
        raise ValueError(f"Unsupported gene_id_mode: {gene_id_mode}")

    ref = drop_empty_cells_and_genes(ref)
    assert_count_like(ref, ref_name)

    if label_key not in ref.obs.columns:
        raise KeyError(f"Reference missing obs[{label_key!r}]. Available columns: {list(ref.obs.columns)}")

    ref = ref[~ref.obs[label_key].isna()].copy()
    ref.obs[label_key] = ref.obs[label_key].astype(str)

    if REFERENCE_LABEL_RENAME_MAP:
        before = ref.obs[label_key].value_counts().to_dict()
        ref.obs[label_key] = ref.obs[label_key].replace(REFERENCE_LABEL_RENAME_MAP).astype(str)
        n_renamed = int(sum(before.get(k, 0) for k in REFERENCE_LABEL_RENAME_MAP))
        if n_renamed:
            log(f"{ref_name}: renamed {n_renamed} labels using {REFERENCE_LABEL_RENAME_MAP}")

    ref.obs["reference_file"] = ref_name
    ref.obs["reference_mode"] = "GSE183273"
    ref.uns["reference_config"] = {
        "mode_name": "GSE183273",
        "label_key": label_key,
        "counts_source": counts_source,
        "gene_id_mode_requested": gene_id_mode,
        "gene_id_mode_resolved": resolved_gene_id_mode,
        "ref_path": ref_path,
    }
    log(f"Reference loaded: {ref.n_obs} cells x {ref.n_vars} genes; labels={ref.obs[label_key].nunique()}")
    print(ref.obs[label_key].value_counts().head(30), flush=True)
    return ref


def read_query(query_path: str, counts_source: str) -> ad.AnnData:
    query_path = str(Path(query_path).resolve())
    log(f"Reading query: {query_path}")
    q = sc.read_h5ad(query_path)
    q = move_counts_to_X(q, counts_source)
    q = sanitize_var_names(q)
    q = ensure_unique_obs_names(q, prefix=None)
    assert_no_nan_in_X(q, "xenium_query")
    q = drop_empty_cells_and_genes(q)
    assert_count_like(q, "xenium_query")
    guess = detect_gene_id_mode(q.var_names)
    log(f"Query gene_id_mode guess={guess}; first_10_genes={list(q.var_names[:10])}")
    if guess == "ensembl_to_symbol":
        warnings.warn("Query appears to use Ensembl IDs. This script expects Xenium query symbols.")
    q.uns["query_gene_id_mode_guess"] = guess
    return q


# ---------------------------------------------------------------------
# TACCO and scoring
# ---------------------------------------------------------------------
def subset_to_shared_genes(ref: ad.AnnData, query: ad.AnnData) -> Tuple[ad.AnnData, ad.AnnData, pd.Index]:
    shared = ref.var_names.intersection(query.var_names)
    if len(shared) == 0:
        raise ValueError("No overlapping genes between reference and query")
    ref_sub = ref[:, shared].copy()
    query_sub = query[:, shared].copy()
    assert_count_like(ref_sub, "reference_shared")
    assert_count_like(query_sub, "query_shared")
    log(f"Final shared genes used internally for TACCO: {len(shared)}")
    return ref_sub, query_sub, shared


def annotate_with_tacco(query_sub: ad.AnnData, ref_sub: ad.AnnData, label_key: str) -> None:
    tc.tl.annotate(
        query_sub,
        reference=ref_sub,
        annotation_key=label_key,
        result_key="tacco_celltype_scores",
        method=METHOD,
        platform_iterations=PLATFORM_ITERATIONS,
        normalize_to=NORMALIZE_TO,
        max_annotation=MAX_ANNOTATION,
        n_hvg=N_HVG,
        remove_constant_genes=True,
        remove_zero_cells=True,
        assume_valid_counts=False,
        verbose=1,
    )


def add_top_label_no_removal(
    adata: ad.AnnData,
    score_key: str = "tacco_celltype_scores",
    label_out: str = "tacco_celltype",
    min_score: float = MIN_TOP_SCORE,
    min_margin: float = MIN_MARGIN,
) -> dict:
    scores = adata.obsm[score_key]
    if not isinstance(scores, pd.DataFrame):
        scores = pd.DataFrame(scores, index=adata.obs_names)
    scores = scores.copy()
    scores.index = adata.obs_names
    scores = scores.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    adata.obsm[score_key] = scores

    top1_label = scores.idxmax(axis=1).astype(str)
    top1_score = scores.max(axis=1).astype(float)
    arr = scores.to_numpy()
    cols = np.asarray(scores.columns.astype(str))

    if arr.shape[1] > 1:
        order = np.argsort(arr, axis=1)
        top2_idx = order[:, -2]
        top2_label = pd.Series(cols[top2_idx], index=adata.obs_names, dtype="object")
        top2_score = arr[np.arange(arr.shape[0]), top2_idx].astype(float)
    else:
        top2_label = pd.Series(pd.NA, index=adata.obs_names, dtype="object")
        top2_score = np.zeros(arr.shape[0], dtype=float)

    margin = top1_score.to_numpy() - top2_score
    eps = 1e-12
    score_sum = arr.sum(axis=1)
    p = arr / np.maximum(score_sum[:, None], eps)
    entropy = -(p * np.log2(np.maximum(p, eps))).sum(axis=1)

    low_score = top1_score < min_score
    low_margin = margin < min_margin
    low_conf = low_score | low_margin

    adata.obs[f"{label_out}_top1"] = top1_label
    adata.obs[f"{label_out}_top1_score"] = top1_score
    adata.obs[f"{label_out}_top2"] = top2_label
    adata.obs[f"{label_out}_top2_score"] = top2_score
    adata.obs[f"{label_out}_score_margin"] = margin
    adata.obs[f"{label_out}_score_entropy"] = entropy
    adata.obs[f"{label_out}_n_labels_ge_0_05"] = (arr >= 0.05).sum(axis=1)
    adata.obs[f"{label_out}_n_labels_ge_0_10"] = (arr >= 0.10).sum(axis=1)

    # Backward-compatible aliases. No removal: tacco_celltype remains top1 for every cell.
    adata.obs[label_out] = top1_label
    adata.obs[f"{label_out}_score"] = top1_score
    adata.obs[f"{label_out}_score_top2"] = top2_score

    # QC flags only. These do NOT remove/NA any labels.
    adata.obs[f"{label_out}_low_score_flag"] = low_score.astype(bool)
    adata.obs[f"{label_out}_low_margin_flag"] = low_margin.astype(bool)
    adata.obs[f"{label_out}_low_confidence_flag"] = low_conf.astype(bool)
    adata.obs[f"{label_out}_removed_by_confidence"] = False

    summary = {
        "n_total_cells": int(adata.n_obs),
        "n_low_confidence_cells_flagged_not_removed": int(low_conf.sum()),
        "frac_low_confidence_cells_flagged_not_removed": float(low_conf.mean()) if adata.n_obs else 0.0,
        "n_removed_cells": 0,
        "frac_removed_cells": 0.0,
        "min_top_score": float(min_score),
        "min_margin": float(min_margin),
        "n_labels": int(top1_label.nunique()),
        "median_top1_score": float(top1_score.median()),
        "median_margin": float(np.median(margin)),
        "median_entropy": float(np.median(entropy)),
    }
    return summary


def prediction_columns(label_out: str = "tacco_celltype") -> list:
    return [
        label_out,
        f"{label_out}_top1",
        f"{label_out}_top1_score",
        f"{label_out}_top2",
        f"{label_out}_top2_score",
        f"{label_out}_score",
        f"{label_out}_score_top2",
        f"{label_out}_score_margin",
        f"{label_out}_score_entropy",
        f"{label_out}_n_labels_ge_0_05",
        f"{label_out}_n_labels_ge_0_10",
        f"{label_out}_low_score_flag",
        f"{label_out}_low_margin_flag",
        f"{label_out}_low_confidence_flag",
        f"{label_out}_removed_by_confidence",
    ]


# ---------------------------------------------------------------------
# H5AD sanitization
# ---------------------------------------------------------------------
def sanitize_h5_key(x: str) -> str:
    return str(x).replace("/", "_")


def _make_unique_index(values: Iterable[str]) -> pd.Index:
    idx = pd.Index(values)
    if not idx.has_duplicates:
        return idx
    seen = {}
    fixed = []
    for x in idx:
        n = seen.get(x, 0)
        fixed.append(x if n == 0 else f"{x}__{n}")
        seen[x] = n + 1
    return pd.Index(fixed)


def sanitize_score_obsm_for_h5ad(
    adata: ad.AnnData,
    score_key: str = "tacco_celltype_scores",
    mapping_uns_key: str = "tacco_celltype_score_label_mapping",
) -> ad.AnnData:
    mappings = {}
    for key in list(adata.obsm.keys()):
        if key != score_key and "score" not in key.lower():
            continue
        x = adata.obsm[key]
        if isinstance(x, pd.DataFrame):
            df = x.copy()
            df.index = adata.obs_names
            original_cols = pd.Index(df.columns.astype(str))
            sanitized_cols = pd.Index([sanitize_h5_key(c) for c in original_cols])
            sanitized_cols = _make_unique_index(sanitized_cols)
            df.columns = sanitized_cols
            df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
            adata.obsm[key] = df.astype(np.float32)
            mappings[key] = pd.DataFrame({"original_label": original_cols, "sanitized_label": sanitized_cols})
        else:
            arr = np.asarray(x)
            if arr.dtype == object:
                arr = pd.DataFrame(arr).apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
            adata.obsm[key] = arr.astype(np.float32)
    if mappings:
        adata.uns[mapping_uns_key] = mappings.get(score_key, mappings)
    return adata


def sanitize_anndata_for_h5ad(adata: ad.AnnData) -> ad.AnnData:
    slot_mappings = {}
    for slot_name in ["obsm", "varm"]:
        slot = getattr(adata, slot_name)
        for key in list(slot.keys()):
            value = slot[key]
            if isinstance(value, pd.DataFrame):
                original_cols = pd.Index(value.columns.astype(str))
                sanitized_cols = pd.Index([sanitize_h5_key(c) for c in original_cols])
                sanitized_cols = _make_unique_index(sanitized_cols)
                if not original_cols.equals(sanitized_cols):
                    df = value.copy()
                    df.columns = sanitized_cols
                    slot[key] = df
                    slot_mappings[f"{slot_name}::{key}"] = pd.DataFrame(
                        {"original_label": original_cols, "sanitized_label": sanitized_cols}
                    )
    if slot_mappings:
        adata.uns["h5ad_sanitized_label_mappings"] = slot_mappings
    return adata


# ---------------------------------------------------------------------
# Full-panel writing
# ---------------------------------------------------------------------
def copy_annotations_back_to_full_panel(query_full: ad.AnnData, query_sub: ad.AnnData) -> ad.AnnData:
    if not query_full.obs_names.equals(query_sub.obs_names):
        raise ValueError("query_full and query_sub obs_names are not aligned; cannot copy TACCO labels safely")

    out = query_full.copy()
    cols = prediction_columns()
    for c in cols:
        out.obs[c] = query_sub.obs[c].reindex(out.obs_names).values

    # Standard aliases for interpTFM / downstream convenience.
    out.obs["cell_type"] = out.obs["tacco_celltype"].astype(str)
    out.obs["cell_type_top1"] = out.obs["tacco_celltype_top1"].astype(str)
    out.obs["cell_type_source"] = "TACCO_GSE183273_subclass.l1"

    scores = query_sub.obsm["tacco_celltype_scores"]
    if isinstance(scores, pd.DataFrame):
        scores = scores.reindex(out.obs_names)
    out.obsm["tacco_celltype_scores"] = scores

    out.uns["tacco_run"] = query_sub.uns.get("tacco_run", {}).copy()
    out.uns["confidence_summary_no_removal"] = query_sub.uns.get("confidence_summary_no_removal", {}).copy()
    return out


def write_model_union_filtered(full_annotated: ad.AnnData, output_path: str, run_dir: Path) -> ad.AnnData:
    if "feature_name" in full_annotated.var.columns:
        names = full_annotated.var["feature_name"].astype(str)
    else:
        names = pd.Series(full_annotated.var_names.astype(str), index=full_annotated.var_names)

    keep = ~names.isin(MODEL_UNION_DROP_GENES).to_numpy()
    removed = full_annotated.var.loc[~keep].copy()
    removed_csv = run_dir / "model_union_removed_genes.csv"
    removed.to_csv(removed_csv)

    filtered = full_annotated[:, keep].copy()
    filtered.uns["model_union_filter"] = {
        "drop_genes": sorted(MODEL_UNION_DROP_GENES),
        "n_genes_before": int(full_annotated.n_vars),
        "n_genes_after": int(filtered.n_vars),
        "removed_genes_csv": str(removed_csv),
        "note": "Dropped only model-union missing genes; did not drop TACCO/ref-overlap-missing genes.",
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    log(f"Writing model-union-filtered annotated AnnData: {output_path}")
    filtered.write_h5ad(output_path)
    log(f"Union-filtered shape: {filtered.shape}")
    log(f"Removed genes written to: {removed_csv}")
    return filtered


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="TACCO cell typing for Xenium kidney using GSE183273 subclass.l1")
    parser.add_argument("--query-path", default=DEFAULT_QUERY_PATH)
    parser.add_argument("--ref-path", default=DEFAULT_REF_PATH)
    parser.add_argument("--gtf-path", default=DEFAULT_GTF_PATH)
    parser.add_argument("--runs-root", default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--full-output", default=DEFAULT_FULL_OUT)
    parser.add_argument("--model-output", default=DEFAULT_MODEL_OUT)
    parser.add_argument("--label-key", default="subclass.l1")
    parser.add_argument("--ref-counts-source", default="X")
    parser.add_argument("--query-counts-source", default="X")
    parser.add_argument("--ref-gene-id-mode", default="auto", choices=["auto", "symbol", "ensembl_to_symbol"])
    parser.add_argument("--disable-strict-count-check", action="store_true")
    parser.add_argument("--skip-model-filtered-output", action="store_true")
    args = parser.parse_args()

    global STRICT_COUNT_CHECK
    if args.disable_strict_count_check:
        STRICT_COUNT_CHECK = False

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.runs_root) / f"{timestamp}__GSE183273__subclass_l1__xenium_kidney"
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "timestamp": timestamp,
        "run_dir": str(run_dir),
        "reference_mode": "GSE183273",
        "label_key": args.label_key,
        "query_path": args.query_path,
        "ref_path": args.ref_path,
        "gtf_path": args.gtf_path,
        "method": METHOD,
        "platform_iterations": PLATFORM_ITERATIONS,
        "normalize_to": NORMALIZE_TO,
        "n_hvg": N_HVG,
        "strict_count_check": STRICT_COUNT_CHECK,
        "confidence_filtering": "disabled_for_label_removal; flags only",
        "model_union_drop_genes": sorted(MODEL_UNION_DROP_GENES),
        "full_output": args.full_output,
        "model_output": None if args.skip_model_filtered_output else args.model_output,
    }
    with open(run_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    log(f"Run directory: {run_dir}")

    ensembl_to_symbol = build_ensembl_to_symbol_map_from_gtf(args.gtf_path)

    ref = read_reference(
        ref_path=args.ref_path,
        label_key=args.label_key,
        counts_source=args.ref_counts_source,
        gene_id_mode=args.ref_gene_id_mode,
        ensembl_to_symbol=ensembl_to_symbol,
    )
    query_full = read_query(args.query_path, counts_source=args.query_counts_source)

    # Report before TACCO.
    shared_pre = ref.var_names.intersection(query_full.var_names)
    report = {
        "reference": inspect_matrix_kind(ref, "reference"),
        "query_full": inspect_matrix_kind(query_full, "query_full"),
        "label_key": args.label_key,
        "n_reference_labels": int(ref.obs[args.label_key].nunique()),
        "reference_label_counts_top50": ref.obs[args.label_key].value_counts().head(50).to_dict(),
        "n_shared_genes_for_tacco": int(len(shared_pre)),
        "frac_query_genes_shared_for_tacco": float(len(shared_pre) / max(query_full.n_vars, 1)),
        "frac_ref_genes_shared_for_tacco": float(len(shared_pre) / max(ref.n_vars, 1)),
        "shared_genes_preview": list(shared_pre[:50]),
    }
    with open(run_dir / "compatibility_report.json", "w") as f:
        json.dump(report, f, indent=2)
    log(
        f"Compatibility: shared genes={len(shared_pre)} "
        f"({len(shared_pre)}/{query_full.n_vars} query genes; {len(shared_pre)}/{ref.n_vars} ref genes)"
    )

    ref_sub, query_sub, shared = subset_to_shared_genes(ref, query_full)
    query_sub.uns["tacco_run"] = {
        "reference_mode": "GSE183273",
        "label_key": args.label_key,
        "ref_path": args.ref_path,
        "query_path": args.query_path,
        "n_ref_cells": int(ref_sub.n_obs),
        "n_query_cells": int(query_sub.n_obs),
        "n_shared_genes_used_in_tacco": int(len(shared)),
        "run_dir": str(run_dir),
        "confidence_filtering": "disabled; tacco_celltype is top1 for all cells",
    }

    log("Running TACCO annotate")
    annotate_with_tacco(query_sub, ref_sub, args.label_key)
    log("Finished TACCO annotate")

    conf_summary = add_top_label_no_removal(query_sub)
    query_sub.uns["confidence_summary_no_removal"] = conf_summary
    with open(run_dir / "confidence_summary_no_removal.json", "w") as f:
        json.dump(conf_summary, f, indent=2)
    log(
        f"Low-confidence cells flagged but NOT removed: "
        f"{conf_summary['n_low_confidence_cells_flagged_not_removed']}/{conf_summary['n_total_cells']} "
        f"({conf_summary['frac_low_confidence_cells_flagged_not_removed']:.2%})"
    )

    # Predictions CSV from full cell set.
    pred_csv = run_dir / "xenium_kidney.predictions.csv"
    query_sub.obs[prediction_columns()].to_csv(pred_csv)
    log(f"Wrote predictions: {pred_csv}")

    log("Copying TACCO labels/scores back to full Xenium gene panel")
    full_annotated = copy_annotations_back_to_full_panel(query_full, query_sub)
    full_annotated = sanitize_score_obsm_for_h5ad(full_annotated)
    full_annotated = sanitize_anndata_for_h5ad(full_annotated)

    Path(args.full_output).parent.mkdir(parents=True, exist_ok=True)
    log(f"Writing full-panel annotated AnnData: {args.full_output}")
    full_annotated.write_h5ad(args.full_output)
    log(f"Full-panel shape: {full_annotated.shape}")

    # Also save a copy inside the run directory for provenance if desired via symlink-safe path note.
    with open(run_dir / "output_paths.json", "w") as f:
        json.dump(
            {
                "full_panel_annotated_h5ad": args.full_output,
                "model_union_filtered_annotated_h5ad": None if args.skip_model_filtered_output else args.model_output,
                "predictions_csv": str(pred_csv),
            },
            f,
            indent=2,
        )

    if not args.skip_model_filtered_output:
        filtered = write_model_union_filtered(full_annotated, args.model_output, run_dir)
        if filtered.n_vars != full_annotated.n_vars - len(MODEL_UNION_DROP_GENES):
            warnings.warn(
                f"Expected {full_annotated.n_vars - len(MODEL_UNION_DROP_GENES)} genes after model-union filter, "
                f"got {filtered.n_vars}. Check model_union_removed_genes.csv."
            )

    log("Done")
    log(f"Full-panel annotated h5ad: {args.full_output}")
    if not args.skip_model_filtered_output:
        log(f"Model-union-filtered annotated h5ad: {args.model_output}")


if __name__ == "__main__":
    main()
