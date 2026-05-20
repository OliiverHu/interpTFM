#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import scanpy as sc

from interp_pipeline.get_annotation.f1_alignment import heldout_report_for_layer
from interp_pipeline.get_annotation.gprofiler_client import GProfilerClient, GProfilerSpec
from interp_pipeline.get_annotation.panel import panel_from_cosmx_adata


DEFAULT_THRESHOLDS = [0.0, 0.15, 0.3, 0.6]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def pick_qval_col(df: pd.DataFrame) -> str:
    for c in ["p_value", "p_value_fdr", "p_value_adj", "p_value_corrected"]:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot find p/q-value column in columns={list(df.columns)[:40]}")


def as_list(x):
    if x is None:
        return []
    if isinstance(x, float) and np.isnan(x):
        return []
    if isinstance(x, list):
        out = []
        for v in x:
            if isinstance(v, str):
                out.append(v)
            elif isinstance(v, dict):
                if "name" in v:
                    out.append(str(v["name"]))
                elif "id" in v:
                    out.append(str(v["id"]))
        return out
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        if "," in s:
            return [p.strip() for p in s.split(",") if p.strip()]
        return [s]
    return [str(x)]


def write_panel_conditioned_manifest(
    out_dir: str,
    adata_path: str,
    alpha: float,
    sources: List[str],
    gt_csv: str,
    enrichment_csv: str,
    terms_tsv: str,
) -> None:
    manifest = {
        "benchmark_type": "panel_conditioned_concept_alignment",
        "important_note": (
            "GT concepts are selected by g:Profiler enrichment on the same CosMx "
            "panel gene universe used for evaluation. This is not classic SAE-label "
            "train/test leakage, but it can inflate F1 through panel curation and "
            "redundant/small concept columns. Interpret as panel-conditioned alignment."
        ),
        "adata_path": adata_path,
        "gprofiler_alpha": alpha,
        "gprofiler_sources": sources,
        "gt_csv": gt_csv,
        "enrichment_csv": enrichment_csv,
        "terms_tsv": terms_tsv,
    }
    with open(os.path.join(out_dir, "panel_conditioned_gt_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


def compute_gt_diagnostics(gt_csv: str, terms_tsv: str | None, out_dir: str) -> Dict[str, Any]:
    ensure_dir(out_dir)
    gt = pd.read_csv(gt_csv, index_col=0)
    if gt.empty:
        raise ValueError(f"GT matrix is empty: {gt_csv}")

    X_df = gt.fillna(0).astype(int).clip(0, 1)
    sizes = X_df.sum(axis=0).astype(int)

    dup_mask = X_df.T.duplicated(keep=False)
    duplicated_terms = list(X_df.columns[dup_mask])
    duplicate_count = int(X_df.T.duplicated().sum())

    X = X_df.values.astype(bool).T
    n_terms = X.shape[0]
    max_jaccard = 0.0
    near_duplicate_pairs = []

    if n_terms >= 2:
        inter = X.astype(np.int32) @ X.astype(np.int32).T
        term_sizes = X.sum(axis=1).astype(np.int32)
        union = term_sizes[:, None] + term_sizes[None, :] - inter
        jacc = inter / np.maximum(union, 1)
        np.fill_diagonal(jacc, 0.0)
        max_jaccard = float(np.max(jacc))

        pairs = np.argwhere(np.triu(jacc >= 0.90, k=1))
        for i, j in pairs:
            near_duplicate_pairs.append(
                {
                    "term_a": str(X_df.columns[int(i)]),
                    "term_b": str(X_df.columns[int(j)]),
                    "jaccard": float(jacc[int(i), int(j)]),
                    "size_a": int(term_sizes[int(i)]),
                    "size_b": int(term_sizes[int(j)]),
                    "intersection": int(inter[int(i), int(j)]),
                }
            )

    size_df = pd.DataFrame(
        {
            "term_id": X_df.columns.astype(str),
            "panel_gene_count": sizes.values,
            "is_duplicate_column": [bool(t in duplicated_terms) for t in X_df.columns.astype(str)],
        }
    )

    if terms_tsv is not None and os.path.exists(terms_tsv):
        meta = pd.read_csv(terms_tsv, sep="\t")
        if "term_id" in meta.columns:
            size_df = size_df.merge(meta, on="term_id", how="left")

    size_df = size_df.sort_values("panel_gene_count")
    size_df.to_csv(os.path.join(out_dir, "gt_concept_size_diagnostics.csv"), index=False)

    pairs_df = pd.DataFrame(near_duplicate_pairs)
    pairs_df.to_csv(os.path.join(out_dir, "gt_near_duplicate_pairs_jaccard_ge_0p90.csv"), index=False)

    summary = {
        "gt_csv": gt_csv,
        "n_panel_genes": int(X_df.shape[0]),
        "n_concepts": int(X_df.shape[1]),
        "duplicate_concept_columns_count": duplicate_count,
        "duplicated_terms_keep_false_count": int(len(duplicated_terms)),
        "near_duplicate_pairs_jaccard_ge_0p90": int(len(near_duplicate_pairs)),
        "max_off_diagonal_jaccard": max_jaccard,
        "concept_size_min": int(sizes.min()),
        "concept_size_p05": float(sizes.quantile(0.05)),
        "concept_size_median": float(sizes.median()),
        "concept_size_mean": float(sizes.mean()),
        "concept_size_p95": float(sizes.quantile(0.95)),
        "concept_size_max": int(sizes.max()),
        "n_concepts_size_le_2": int((sizes <= 2).sum()),
        "n_concepts_size_le_3": int((sizes <= 3).sum()),
        "n_concepts_size_le_5": int((sizes <= 5).sum()),
        "diagnostic_files": {
            "concept_sizes": os.path.join(out_dir, "gt_concept_size_diagnostics.csv"),
            "near_duplicate_pairs": os.path.join(out_dir, "gt_near_duplicate_pairs_jaccard_ge_0p90.csv"),
        },
    }

    with open(os.path.join(out_dir, "gt_diagnostics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def build_gprofiler_gt(
    adata_path: str,
    out_dir: str,
    alpha: float,
    sources: List[str],
    force_rebuild: bool = False,
    min_panel_overlap: int = 2,
    max_panel_overlap_frac: float = 1.0,
) -> str:
    ensure_dir(out_dir)
    bin_path = os.path.join(out_dir, "gprofiler_binary_gene_by_term.csv")
    enr_path = os.path.join(out_dir, "gprofiler_enrichment.csv")
    meta_path = os.path.join(out_dir, "gprofiler_terms.tsv")

    if os.path.exists(bin_path) and not force_rebuild:
        print(f"[gt] Reusing cached GT: {bin_path}")
        print("[gt] Use --force-rebuild-gt if alpha/sources/adata/min-overlap changed.")
        write_panel_conditioned_manifest(out_dir, adata_path, alpha, sources, bin_path, enr_path, meta_path)
        return bin_path

    adata = sc.read_h5ad(adata_path)
    symbol_col = "feature_name" if "feature_name" in adata.var.columns else "index"
    panel = panel_from_cosmx_adata(adata, symbol_col=symbol_col)
    genes_ens = panel.ensembl_ids
    genes_sym = panel.symbols
    sym_to_ens = {s: e for e, s in panel.ens_to_sym.items()}

    gp = GProfilerClient(cache_dir=os.path.join(out_dir, "gprof_cache"))
    spec = GProfilerSpec(
        organism="hsapiens",
        sources=sources,
        user_threshold=float(alpha),
        significance_threshold_method="fdr",
        return_dataframe=True,
    )
    res = gp.profile(genes_sym, spec=spec, query_name="cosmx_panel", force=False)
    if not isinstance(res, pd.DataFrame):
        res = pd.DataFrame(res)
    res.to_csv(enr_path, index=False)

    qcol = pick_qval_col(res)
    res_f = res[res[qcol] <= alpha].copy()

    term_id_col = "native" if "native" in res_f.columns else ("term_id" if "term_id" in res_f.columns else None)
    term_name_col = "name" if "name" in res_f.columns else ("term_name" if "term_name" in res_f.columns else None)
    source_col = "source" if "source" in res_f.columns else None
    intersection_col = "intersections" if "intersections" in res_f.columns else ("intersection" if "intersection" in res_f.columns else None)
    if term_id_col is None or intersection_col is None:
        raise ValueError(f"g:Profiler output missing required columns. Have: {list(res_f.columns)[:50]}")

    gene_index = {g: i for i, g in enumerate(genes_ens)}
    max_panel_overlap = int(np.floor(len(genes_ens) * float(max_panel_overlap_frac)))
    max_panel_overlap = max(max_panel_overlap, int(min_panel_overlap))

    mat = np.zeros((len(genes_ens), len(res_f)), dtype=np.int8)
    terms: List[str] = []
    meta_rows: List[Dict[str, Any]] = []
    kept = 0

    for row in res_f.itertuples(index=False):
        rowd = row._asdict() if hasattr(row, "_asdict") else dict(row)
        overlap_syms = as_list(rowd.get(intersection_col))
        if not overlap_syms:
            continue

        hit_indices = []
        for sym in overlap_syms:
            ens = sym_to_ens.get(str(sym))
            if ens is None:
                continue
            idx = gene_index.get(ens)
            if idx is None:
                continue
            hit_indices.append(idx)

        hit_indices = sorted(set(hit_indices))
        hit = len(hit_indices)
        if hit < int(min_panel_overlap):
            continue
        if hit > max_panel_overlap:
            continue

        for idx in hit_indices:
            mat[idx, kept] = 1

        term_id = str(rowd.get(term_id_col))
        terms.append(term_id)
        meta_rows.append(
            {
                "term_id": term_id,
                "term_name": str(rowd.get(term_name_col)) if term_name_col else "",
                "source": str(rowd.get(source_col)) if source_col else "",
                "q_value_col": qcol,
                "q_value": float(rowd.get(qcol)),
                "overlap_in_panel": hit,
            }
        )
        kept += 1

    mat = mat[:, :kept]
    pd.DataFrame(mat, index=genes_ens, columns=terms).to_csv(bin_path, index=True)
    pd.DataFrame(meta_rows).to_csv(meta_path, sep="\t", index=False)
    write_panel_conditioned_manifest(out_dir, adata_path, alpha, sources, bin_path, enr_path, meta_path)
    return bin_path


def prepare_adata_schema(adata_path: str, out_root: str) -> str:
    """
    Schema adapter only. This does NOT create a heldout dataset.
    It writes the same adata with expected var fields for downstream evaluation.
    """
    adata = sc.read_h5ad(adata_path).copy()
    adata.var["ensembl_id"] = adata.var_names.astype(str)

    if "feature_name" in adata.var.columns:
        adata.var["gene_symbol"] = adata.var["feature_name"].astype(str)
        adata.var["index"] = adata.var["feature_name"].astype(str)
    elif "index" in adata.var.columns:
        adata.var["gene_symbol"] = adata.var["index"].astype(str)
    else:
        raise ValueError("Need feature_name or index in adata.var")

    tmp_dir = os.path.join(out_root, "_tmp")
    ensure_dir(tmp_dir)
    out = os.path.join(tmp_dir, "schema_patched_same_adata.h5ad")
    adata.write(out)

    with open(os.path.join(tmp_dir, "schema_patch_manifest.json"), "w") as f:
        json.dump(
            {
                "important_note": (
                    "This file is the same input adata with var schema patched. "
                    "It is not a heldout split."
                ),
                "input_adata_path": adata_path,
                "output_adata_path": out,
                "n_obs": int(adata.n_obs),
                "n_vars": int(adata.n_vars),
                "var_columns_written_or_checked": ["ensembl_id", "gene_symbol", "index"],
            },
            f,
            indent=2,
        )

    return out


def pick_model_aux_ckpt(label: str, scgpt_foundation_ckpt: str | None) -> str | None:
    if label.lower() == "scgpt":
        return scgpt_foundation_ckpt
    return None


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Run panel-conditioned concept-alignment F1 reports for 3 models. "
            "heldout_report_for_layer performs gene-level valid/test splitting."
        )
    )
    ap.add_argument("--labels", nargs=3, required=True)
    ap.add_argument("--store-roots", nargs=3, required=True)
    ap.add_argument("--layers", nargs=3, required=True)
    ap.add_argument("--sae-ckpts", nargs=3, required=True)

    ap.add_argument(
        "--adata-path",
        default="/maiziezhou_lab2/yunfei/Projects/FM_temp/interGFM/ge_shards/cosmx_human_lung_sec8.h5ad",
    )
    ap.add_argument("--out-root", default="runs/heldout_3models")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--latent-thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS)

    ap.add_argument(
        "--scgpt-foundation-ckpt",
        default="/maiziezhou_lab/zihang/SpatialFoundationModel/eval/scGPT/whole-human-pretrain",
    )

    ap.add_argument("--gprof-alpha", type=float, default=0.05)
    ap.add_argument("--gprof-sources", nargs="+", default=["GO:BP", "GO:MF", "GO:CC", "REAC", "KEGG"])
    ap.add_argument("--force-rebuild-gt", action="store_true")
    ap.add_argument("--min-panel-overlap", type=int, default=2)
    ap.add_argument("--max-panel-overlap-frac", type=float, default=1.0)

    ap.add_argument("--valid-frac", type=float, default=0.2)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--topM-valid-per-concept-per-threshold", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=8192)

    args = ap.parse_args()
    ensure_dir(args.out_root)

    gt_dir = os.path.join(args.out_root, "gprofiler")
    gt_csv = build_gprofiler_gt(
        adata_path=args.adata_path,
        out_dir=gt_dir,
        alpha=args.gprof_alpha,
        sources=args.gprof_sources,
        force_rebuild=args.force_rebuild_gt,
        min_panel_overlap=args.min_panel_overlap,
        max_panel_overlap_frac=args.max_panel_overlap_frac,
    )

    gt_diag = compute_gt_diagnostics(
        gt_csv=gt_csv,
        terms_tsv=os.path.join(gt_dir, "gprofiler_terms.tsv"),
        out_dir=os.path.join(gt_dir, "diagnostics"),
    )
    print("[gt diagnostics]")
    print(json.dumps(gt_diag, indent=2))

    patched_adata = prepare_adata_schema(args.adata_path, args.out_root)

    for label, store_root, layer, sae_ckpt in zip(args.labels, args.store_roots, args.layers, args.sae_ckpts):
        out_dir = os.path.join(args.out_root, label, layer)
        ensure_dir(out_dir)

        aux_ckpt = pick_model_aux_ckpt(label, args.scgpt_foundation_ckpt)

        print("=" * 100)
        print(f"[heldout / concept alignment] model={label}")
        print("  benchmark_type=panel_conditioned_concept_alignment")
        print("  split_type=gene_level_valid_test_inside_heldout_report_for_layer")
        print(f"  valid_frac={args.valid_frac}")
        print(f"  test_frac={args.test_frac}")
        print(f"  seed={args.seed}")
        print(f"  store_root={store_root}")
        print(f"  layer={layer}")
        print(f"  sae_ckpt={sae_ckpt}")
        print(f"  aux_scgpt_ckpt={aux_ckpt}")
        print(f"  out_dir={out_dir}")
        print("=" * 100)

        heldout_report_for_layer(
            layer=layer,
            store_root=store_root,
            gt_csv=gt_csv,
            sae_ckpt_path=sae_ckpt,
            out_dir=out_dir,
            latent_thresholds=list(args.latent_thresholds),
            valid_frac=args.valid_frac,
            test_frac=args.test_frac,
            seed=args.seed,
            topM_valid_per_concept_per_threshold=args.topM_valid_per_concept_per_threshold,
            batch_size=args.batch_size,
            adata_path=patched_adata,
            scgpt_ckpt=aux_ckpt,
            device=args.device,
            dev_mode=False,
            dev_max_shards=None,
            dev_max_rows_per_split_per_shard=None,
            dev_only_valid=False,
        )

    print("\nDONE")


if __name__ == "__main__":
    main()


# Example:
#
# python test_run_f1heldout_3models_updated.py \
#   --labels scgpt c2sscale geneformer \
#   --store-roots \
#     /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/full_scgpt_cosmx \
#     /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/full_c2sscale_cosmx \
#     /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/full_geneformer_cosmx \
#   --layers layer_4.norm2 layer_17 layer_4 \
#   --sae-ckpts \
#     /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/full_scgpt_cosmx/sae/layer_4.norm2/nr_on__steps_6000__l1_3e-3/sae_layer_4.norm2_best.pt \
#     /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/full_c2sscale_cosmx/sae/layer_17/nr_on__steps_6000__l1_3e-3/sae_layer_17_best.pt \
#     /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/full_geneformer_cosmx/sae/layer_4/nr_on__steps_6000__l1_3e-3/sae_layer_4_best.pt \
#   --out-root /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/heldout_3models_l1_3e-3_geneheldout \
#   --valid-frac 0.2 \
#   --test-frac 0.1 \
#   --seed 0 \
#   --min-panel-overlap 2 \
#   --force-rebuild-gt \
#   --device cuda \
#   --scgpt-foundation-ckpt /maiziezhou_lab/zihang/SpatialFoundationModel/eval/scGPT/whole-human-pretrain
