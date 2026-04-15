#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


def build_gprofiler_gt(adata_path: str, out_dir: str, alpha: float, sources: List[str]) -> str:
    ensure_dir(out_dir)
    bin_path = os.path.join(out_dir, "gprofiler_binary_gene_by_term.csv")
    enr_path = os.path.join(out_dir, "gprofiler_enrichment.csv")
    meta_path = os.path.join(out_dir, "gprofiler_terms.tsv")
    if os.path.exists(bin_path):
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
    mat = np.zeros((len(genes_ens), len(res_f)), dtype=np.int8)
    terms: List[str] = []
    meta_rows: List[Dict[str, Any]] = []
    kept = 0

    for row in res_f.itertuples(index=False):
        rowd = row._asdict() if hasattr(row, "_asdict") else dict(row)
        overlap_syms = as_list(rowd.get(intersection_col))
        if not overlap_syms:
            continue

        hit = 0
        for sym in overlap_syms:
            ens = sym_to_ens.get(str(sym))
            if ens is None:
                continue
            idx = gene_index.get(ens)
            if idx is None:
                continue
            mat[idx, kept] = 1
            hit += 1

        if hit == 0:
            continue

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
    return bin_path


def prepare_heldout_adata_schema(adata_path: str, out_root: str) -> str:
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
    out = os.path.join(tmp_dir, "heldout_schema_patched.h5ad")
    adata.write(out)
    return out


def pick_model_aux_ckpt(label: str, scgpt_foundation_ckpt: str | None) -> str | None:
    if label.lower() == "scgpt":
        return scgpt_foundation_ckpt
    return None


def main():
    ap = argparse.ArgumentParser(description="Run heldout F1 reports for 3 models with model-specific aux checkpoint handling.")
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

    # only needed for scGPT-style vocab-id decoding
    ap.add_argument(
        "--scgpt-foundation-ckpt",
        default="/maiziezhou_lab/zihang/SpatialFoundationModel/eval/scGPT/whole-human-pretrain",
    )

    ap.add_argument("--gprof-alpha", type=float, default=0.05)
    ap.add_argument("--gprof-sources", nargs="+", default=["GO:BP", "GO:MF", "GO:CC", "REAC", "KEGG"])
    args = ap.parse_args()

    ensure_dir(args.out_root)

    gt_csv = build_gprofiler_gt(
        adata_path=args.adata_path,
        out_dir=os.path.join(args.out_root, "gprofiler"),
        alpha=args.gprof_alpha,
        sources=args.gprof_sources,
    )
    patched_adata = prepare_heldout_adata_schema(args.adata_path, args.out_root)

    for label, store_root, layer, sae_ckpt in zip(args.labels, args.store_roots, args.layers, args.sae_ckpts):
        out_dir = os.path.join(args.out_root, label, layer)
        ensure_dir(out_dir)

        aux_ckpt = pick_model_aux_ckpt(label, args.scgpt_foundation_ckpt)

        print("=" * 100)
        print(f"[heldout] model={label}")
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


# python test_run_f1heldout_3models.py \
#   --labels scgpt c2sscale geneformer \
#   --store-roots \
#     runs/full_scgpt_cosmx \
#     runs/full_c2sscale_cosmx \
#     runs/full_geneformer_cosmx \
#   --layers layer_4.norm2 layer_17 layer_4 \
#   --sae-ckpts \
#     runs/full_scgpt_cosmx/sae/layer_4.norm2/nr_on__steps_6000__l1_3e-3/sae_layer_4.norm2_best.pt \
#     runs/full_c2sscale_cosmx/sae/layer_17/nr_on__steps_6000__l1_3e-3/sae_layer_17_best.pt \
#     runs/full_geneformer_cosmx/sae/layer_4/nr_on__steps_6000__l1_3e-3/sae_layer_4_best.pt \
#   --out-root runs/heldout_3models_l1_3e-3 \
#   --device cuda \
#   --scgpt-foundation-ckpt /maiziezhou_lab/zihang/SpatialFoundationModel/eval/scGPT/whole-human-pretrain