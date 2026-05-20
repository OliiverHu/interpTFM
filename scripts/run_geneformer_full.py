from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import scanpy as sc

from interp_pipeline.extraction.geneformer_extraction import (
    prepare_geneformer_h5ad,
    tokenize_geneformer_dataset,
    extract_geneformer_to_store,
)
from interp_pipeline.get_annotation.f1_alignment import heldout_report_for_layer
from interp_pipeline.get_annotation.gprofiler_client import GProfilerClient, GProfilerSpec
from interp_pipeline.get_annotation.panel import panel_from_cosmx_adata
from interp_pipeline.io.activation_store import ActivationStore, ActivationStoreSpec
from interp_pipeline.sae.sae_base import SAESpec
from interp_pipeline.sae.trainers import fit_sae_for_layer

try:
    from interp_pipeline.sae.normalize import normalize_sae_checkpoint
except Exception:
    normalize_sae_checkpoint = None


ADATA_PATH = "/maiziezhou_lab2/yunfei/Projects/FM_temp/interGFM/ge_shards/cosmx_human_lung_sec8.h5ad"
MODEL_DIR = "/maiziezhou_lab2/yunfei/geneformer_hf"
OUT_ROOT = "runs/full_geneformer_cosmx"
DEFAULT_LAYERS = ["layer_4"]
DEFAULT_THRESHOLDS = [0.0, 0.15, 0.3, 0.6]

SAE_SPEC = SAESpec(
    n_latents=4096,
    l1=1e-3,
    lr=1e-4,
    steps=20000,
    warmup_steps=1000,
    resample_steps=2000,
    seed=0,
)
SAE_BATCH = 1024

GPROF_ALPHA = 0.05
GPROF_SOURCES = ["GO:BP", "GO:MF", "GO:CC", "REAC", "KEGG"]


def geneformer_activation_shards_exist(out_root: str, layer: str) -> bool:
    layer_dir = os.path.join(out_root, "activations", layer)
    if not os.path.isdir(layer_dir):
        return False
    return any(
        os.path.exists(os.path.join(layer_dir, d, "activations.pt"))
        and os.path.exists(os.path.join(layer_dir, d, "index.pt"))
        for d in os.listdir(layer_dir)
        if d.startswith("shard_")
    )


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def pick_qval_col(df: pd.DataFrame) -> str:
    for c in ["p_value", "p_value_fdr", "p_value_adj", "p_value_corrected"]:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot find p/q-value column in {list(df.columns)[:40]}")


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


def maybe_normalize(layer: str, out_root: str, ckpt_path: str, device: str, enabled: bool) -> str:
    if not enabled:
        return ckpt_path
    if normalize_sae_checkpoint is None:
        print("[warn] normalize_sae_checkpoint not available; using raw checkpoint")
        return ckpt_path
    return normalize_sae_checkpoint(
        ckpt_path=ckpt_path,
        store_root=out_root,
        layer=layer,
        output_path=None,
        device=device,
        max_shards=None,
        token_chunk_size=4096,
        feature_chunk_size=1024,
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adata-path", default=ADATA_PATH)
    p.add_argument("--model-dir", default=MODEL_DIR)
    p.add_argument("--out-root", default=OUT_ROOT)
    p.add_argument("--device", default="cuda")
    p.add_argument("--layers", nargs="+", default=DEFAULT_LAYERS)
    p.add_argument("--latent-thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS)
    p.add_argument("--skip-tokenize", action="store_true")
    p.add_argument("--skip-extract", action="store_true")
    p.add_argument("--skip-sae", action="store_true")
    p.add_argument("--skip-heldout", action="store_true")
    p.add_argument("--normalize-sae-features", dest="normalize_sae_features", action="store_true", default=True)
    p.add_argument("--no-normalize-sae-features", dest="normalize_sae_features", action="store_false")
    p.add_argument("--forward-batch-size", type=int, default=8)
    p.add_argument("--tokenizer-nproc", type=int, default=1)
    p.add_argument("--model-version", default="V2")
    return p.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.out_root)

    tokenized_prefix = f"cosmx_{args.model_version.lower()}"
    tokenized_path = os.path.join(args.out_root, "tokenized", f"{tokenized_prefix}.dataset")

    print("=" * 80)
    print("run_geneformer_full.py")
    print(f"ADATA_PATH      = {args.adata_path}")
    print(f"MODEL_DIR       = {args.model_dir}")
    print(f"OUT_ROOT        = {args.out_root}")
    print(f"MODEL_VERSION   = {args.model_version}")
    print(f"LAYERS          = {args.layers}")
    print(f"TOKENIZED_PATH  = {tokenized_path}")
    print("=" * 80)

    gt_csv = build_gprofiler_gt(
        adata_path=args.adata_path,
        out_dir=os.path.join(args.out_root, "gprofiler"),
        alpha=GPROF_ALPHA,
        sources=GPROF_SOURCES,
    )

    prepared = prepare_geneformer_h5ad(
        adata_path=args.adata_path,
        output_path=os.path.join(args.out_root, "prepared", "cosmx.prepared.h5ad"),
    )

    if not args.skip_tokenize:
        tokenized_path = tokenize_geneformer_dataset(
            prepared_h5ad_path=prepared,
            output_dir=os.path.join(args.out_root, "tokenized"),
            output_prefix=tokenized_prefix,
            model_version=args.model_version,
            nproc=args.tokenizer_nproc,
        )
    else:
        print("[tokenize] skipped")

    if not os.path.exists(tokenized_path):
        raise FileNotFoundError(f"Tokenized dataset not found: {tokenized_path}")

        if args.skip_extract:
            print("[extract] skipped by flag")
        else:
            missing_layers = [
                layer for layer in args.layers
                if not geneformer_activation_shards_exist(args.out_root, layer)
            ]

            if not missing_layers:
                print(f"[extract] found existing activation shards for layers={args.layers}; skipping extraction")
            else:
                print(f"[extract] missing activation shards for layers={missing_layers}; running extraction")
                extract_geneformer_to_store(
                    model_dir=args.model_dir,
                    tokenized_dataset_path=tokenized_path,
                    store_root=args.out_root,
                    layers=missing_layers,
                    model_version=args.model_version,
                    device=args.device,
                    forward_batch_size=args.forward_batch_size,
                )

    store = ActivationStore(ActivationStoreSpec(root=args.out_root))
    sae_ckpts: Dict[str, str] = {}

    for layer in args.layers:
        sae_out_dir = os.path.join(args.out_root, "sae", layer)
        ensure_dir(sae_out_dir)

        if not args.skip_sae:
            res = fit_sae_for_layer(
                store=store,
                layer=layer,
                spec=SAE_SPEC,
                output_dir=sae_out_dir,
                device=args.device,
                batch_size=SAE_BATCH,
            )
            sae_ckpt = res.model_path
        else:
            sae_ckpt = os.path.join(sae_out_dir, f"sae_{layer}.pt")
            if not os.path.exists(sae_ckpt):
                raise FileNotFoundError(f"Missing SAE checkpoint: {sae_ckpt}")

        sae_ckpts[layer] = maybe_normalize(
            layer=layer,
            out_root=args.out_root,
            ckpt_path=sae_ckpt,
            device=args.device,
            enabled=args.normalize_sae_features,
        )

    if args.skip_heldout:
        print("[heldout] skipped")
        print("DONE")
        return

    patched_adata = prepare_heldout_adata_schema(args.adata_path, args.out_root)

    for layer in args.layers:
        out_dir = os.path.join(args.out_root, "heldout_report", layer)
        ensure_dir(out_dir)

        heldout_report_for_layer(
            layer=layer,
            store_root=args.out_root,
            gt_csv=gt_csv,
            sae_ckpt_path=sae_ckpts[layer],
            out_dir=out_dir,
            latent_thresholds=list(args.latent_thresholds),
            adata_path=patched_adata,
            scgpt_ckpt=None,
            device=args.device,
            dev_mode=False,
            dev_max_shards=None,
            dev_max_rows_per_split_per_shard=None,
            dev_only_valid=False,
        )

    print("DONE")


if __name__ == "__main__":
    main()