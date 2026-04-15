from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from tqdm import tqdm

from interp_pipeline.extraction.c2s_extraction import extract_c2s_dataset
from interp_pipeline.get_annotation.f1_alignment import heldout_report_for_layer
from interp_pipeline.get_annotation.gprofiler_client import GProfilerClient, GProfilerSpec
from interp_pipeline.get_annotation.panel import panel_from_cosmx_adata
from interp_pipeline.io.activation_store import ActivationStore, ActivationStoreSpec
from interp_pipeline.sae.normalize import normalize_sae_features
from interp_pipeline.sae.sae_base import SAESpec
from interp_pipeline.sae.trainers import fit_sae_for_layer

ADATA_SHARD_PATH = "/maiziezhou_lab2/yunfei/Projects/FM_temp/interGFM/ge_shards/cosmx_human_lung_sec8.h5ad"
ADATA_SOURCE_PATH = "/maiziezhou_lab2/yunfei/Projects/FM_temp/datasets/cosmx/lung/cosmx_human_lung.h5ad"
MODEL_PATH = "/maiziezhou_lab2/yunfei/Projects/interpTFM-legacy/c2sscale/models/C2S-Scale-Gemma-2-2B"
EXTRACTION_ROOT = "/maiziezhou_lab2/yunfei/Projects/interpTFM/c2s_full_extraction"
OUT_ROOT = "runs/full_c2sscale_cosmx"
DEVICE = "cuda:0"
DEFAULT_LAYERS = ["layer_17"]
DEFAULT_EXTRACT_LAYERS = [f"layer_{i}" for i in [0, 6, 13, 15, 17, 19, 21, 23, 25]]
DEFAULT_LATENT_THRESHOLDS = [0.0, 0.15, 0.3, 0.6]

SAE_SPEC = SAESpec(
    n_latents=4096,
    l1=1e-3,
    lr=1e-4,
    steps=20_000,
    seed=0,
)
SAE_TRAIN_BATCH = 1024

GPROF_ALPHA = 0.05
GPROF_SOURCES = ["GO:BP", "GO:MF", "GO:CC", "REAC", "KEGG"]


@dataclass(frozen=True)
class ExtractionConfig:
    shards: int = 60
    shard_key: str = "shards"
    batch_size: int = 4
    max_genes: int = 256
    device: str = DEVICE
    pooling: str = "last"
    save_dtype: str = "fp16"
    pool_dtype: str = "fp32"
    normalize: bool = True
    cache_dir: Optional[str] = None


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def pick_qval_col(df: pd.DataFrame) -> str:
    for c in ["p_value", "p_value_fdr", "p_value_adj", "p_value_corrected"]:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot find p/q-value column in dataframe columns: {list(df.columns)[:40]}")


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
        if s.startswith("[") and s.endswith("]"):
            s2 = s.strip("[]").strip()
            if not s2:
                return []
            parts = [p.strip().strip("'").strip('"') for p in s2.split(",")]
            return [p for p in parts if p]
        if "," in s:
            return [p.strip() for p in s.split(",") if p.strip()]
        return [s]
    return [str(x)]


def build_gprofiler_gt(
    adata_path: str,
    out_dir: str,
    alpha: float,
    sources: List[str],
    reuse_if_exists: bool = True,
) -> str:
    ensure_dir(out_dir)
    enr_path = os.path.join(out_dir, "gprofiler_enrichment.csv")
    bin_path = os.path.join(out_dir, "gprofiler_binary_gene_by_term.csv")
    meta_path = os.path.join(out_dir, "gprofiler_terms.tsv")
    if reuse_if_exists and os.path.exists(bin_path):
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
        term_id = str(rowd.get(term_id_col))
        term_name = str(rowd.get(term_name_col)) if term_name_col else ""
        source = str(rowd.get(source_col)) if source_col else ""
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

        terms.append(term_id)
        meta_rows.append(
            {
                "term_id": term_id,
                "term_name": term_name,
                "source": source,
                "q_value_col": qcol,
                "q_value": float(rowd.get(qcol)),
                "overlap_in_panel": hit,
                "overlap_list_len": len(overlap_syms),
            }
        )
        kept += 1

    mat = mat[:, :kept]
    pd.DataFrame(mat, index=genes_ens, columns=terms).to_csv(bin_path, index=True)
    pd.DataFrame(meta_rows).to_csv(meta_path, sep="\t", index=False)
    return bin_path


def load_or_prepare_sec8_h5ad(shard_path: str, source_path: str, n_shards: int = 60, seed: int = 0):
    p = Path(shard_path)
    if p.exists():
        print(f"[adata] using cached shard file: {p}")
        return sc.read_h5ad(p)

    print(f"[adata] building shard file at: {p}")
    adata = sc.read_h5ad(source_path)
    adata = adata[adata.obs["library_key"] == 7].copy()

    X = adata.X
    if sp.issparse(X):
        row_sums = np.asarray(X.sum(axis=1)).ravel()
        non_zero_mask = row_sums != 0
    else:
        X = X if isinstance(X, np.ndarray) else np.asarray(X)
        non_zero_mask = ~(np.all(X == 0, axis=1))
    adata = adata[non_zero_mask].copy()

    rng = np.random.default_rng(seed)
    adata.obs["shards"] = rng.choice([f"shard_{i}" for i in range(n_shards)], size=adata.n_obs)
    p.parent.mkdir(parents=True, exist_ok=True)
    adata.write(p)
    return adata


def c2s_gene_batches_exist(extraction_root: str, layer: str) -> bool:
    pat = os.path.join(extraction_root, "activations", layer, "shard_*", "batch_*_gene_acts.pt")
    return len(glob.glob(pat)) > 0


def generic_activation_shards_exist(out_root: str, layer: str) -> bool:
    pat = os.path.join(out_root, "activations", layer, "shard_*", "activations.pt")
    return len(glob.glob(pat)) > 0


def prepare_heldout_adata_schema(adata_path: str, out_root: str) -> str:
    adata = sc.read_h5ad(adata_path).copy()

    original_ens = adata.var_names.astype(str)
    adata.var["ensembl_id"] = original_ens

    if "feature_name" in adata.var.columns:
        gene_symbols = adata.var["feature_name"].astype(str)
        adata.var["gene_symbol"] = gene_symbols
        adata.var["index"] = gene_symbols
    elif "index" in adata.var.columns:
        gene_symbols = adata.var["index"].astype(str)
        adata.var["gene_symbol"] = gene_symbols
    else:
        raise ValueError(
            "Need either adata.var['feature_name'] or adata.var['index'] to build symbol->Ensembl mapping."
        )

    mapped = sum(
        1
        for sym, ens in zip(adata.var["gene_symbol"].astype(str), adata.var["ensembl_id"].astype(str))
        if sym and ens and sym.lower() != "nan" and ens.lower() != "nan"
    )
    print(f"[mapping] prepared heldout AnnData schema with {mapped}/{adata.n_vars} symbol->Ensembl entries")

    tmp_dir = os.path.join(out_root, "_tmp")
    ensure_dir(tmp_dir)
    patched_path = os.path.join(tmp_dir, "heldout_schema_patched.h5ad")
    adata.write(patched_path)
    print(f"[mapping] wrote patched heldout h5ad: {patched_path}")
    return patched_path


def _read_pairs_file(path: str) -> Tuple[List[str], List[str]]:
    example_ids: List[str] = []
    token_ids: List[str] = []
    with open(path, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise RuntimeError(f"Malformed cell/gene pair line in {path!r}: {line!r}")
            cell_id, gene = parts
            example_ids.append(cell_id)
            token_ids.append(gene)
    return example_ids, token_ids


def convert_c2s_layer_to_activation_store(c2s_root: str, out_root: str, layer: str, overwrite: bool = False) -> int:
    layer_dir = os.path.join(c2s_root, "activations", layer)
    if not os.path.isdir(layer_dir):
        raise FileNotFoundError(f"Missing extracted c2s layer directory: {layer_dir}")

    store = ActivationStore(ActivationStoreSpec(root=out_root))
    shard_dirs = sorted([p for p in glob.glob(os.path.join(layer_dir, "shard_*")) if os.path.isdir(p)])
    if not shard_dirs:
        raise RuntimeError(f"No c2s shard dirs found for {layer} under {layer_dir}")

    written = 0
    for shard_dir in tqdm(shard_dirs, desc=f"convert:{layer}"):
        shard_name = os.path.basename(shard_dir)
        shard_id = int(shard_name.split("_")[-1])
        generic_dir = os.path.join(out_root, "activations", layer, shard_name)
        acts_out = os.path.join(generic_dir, "activations.pt")
        idx_out = os.path.join(generic_dir, "index.pt")
        if (not overwrite) and os.path.exists(acts_out) and os.path.exists(idx_out):
            continue

        act_files = sorted(glob.glob(os.path.join(shard_dir, "batch_*_gene_acts.pt")))
        if not act_files:
            continue

        acts_parts: List[torch.Tensor] = []
        example_ids: List[str] = []
        token_ids: List[str] = []
        n_source_batches = 0

        for act_path in act_files:
            stem = os.path.basename(act_path).replace("_gene_acts.pt", "")
            pair_path = os.path.join(shard_dir, f"{stem}_cell_gene_pairs.txt")
            if not os.path.exists(pair_path):
                raise FileNotFoundError(f"Missing pair file for {act_path}: {pair_path}")

            acts = torch.load(act_path, map_location="cpu")
            if not isinstance(acts, torch.Tensor) or acts.ndim != 2:
                raise RuntimeError(f"Expected 2D tensor in {act_path}, got {type(acts)} / ndim={getattr(acts, 'ndim', None)}")
            ex_ids, tok_ids = _read_pairs_file(pair_path)
            if len(ex_ids) != acts.shape[0] or len(tok_ids) != acts.shape[0]:
                raise RuntimeError(
                    f"Row mismatch in {act_path}: acts_rows={acts.shape[0]} pairs={len(ex_ids)}"
                )
            acts_parts.append(acts)
            example_ids.extend(ex_ids)
            token_ids.extend(tok_ids)
            n_source_batches += 1

        if not acts_parts:
            continue

        acts_cat = torch.cat(acts_parts, dim=0)
        store.write_token_shard(
            layer=layer,
            shard_id=shard_id,
            acts=acts_cat,
            example_ids=example_ids,
            token_ids=token_ids,
            token_unit="gene",
            meta={
                "source_format": "c2s_gene_batches",
                "source_layer": layer,
                "source_batches": n_source_batches,
                "source_root": c2s_root,
            },
        )
        written += 1
    return written


def maybe_run_extraction(
    adata,
    model_path: str,
    extraction_root: str,
    layers: Sequence[str],
    cfg: ExtractionConfig,
) -> None:
    missing = [layer for layer in layers if not c2s_gene_batches_exist(extraction_root, layer)]
    if not missing:
        print(f"[extract] found existing c2s activations for layers={list(layers)}; skipping extraction")
        return

    print(f"[extract] missing c2s activations for {missing}; running extract_c2s_dataset(...) into {extraction_root}")
    print(f"[extract] normalize={cfg.normalize}")
    extract_c2s_dataset(
        adata=adata,
        model_path=model_path,
        output_dir=extraction_root,
        layers=list(layers),
        shards=cfg.shards,
        shard_key=cfg.shard_key,
        batch_size=cfg.batch_size,
        max_genes=cfg.max_genes,
        device=cfg.device,
        pooling=cfg.pooling,
        save_dtype=cfg.save_dtype,
        pool_dtype=cfg.pool_dtype,
        normalize=cfg.normalize,
        cache_dir=cfg.cache_dir,
    )


def maybe_convert_c2s_to_generic(c2s_root: str, out_root: str, layers: Sequence[str], overwrite: bool = False) -> None:
    for layer in layers:
        if generic_activation_shards_exist(out_root, layer) and not overwrite:
            print(f"[convert] generic activation shards already exist for {layer}; skipping conversion")
            continue
        n_written = convert_c2s_layer_to_activation_store(c2s_root, out_root, layer, overwrite=overwrite)
        print(f"[convert] layer={layer} wrote/updated {n_written} generic shards")


def train_sae_for_layer(out_root: str, layer: str, device: str) -> str:
    store = ActivationStore(ActivationStoreSpec(root=out_root))
    sae_out_dir = os.path.join(out_root, "sae", layer)
    ensure_dir(sae_out_dir)
    print(f"[sae] training layer={layer}")
    res = fit_sae_for_layer(
        store=store,
        layer=layer,
        spec=SAE_SPEC,
        output_dir=sae_out_dir,
        device=device,
        batch_size=SAE_TRAIN_BATCH,
    )
    print(f"[sae] checkpoint: {res.model_path}")
    return res.model_path



def maybe_normalize_sae_checkpoint(
    out_root: str,
    layer: str,
    sae_ckpt_path: str,
    device: str,
    enabled: bool,
    max_shards: Optional[int] = None,
    token_chunk_size: int = 4096,
    feature_chunk_size: int = 1024,
) -> str:
    if not enabled:
        print(f"[sae-normalize] disabled for {layer}; using raw SAE checkpoint")
        return sae_ckpt_path

    print(f"[sae-normalize] normalizing SAE features for {layer}")
    norm_ckpt = normalize_sae_features(
        ckpt_path=sae_ckpt_path,
        store_root=out_root,
        layer=layer,
        output_path=None,
        device=device,
        max_shards=max_shards,
        token_chunk_size=token_chunk_size,
        feature_chunk_size=feature_chunk_size,
    )
    print(f"[sae-normalize] normalized checkpoint: {norm_ckpt}")
    return norm_ckpt


def run_heldout_for_layer(
    out_root: str,
    layer: str,
    gt_csv: str,
    sae_ckpt_path: str,
    adata_path: str,
    latent_thresholds: Sequence[float],
    device: str,
) -> str:
    heldout_out = os.path.join(out_root, "heldout_report", layer)
    ensure_dir(heldout_out)
    patched_adata_path = prepare_heldout_adata_schema(adata_path=adata_path, out_root=out_root)
    print(f"[heldout] layer={layer}")
    heldout_report_for_layer(
        layer=layer,
        store_root=out_root,
        gt_csv=gt_csv,
        sae_ckpt_path=sae_ckpt_path,
        out_dir=heldout_out,
        latent_thresholds=list(latent_thresholds),
        adata_path=patched_adata_path,
        scgpt_ckpt=None,
        device=device,
        dev_mode=False,
        dev_max_shards=None,
        dev_max_rows_per_split_per_shard=None,
        dev_only_valid=False,
    )
    return heldout_out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run c2s-scale full pipeline up through SAE + heldout report.")
    p.add_argument("--adata-path", default=ADATA_SHARD_PATH)
    p.add_argument("--adata-source-path", default=ADATA_SOURCE_PATH)
    p.add_argument("--model-path", default=MODEL_PATH)
    p.add_argument("--extraction-root", default=EXTRACTION_ROOT)
    p.add_argument("--out-root", default=OUT_ROOT)
    p.add_argument("--device", default=DEVICE)
    p.add_argument("--layers", nargs="+", default=DEFAULT_LAYERS)
    p.add_argument("--extract-layers", nargs="+", default=DEFAULT_EXTRACT_LAYERS)
    p.add_argument("--latent-thresholds", nargs="+", type=float, default=DEFAULT_LATENT_THRESHOLDS)
    p.add_argument("--skip-extraction", action="store_true")
    p.add_argument("--skip-conversion", action="store_true")
    p.add_argument("--overwrite-conversion", action="store_true")
    p.add_argument("--skip-sae", action="store_true")
    p.add_argument("--skip-heldout", action="store_true")
    p.add_argument("--extract-shards", type=int, default=60)
    p.add_argument("--extract-batch-size", type=int, default=4)
    p.add_argument("--extract-max-genes", type=int, default=256)
    p.add_argument("--extract-pooling", default="last")
    p.add_argument("--extract-save-dtype", default="fp16")
    p.add_argument("--extract-pool-dtype", default="fp32")
    p.add_argument(
        "--extract-normalize",
        dest="extract_normalize",
        action="store_true",
        default=True,
        help="Use normalized inputs for c2s extraction (default: on).",
    )
    p.add_argument(
        "--no-extract-normalize",
        dest="extract_normalize",
        action="store_false",
        help="Disable normalization during c2s extraction.",
    )
    p.add_argument("--extract-cache-dir", default=None)
    p.add_argument("--normalize-sae-features", dest="normalize_sae_features", action="store_true", default=True,
                   help="Normalize SAE features to unit max activation before heldout (default: on).")
    p.add_argument("--no-normalize-sae-features", dest="normalize_sae_features", action="store_false",
                   help="Disable post-training SAE feature normalization.")
    p.add_argument("--normalize-sae-max-shards", type=int, default=None)
    p.add_argument("--normalize-sae-token-chunk-size", type=int, default=4096)
    p.add_argument("--normalize-sae-feature-chunk-size", type=int, default=1024)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_root)
    ensure_dir(os.path.join(args.out_root, "gprofiler"))

    print("=" * 80)
    print("run_c2sscale_full.py")
    print(f"ADATA_PATH       = {args.adata_path}")
    print(f"MODEL_PATH       = {args.model_path}")
    print(f"EXTRACTION_ROOT  = {args.extraction_root}")
    print(f"OUT_ROOT         = {args.out_root}")
    print(f"LAYERS           = {args.layers}")
    print(f"EXTRACT_NORMALIZE= {args.extract_normalize}")
    print(f"SAE_NORMALIZE    = {args.normalize_sae_features}")
    print("=" * 80)

    print("[0] build/load g:Profiler GT")
    gt_csv = build_gprofiler_gt(
        adata_path=args.adata_path,
        out_dir=os.path.join(args.out_root, "gprofiler"),
        alpha=GPROF_ALPHA,
        sources=GPROF_SOURCES,
        reuse_if_exists=True,
    )
    print(f"[0] GT csv: {gt_csv}")

    adata = None
    if not args.skip_extraction:
        adata = load_or_prepare_sec8_h5ad(
            shard_path=args.adata_path,
            source_path=args.adata_source_path,
            n_shards=args.extract_shards,
            seed=0,
        )
        maybe_run_extraction(
            adata=adata,
            model_path=args.model_path,
            extraction_root=args.extraction_root,
            layers=args.extract_layers,
            cfg=ExtractionConfig(
                shards=args.extract_shards,
                batch_size=args.extract_batch_size,
                max_genes=args.extract_max_genes,
                device=args.device,
                pooling=args.extract_pooling,
                save_dtype=args.extract_save_dtype,
                pool_dtype=args.extract_pool_dtype,
                normalize=args.extract_normalize,
                cache_dir=args.extract_cache_dir,
            ),
        )
    else:
        print("[1] extraction skipped by flag")

    if not args.skip_conversion:
        print("[2] convert c2s gene batches -> generic ActivationStore shards")
        maybe_convert_c2s_to_generic(
            c2s_root=args.extraction_root,
            out_root=args.out_root,
            layers=args.layers,
            overwrite=args.overwrite_conversion,
        )
    else:
        print("[2] conversion skipped by flag")

    sae_ckpts: Dict[str, str] = {}
    for layer in args.layers:
        if not args.skip_sae:
            sae_ckpts[layer] = train_sae_for_layer(args.out_root, layer, args.device)
        else:
            sae_ckpt = os.path.join(args.out_root, "sae", layer, f"sae_{layer}.pt")
            if not os.path.exists(sae_ckpt):
                raise FileNotFoundError(f"--skip-sae was set but checkpoint not found: {sae_ckpt}")
            sae_ckpts[layer] = sae_ckpt
            print(f"[sae] using existing checkpoint for {layer}: {sae_ckpt}")

    for layer in args.layers:
        sae_ckpts[layer] = maybe_normalize_sae_checkpoint(
            out_root=args.out_root,
            layer=layer,
            sae_ckpt_path=sae_ckpts[layer],
            device=args.device,
            enabled=args.normalize_sae_features,
            max_shards=args.normalize_sae_max_shards,
            token_chunk_size=args.normalize_sae_token_chunk_size,
            feature_chunk_size=args.normalize_sae_feature_chunk_size,
        )

    for layer in args.layers:
        if args.skip_heldout:
            print(f"[heldout] skipped for {layer} by flag")
            continue
        run_heldout_for_layer(
            out_root=args.out_root,
            layer=layer,
            gt_csv=gt_csv,
            sae_ckpt_path=sae_ckpts[layer],
            adata_path=args.adata_path,
            latent_thresholds=args.latent_thresholds,
            device=args.device,
        )

    print("\nDONE: c2s-scale full pipeline complete.")


if __name__ == "__main__":
    main()
