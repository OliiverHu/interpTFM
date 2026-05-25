#!/usr/bin/env python
from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import scanpy as sc
import scipy.sparse as sp
import torch
from tqdm import tqdm

from interp_pipeline.extraction.c2s_extraction import extract_c2s_dataset
from interp_pipeline.io.activation_store import ActivationStore, ActivationStoreSpec

DEFAULT_ADATA = "/maiziezhou_lab2/yunfei/Projects/FM_temp/interGFM/ge_shards/cosmx_human_lung_sec8.h5ad"
DEFAULT_SOURCE_ADATA = "/maiziezhou_lab2/yunfei/Projects/FM_temp/datasets/cosmx/lung/cosmx_human_lung.h5ad"
DEFAULT_MODEL_PATH = "/maiziezhou_lab2/yunfei/Projects/interpTFM-legacy/c2sscale/models/C2S-Scale-Gemma-2-2B"
DEFAULT_RAW_ROOT = "/maiziezhou_lab2/yunfei/Projects/interpTFM/acts_extraction_c2s_cosmx"
DEFAULT_STORE_ROOT = "/maiziezhou_lab2/yunfei/Projects/interpTFM/acts_extraction_c2s_cosmx_store"
DEFAULT_LAYERS = ["layer_0", "layer_6", "layer_13", "layer_15", "layer_17", "layer_19", "layer_21", "layer_23", "layer_25"]


@dataclass(frozen=True)
class ExtractionConfig:
    shards: int = 60
    shard_key: str = "shards"
    batch_size: int = 4
    max_genes: int = 256
    device: str = "cuda:0"
    pooling: str = "last"
    save_dtype: str = "fp16"
    pool_dtype: str = "fp32"
    normalize: bool = False
    cache_dir: Optional[str] = None


def raw_gene_batches_exist(raw_root: str, layer: str) -> bool:
    pat = os.path.join(raw_root, "activations", layer, "shard_*", "batch_*_gene_acts.pt")
    return len(glob.glob(pat)) > 0


def generic_activation_shards_exist(store_root: str, layer: str) -> bool:
    pat = os.path.join(store_root, "activations", layer, "shard_*", "activations.pt")
    return len(glob.glob(pat)) > 0


def load_or_prepare_sec8_h5ad(shard_path: str, source_path: str, n_shards: int = 60, seed: int = 0):
    p = Path(shard_path)
    if p.exists():
        print(f"[adata] using existing h5ad: {p}")
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


def convert_c2s_layer_to_activation_store(raw_root: str, store_root: str, layer: str, overwrite: bool = False) -> int:
    layer_dir = os.path.join(raw_root, "activations", layer)
    if not os.path.isdir(layer_dir):
        raise FileNotFoundError(f"Missing raw C2S layer directory: {layer_dir}")

    store = ActivationStore(ActivationStoreSpec(root=store_root))
    shard_dirs = sorted(p for p in glob.glob(os.path.join(layer_dir, "shard_*")) if os.path.isdir(p))
    if not shard_dirs:
        raise RuntimeError(f"No C2S shard dirs found for {layer} under {layer_dir}")

    written = 0
    for shard_dir in tqdm(shard_dirs, desc=f"convert:{layer}"):
        shard_name = os.path.basename(shard_dir)
        shard_id = int(shard_name.split("_")[-1])
        generic_dir = os.path.join(store_root, "activations", layer, shard_name)
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

        for act_path in act_files:
            stem = os.path.basename(act_path).replace("_gene_acts.pt", "")
            pair_path = os.path.join(shard_dir, f"{stem}_cell_gene_pairs.txt")
            if not os.path.exists(pair_path):
                raise FileNotFoundError(f"Missing pair file for {act_path}: {pair_path}")

            acts = torch.load(act_path, map_location="cpu")
            if not isinstance(acts, torch.Tensor) or acts.ndim != 2:
                raise RuntimeError(f"Expected 2D tensor in {act_path}, got {type(acts)} ndim={getattr(acts, 'ndim', None)}")
            ex_ids, tok_ids = _read_pairs_file(pair_path)
            if len(ex_ids) != acts.shape[0]:
                raise RuntimeError(f"Row mismatch in {act_path}: acts_rows={acts.shape[0]} pairs={len(ex_ids)}")

            acts_parts.append(acts)
            example_ids.extend(ex_ids)
            token_ids.extend(tok_ids)

        if not acts_parts:
            continue

        store.write_token_shard(
            layer=layer,
            shard_id=shard_id,
            acts=torch.cat(acts_parts, dim=0),
            example_ids=example_ids,
            token_ids=token_ids,
            token_unit="gene",
            meta={"source_format": "c2s_gene_batches", "source_layer": layer, "source_root": raw_root},
        )
        written += 1
    return written


def maybe_extract(raw_root: str, adata, model_path: str, layers: Sequence[str], cfg: ExtractionConfig, force: bool = False) -> None:
    missing = [layer for layer in layers if force or not raw_gene_batches_exist(raw_root, layer)]
    if not missing:
        print("[extract] all requested layers already have raw C2S gene-batch files; skipping extraction")
        return
    print(f"[extract] missing/raw-overwrite layers: {missing}")
    extract_c2s_dataset(
        adata=adata,
        model_path=model_path,
        output_dir=raw_root,
        layers=list(missing),
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract C2S-scale raw gene activations and convert to generic ActivationStore.")
    p.add_argument("--adata-path", default=DEFAULT_ADATA)
    p.add_argument("--adata-source-path", default=DEFAULT_SOURCE_ADATA)
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--raw-root", default=DEFAULT_RAW_ROOT)
    p.add_argument("--store-root", default=DEFAULT_STORE_ROOT)
    p.add_argument("--layers", nargs="+", default=DEFAULT_LAYERS)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--skip-extract", action="store_true")
    p.add_argument("--skip-convert", action="store_true")
    p.add_argument("--force-extract", action="store_true")
    p.add_argument("--overwrite-convert", action="store_true")
    p.add_argument("--extract-shards", type=int, default=60)
    p.add_argument("--extract-batch-size", type=int, default=4)
    p.add_argument("--extract-max-genes", type=int, default=256)
    p.add_argument("--extract-pooling", default="last")
    p.add_argument("--extract-save-dtype", default="fp16")
    p.add_argument("--extract-pool-dtype", default="fp32")
    p.add_argument("--extract-normalize", action="store_true")
    p.add_argument("--extract-cache-dir", default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.raw_root).mkdir(parents=True, exist_ok=True)
    Path(args.store_root).mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("extract_c2sscale.py")
    print(f"ADATA_PATH = {args.adata_path}")
    print(f"MODEL_PATH = {args.model_path}")
    print(f"RAW_ROOT   = {args.raw_root}")
    print(f"STORE_ROOT = {args.store_root}")
    print(f"LAYERS     = {args.layers}")
    print("=" * 80)

    if args.dry_run:
        print("[dry-run] no extraction/conversion executed")
        return

    if not args.skip_extract:
        adata = load_or_prepare_sec8_h5ad(args.adata_path, args.adata_source_path, n_shards=args.extract_shards)
        maybe_extract(
            raw_root=args.raw_root,
            adata=adata,
            model_path=args.model_path,
            layers=args.layers,
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
            force=args.force_extract,
        )
    else:
        print("[extract] skipped")

    if not args.skip_convert:
        for layer in args.layers:
            if generic_activation_shards_exist(args.store_root, layer) and not args.overwrite_convert:
                print(f"[convert] {layer}: generic shards already exist; skipping")
                continue
            n = convert_c2s_layer_to_activation_store(args.raw_root, args.store_root, layer, overwrite=args.overwrite_convert)
            print(f"[convert] {layer}: wrote/updated {n} shards")
    else:
        print("[convert] skipped")

    print("DONE")


if __name__ == "__main__":
    main()
