#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path

from interp_pipeline.extraction.geneformer_extraction import (
    extract_geneformer_to_store,
    prepare_geneformer_h5ad,
    tokenize_geneformer_dataset,
)

DEFAULT_ADATA = "/maiziezhou_lab2/yunfei/Projects/FM_temp/interGFM/ge_shards/cosmx_human_lung_sec8.h5ad"
DEFAULT_MODEL_DIR = "/maiziezhou_lab2/yunfei/geneformer_hf"
DEFAULT_OUT = "/maiziezhou_lab2/yunfei/Projects/interpTFM/acts_extraction_geneformer_cosmx"
DEFAULT_LAYERS = ["layer_1", "layer_4", "layer_7", "layer_10", "layer_12", "layer_14", "layer_16", "layer_17"]


def activation_shards_exist(out_root: str, layer: str) -> bool:
    layer_dir = Path(out_root) / "activations" / layer
    if not layer_dir.is_dir():
        return False
    return any((p / "activations.pt").exists() and (p / "index.pt").exists() for p in layer_dir.glob("shard_*"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract Geneformer activations only; no SAE, no heldout, no TIS.")
    p.add_argument("--adata-path", default=DEFAULT_ADATA)
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    p.add_argument("--out-root", default=DEFAULT_OUT)
    p.add_argument("--layers", nargs="+", default=DEFAULT_LAYERS)
    p.add_argument("--device", default="cuda")
    p.add_argument("--model-version", default="V2")
    p.add_argument("--forward-batch-size", type=int, default=8)
    p.add_argument("--tokenizer-nproc", type=int, default=1)
    p.add_argument("--skip-prepare", action="store_true")
    p.add_argument("--skip-tokenize", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    tokenized_prefix = f"cosmx_{args.model_version.lower()}"
    prepared_path = out_root / "prepared" / "cosmx.prepared.h5ad"
    tokenized_path = out_root / "tokenized" / f"{tokenized_prefix}.dataset"

    print("=" * 80)
    print("extract_geneformer.py")
    print(f"ADATA_PATH     = {args.adata_path}")
    print(f"MODEL_DIR      = {args.model_dir}")
    print(f"OUT_ROOT       = {args.out_root}")
    print(f"MODEL_VERSION  = {args.model_version}")
    print(f"LAYERS         = {args.layers}")
    print(f"TOKENIZED_PATH = {tokenized_path}")
    print("=" * 80)

    missing = [layer for layer in args.layers if args.force or not activation_shards_exist(args.out_root, layer)]
    if not missing:
        print("[extract] all requested layers already have generic activation shards; nothing to do.")
        return

    if args.dry_run:
        print(f"[dry-run] would prepare/tokenize if needed and extract missing layers: {missing}")
        return

    if args.skip_prepare:
        print("[prepare] skipped")
        prepared = str(prepared_path)
    else:
        prepared = prepare_geneformer_h5ad(
            adata_path=args.adata_path,
            output_path=str(prepared_path),
        )

    if args.skip_tokenize:
        print("[tokenize] skipped")
        tokenized = str(tokenized_path)
    else:
        tokenized = tokenize_geneformer_dataset(
            prepared_h5ad_path=prepared,
            output_dir=str(out_root / "tokenized"),
            output_prefix=tokenized_prefix,
            model_version=args.model_version,
            nproc=args.tokenizer_nproc,
        )

    if not os.path.exists(tokenized):
        raise FileNotFoundError(f"Tokenized dataset not found: {tokenized}")

    print(f"[extract] missing layers: {missing}")
    extract_geneformer_to_store(
        model_dir=args.model_dir,
        tokenized_dataset_path=tokenized,
        store_root=args.out_root,
        layers=missing,
        model_version=args.model_version,
        device=args.device,
        forward_batch_size=args.forward_batch_size,
    )

    print("DONE")


if __name__ == "__main__":
    main()
