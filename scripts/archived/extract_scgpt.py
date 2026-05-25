#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List

from interp_pipeline.adapters.datasets.cosmx import CosMxDatasetAdapter
from interp_pipeline.adapters.model_base import ModelSpec
from interp_pipeline.adapters.models.scgpt import ScGPTAdapter
from interp_pipeline.extraction.extractor import extract_activations
from interp_pipeline.io.activation_store import ActivationStore, ActivationStoreSpec

DEFAULT_ADATA = "/maiziezhou_lab2/yunfei/Projects/FM_temp/interGFM/ge_shards/cosmx_human_lung_sec8.h5ad"
DEFAULT_CKPT = "/maiziezhou_lab/zihang/SpatialFoundationModel/eval/scGPT/whole-human-pretrain"
DEFAULT_OUT = "/maiziezhou_lab2/yunfei/Projects/interpTFM/acts_extraction_scgpt_cosmx"
DEFAULT_LAYERS = [f"layer_{i}.norm2" for i in range(12)]


def activation_shards_exist(out_root: str, layer: str) -> bool:
    layer_dir = Path(out_root) / "activations" / layer
    if not layer_dir.is_dir():
        return False
    return any((p / "activations.pt").exists() and (p / "index.pt").exists() for p in layer_dir.glob("shard_*"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract scGPT activations only; no SAE, no heldout, no TIS.")
    p.add_argument("--adata-path", default=DEFAULT_ADATA)
    p.add_argument("--checkpoint", default=DEFAULT_CKPT)
    p.add_argument("--out-root", default=DEFAULT_OUT)
    p.add_argument("--device", default="cuda")
    p.add_argument("--layers", nargs="+", default=DEFAULT_LAYERS)
    p.add_argument("--list-model-layers", action="store_true", help="Load model and print adapter.list_layers(handle), then exit.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--force", action="store_true", help="Re-extract even if generic activation shards already exist.")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.out_root).mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("extract_scgpt.py")
    print(f"ADATA_PATH = {args.adata_path}")
    print(f"CHECKPOINT = {args.checkpoint}")
    print(f"OUT_ROOT   = {args.out_root}")
    print(f"LAYERS     = {args.layers}")
    print("=" * 80)

    adapter = ScGPTAdapter()
    handle = adapter.load(ModelSpec(name="scgpt", checkpoint=args.checkpoint, device=args.device, options={}))

    if args.list_model_layers:
        print("Model layers:")
        for layer in adapter.list_layers(handle):
            print(layer)
        return

    missing = [layer for layer in args.layers if args.force or not activation_shards_exist(args.out_root, layer)]
    if not missing:
        print("[extract] all requested layers already have generic activation shards; nothing to do.")
        return

    print(f"[extract] layers to extract: {missing}")
    if args.dry_run:
        return

    ds = CosMxDatasetAdapter().load({"path": args.adata_path, "obs_key_map": {}})
    store = ActivationStore(ActivationStoreSpec(root=args.out_root))
    extraction_cfg: Dict[str, Any] = {
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "model_name": "scgpt",
        "max_shards": None,
    }

    for layer in missing:
        print("-" * 80)
        print(f"[extract] scGPT {layer}")
        extract_activations(
            dataset=ds,
            model_handle=handle,
            model_adapter=adapter,
            layers=[layer],
            store=store,
            extraction_cfg=extraction_cfg,
        )

    print("DONE")


if __name__ == "__main__":
    main()
