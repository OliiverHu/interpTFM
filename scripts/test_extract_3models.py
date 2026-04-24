from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import scanpy as sc
from datasets import load_from_disk
from transformers import BertForMaskedLM

from interp_pipeline.adapters.datasets.cosmx import CosMxDatasetAdapter
from interp_pipeline.adapters.model_base import ModelSpec
from interp_pipeline.adapters.models.scgpt import ScGPTAdapter
from interp_pipeline.extraction.extractor import extract_activations
from interp_pipeline.extraction.c2s_extraction import extract_c2s_dataset
from interp_pipeline.extraction.geneformer_extraction import (
    prepare_geneformer_h5ad,
    tokenize_geneformer_dataset,
    extract_geneformer_to_store,
)
from interp_pipeline.io.activation_store import ActivationStore, ActivationStoreSpec


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def generic_activation_shards_exist(store_root: str, layer: str) -> bool:
    shard_dir = os.path.join(store_root, "activations", layer)
    return os.path.isdir(shard_dir) and any(
        name.startswith("shard_") for name in os.listdir(shard_dir)
    )


def all_generic_layers_exist(store_root: str, layers: List[str]) -> bool:
    return all(generic_activation_shards_exist(store_root, layer) for layer in layers)


def inspect_tokenized_dataset(tokenized_path: str, model_dir: str, check_rows: int = 2000) -> None:
    print("\n=== INSPECT TOKENIZED DATASET ===")
    ds = load_from_disk(tokenized_path)
    print("dataset path:", tokenized_path)
    print("num rows:", len(ds))
    print("columns:", ds.column_names)

    if len(ds) == 0:
        raise RuntimeError("Tokenized dataset is empty.")

    row0 = ds[0]
    print("first row keys:", list(row0.keys()))
    if "input_ids" not in row0:
        raise RuntimeError("Tokenized dataset missing 'input_ids'.")

    sample_n = min(len(ds), check_rows)
    mins = []
    maxs = []
    bad_rows = []
    lens = []
    non_int_rows = 0

    for i in range(sample_n):
        ids = ds[i]["input_ids"]
        if not isinstance(ids, (list, tuple)) or len(ids) == 0:
            bad_rows.append((i, "empty_or_nonlist"))
            continue
        try:
            ids = [int(x) for x in ids]
        except Exception:
            non_int_rows += 1
            bad_rows.append((i, "non_integer_tokens"))
            continue

        lens.append(len(ids))
        mins.append(min(ids))
        maxs.append(max(ids))

    print(f"checked rows: {sample_n}")
    if lens:
        print("token length min/median/max:", min(lens), sorted(lens)[len(lens)//2], max(lens))
        print("sample token min:", min(mins) if mins else None)
        print("sample token max:", max(maxs) if maxs else None)
    print("bad row count in sample:", len(bad_rows))
    if bad_rows[:10]:
        print("first bad rows:", bad_rows[:10])
    if non_int_rows:
        print("non_int_rows:", non_int_rows)

    print("\n=== INSPECT MODEL ===")
    model = BertForMaskedLM.from_pretrained(model_dir)
    vocab_size = int(model.config.vocab_size)
    print("model dir:", model_dir)
    print("vocab size:", vocab_size)
    print("hidden size:", int(model.config.hidden_size))
    print("num hidden layers:", int(model.config.num_hidden_layers))

    out_of_range = []
    negative = []
    for i in range(sample_n):
        ids = ds[i]["input_ids"]
        try:
            ids = [int(x) for x in ids]
        except Exception:
            continue

        local_bad = [t for t in ids if t >= vocab_size]
        local_neg = [t for t in ids if t < 0]
        if local_bad:
            out_of_range.append((i, local_bad[:10], max(ids)))
        if local_neg:
            negative.append((i, local_neg[:10], min(ids)))

    print("\n=== VOCAB CHECK ===")
    print("rows with out-of-range ids:", len(out_of_range))
    if out_of_range[:5]:
        print("first out-of-range rows:", out_of_range[:5])
    print("rows with negative ids:", len(negative))
    if negative[:5]:
        print("first negative-id rows:", negative[:5])

    if out_of_range or negative:
        raise ValueError(
            "Tokenized dataset contains ids incompatible with the model vocab. "
            "Fix tokenizer/model version alignment before extraction."
        )

    print("OK: sampled token ids fit model vocab.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run activation extraction for 3 models, one section at a time."
    )

    p.add_argument("--model", choices=["scgpt", "c2sscale", "geneformer"], required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--reuse-existing", action="store_true")

    # scGPT
    p.add_argument("--scgpt-adata-path", default=None)
    p.add_argument("--scgpt-ckpt", default=None)
    p.add_argument("--scgpt-out-root", default=None)
    p.add_argument("--scgpt-layers", nargs="+", default=None)
    p.add_argument("--scgpt-batch-size", type=int, default=8)
    p.add_argument("--scgpt-max-length", type=int, default=512)

    # c2s-scale
    p.add_argument("--c2s-adata-path", default=None)
    p.add_argument("--c2s-model-path", default=None)
    p.add_argument("--c2s-out-root", default=None)
    p.add_argument("--c2s-layers", nargs="+", default=None)
    p.add_argument("--c2s-shards", type=int, default=60)
    p.add_argument("--c2s-shard-key", default="shards")
    p.add_argument("--c2s-batch-size", type=int, default=4)
    p.add_argument("--c2s-max-genes", type=int, default=256)
    p.add_argument("--c2s-pooling", default="last")
    p.add_argument("--c2s-save-dtype", default="fp16")
    p.add_argument("--c2s-pool-dtype", default="fp32")
    p.add_argument("--c2s-normalize", action="store_true")
    p.add_argument("--c2s-cache-dir", default=None)

    # geneformer
    p.add_argument("--gf-adata-path", default=None)
    p.add_argument("--gf-model-dir", default=None)
    p.add_argument("--gf-out-root", default=None)
    p.add_argument("--gf-layers", nargs="+", default=None)
    p.add_argument("--gf-model-version", default="V2")
    p.add_argument("--gf-nproc", type=int, default=1)
    p.add_argument("--gf-forward-batch-size", type=int, default=8)
    p.add_argument("--gf-check-rows", type=int, default=2000)
    p.add_argument("--gf-inspect-tokenized", action="store_true")

    return p.parse_args()


def run_scgpt_extract(args: argparse.Namespace) -> None:
    if not args.scgpt_adata_path or not args.scgpt_ckpt or not args.scgpt_out_root or not args.scgpt_layers:
        raise ValueError(
            "scGPT extraction requires --scgpt-adata-path, --scgpt-ckpt, "
            "--scgpt-out-root, and --scgpt-layers."
        )

    ensure_dir(args.scgpt_out_root)

    if args.reuse_existing and all_generic_layers_exist(args.scgpt_out_root, args.scgpt_layers):
        print(f"[scgpt] generic activation shards already exist for {args.scgpt_layers}; skipping")
        return

    print("[scgpt] loading dataset")
    ds = CosMxDatasetAdapter().load(
        {"path": args.scgpt_adata_path, "obs_key_map": {}}
    )

    print("[scgpt] loading model")
    adapter = ScGPTAdapter()
    handle = adapter.load(
        ModelSpec(
            name="scgpt",
            checkpoint=args.scgpt_ckpt,
            device=args.device,
            options={},
        )
    )

    print("[scgpt] creating activation store")
    store = ActivationStore(ActivationStoreSpec(root=args.scgpt_out_root))

    print(f"[scgpt] extracting layers={args.scgpt_layers}")
    extract_activations(
        dataset=ds,
        model_handle=handle,
        model_adapter=adapter,
        layers=args.scgpt_layers,
        store=store,
        extraction_cfg={
            "batch_size": args.scgpt_batch_size,
            "max_length": args.scgpt_max_length,
            "model_name": "scgpt",
        },
    )

    print("[scgpt] extraction complete")


def run_c2s_extract(args: argparse.Namespace) -> None:
    if not args.c2s_adata_path or not args.c2s_model_path or not args.c2s_out_root or not args.c2s_layers:
        raise ValueError(
            "c2s-scale extraction requires --c2s-adata-path, --c2s-model-path, "
            "--c2s-out-root, and --c2s-layers."
        )

    ensure_dir(args.c2s_out_root)

    print("[c2s_scale] loading AnnData")
    adata = sc.read_h5ad(args.c2s_adata_path)

    print(f"[c2s_scale] extracting layers={args.c2s_layers}")
    extract_c2s_dataset(
        adata=adata,
        model_path=args.c2s_model_path,
        output_dir=args.c2s_out_root,
        layers=args.c2s_layers,
        shards=args.c2s_shards,
        shard_key=args.c2s_shard_key,
        batch_size=args.c2s_batch_size,
        max_genes=args.c2s_max_genes,
        device=args.device,
        pooling=args.c2s_pooling,
        save_dtype=args.c2s_save_dtype,
        pool_dtype=args.c2s_pool_dtype,
        normalize=args.c2s_normalize,
        cache_dir=args.c2s_cache_dir,
    )

    print("[c2s_scale] extraction complete")


def run_geneformer_extract(args: argparse.Namespace) -> None:
    if not args.gf_adata_path or not args.gf_model_dir or not args.gf_out_root or not args.gf_layers:
        raise ValueError(
            "geneformer extraction requires --gf-adata-path, --gf-model-dir, "
            "--gf-out-root, and --gf-layers."
        )

    ensure_dir(args.gf_out_root)

    if args.reuse_existing and all_generic_layers_exist(args.gf_out_root, args.gf_layers):
        print(f"[geneformer] generic activation shards already exist for {args.gf_layers}; skipping")
        return

    prepared_dir = os.path.join(args.gf_out_root, "prepared")
    tokenized_dir = os.path.join(args.gf_out_root, "tokenized")
    ensure_dir(prepared_dir)
    ensure_dir(tokenized_dir)

    prepared_path = os.path.join(prepared_dir, "prepared.h5ad")

    print("[geneformer] preparing h5ad")
    prepared = prepare_geneformer_h5ad(
        adata_path=args.gf_adata_path,
        output_path=prepared_path,
    )

    print("[geneformer] tokenizing dataset")
    tokenized = tokenize_geneformer_dataset(
        prepared_h5ad_path=prepared,
        output_dir=tokenized_dir,
        output_prefix="geneformer_tokens",
        model_version=args.gf_model_version,
        nproc=args.gf_nproc,
    )

    if args.gf_inspect_tokenized:
        inspect_tokenized_dataset(
            tokenized_path=tokenized,
            model_dir=args.gf_model_dir,
            check_rows=args.gf_check_rows,
        )

    print(f"[geneformer] extracting layers={args.gf_layers}")
    layers = extract_geneformer_to_store(
        model_dir=args.gf_model_dir,
        tokenized_dataset_path=tokenized,
        store_root=args.gf_out_root,
        layers=args.gf_layers,
        model_version=args.gf_model_version,
        device=args.device,
        forward_batch_size=args.gf_forward_batch_size,
    )

    print("[geneformer] extraction complete")
    print("[geneformer] layers:", layers)


def main() -> None:
    args = parse_args()

    print("=" * 80)
    print("test_extract_3models.py")
    print(f"section test model = {args.model}")
    print(f"device = {args.device}")
    print("=" * 80)

    if args.model == "scgpt":
        run_scgpt_extract(args)
    elif args.model == "c2sscale":
        run_c2s_extract(args)
    elif args.model == "geneformer":
        run_geneformer_extract(args)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    print("\nDONE: extraction section complete.")


if __name__ == "__main__":
    main()