from __future__ import annotations

import os
from pathlib import Path

from datasets import load_from_disk
from transformers import BertForMaskedLM

# python run_geneformer_full.py \
#   --layers layer_4 \
#   --model-dir /maiziezhou_lab2/yunfei/geneformer_hf \
#   --adata-path /maiziezhou_lab2/yunfei/Projects/FM_temp/interGFM/ge_shards/cosmx_human_lung_sec8.h5ad \
#   --out-root runs/full_geneformer_cosmx \
#   --device cuda \
#   --skip-heldout

from interp_pipeline.extraction.geneformer_extraction import (
    prepare_geneformer_h5ad,
    tokenize_geneformer_dataset,
    extract_geneformer_to_store,
)

ADATA_PATH = "/maiziezhou_lab2/yunfei/Projects/FM_temp/interGFM/ge_shards/cosmx_human_lung_sec8.h5ad"
MODEL_DIR = "/maiziezhou_lab2/yunfei/geneformer_hf"
OUT_ROOT = "runs/test_geneformer"
LAYERS = ["layer_0"]

MODEL_VERSION = "V2"
NPROC = 1
FORWARD_BATCH_SIZE = 8
DEVICE = os.environ.get("GENEFORMER_DEVICE", "cuda")
CHECK_ROWS = int(os.environ.get("GENEFORMER_CHECK_ROWS", "2000"))


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


prepared = prepare_geneformer_h5ad(
    adata_path=ADATA_PATH,
    output_path=os.path.join(OUT_ROOT, "prepared", "cosmx_human_lung_sec8.prepared.h5ad"),
)

tokenized = tokenize_geneformer_dataset(
    prepared_h5ad_path=prepared,
    output_dir=os.path.join(OUT_ROOT, "tokenized"),
    output_prefix="cosmx_human_lung_sec8_v1",
    model_version=MODEL_VERSION,
    nproc=NPROC,
)

inspect_tokenized_dataset(
    tokenized_path=tokenized,
    model_dir=MODEL_DIR,
    check_rows=CHECK_ROWS,
)

print("\n=== START EXTRACTION ===")
print("device:", DEVICE)
print("forward_batch_size:", FORWARD_BATCH_SIZE)

layers = extract_geneformer_to_store(
    model_dir=MODEL_DIR,
    tokenized_dataset_path=tokenized,
    store_root=OUT_ROOT,
    layers=LAYERS,
    model_version=MODEL_VERSION,
    device=DEVICE,
    forward_batch_size=FORWARD_BATCH_SIZE,
)

print("\ndone")
print("prepared:", prepared)
print("tokenized:", tokenized)
print("layers:", layers)
print("activation dir:", Path(OUT_ROOT) / "activations" / "layer_0")
