# %% [markdown]
# # C2S-Scale Model Testing
#
# Tests: model loading, adata preprocessing, cell sentence construction,
# prompt formatting (cell generation), tokenization, and activation extraction.

# %%
import os
import glob
import torch
import numpy as np
import anndata as ad
from scipy import sparse

# %% [markdown]
# ## 1. Load Model

# %%
from interp_pipeline.adapters.models.c2s import C2SScaleAdapter
from interp_pipeline.adapters.model_base import ModelSpec

MODEL_NAME = "vandijklab/C2S-Pythia-410m-cell-type-conditioned-cell-generation"
CACHE_DIR = "/maiziezhou_lab2/zihang/interpTFM/cache/c2s"

spec = ModelSpec(
    name="c2s-scale",
    checkpoint=MODEL_NAME,
    device="cuda",
    options={"cache_dir": CACHE_DIR}
)

# MAX_GENES = 256
# model_path = "/maiziezhou_lab2/yunfei/Projects/interpTFM-legacy/c2sscale/models/C2S-Scale-Gemma-2-2B"
# spec = ModelSpec(
#     name="c2s-scale",
#     checkpoint=model_path,
#     device="cuda:0",
#     options={
#         "max_genes": MAX_GENES,
#         "cache_dir": None,
#     },
# )


adapter = C2SScaleAdapter()
handle = adapter.load(spec)

layers = adapter.list_layers(handle)
print("Loaded C2S-Scale")
print("#layers:", len(layers))
print("First 5:", layers[:5])

# %% [markdown]
# ## 2. Synthetic AnnData Setup

# %%
# Build a small synthetic AnnData to test preprocessing without real data.
# gene names are human-readable strings (not Ensembl IDs).
from interp_pipeline.adapters.datasets.cosmx import CosMxDatasetAdapter

# ADATA_PATH = "/maiziezhou_lab2/zihang/interpTFM-data/c2s/dominguez_conde_immune_tissue_two_donors.h5ad"
ADATA_PATH  = "/maiziezhou_lab2/yunfei/Projects/FM_temp/InterPLM/interplm/ge_shards/cosmx_human_lung_sec8.h5ad"

ds = CosMxDatasetAdapter().load({"path": ADATA_PATH, "obs_key_map": {}})
adata = ds.adata[:1000]
print("Dataset loaded")
print("n_obs:", adata.n_obs)
print("n_vars:", adata.n_vars)
print("obs columns:", list(adata.obs.columns))

# adata.obs["cell_type"] = adata.obs["author_cell_type"]
# adata.obs["cell_name"] = adata.obs_names
# adata.var_names = adata.var["feature_name"].astype(str).values

# %% [markdown]
# ## 3. Utility: generate_vocabulary & generate_sentences

# %%
from interp_pipeline.c2s_local.util import generate_vocabulary, generate_sentences

vocab = generate_vocabulary(adata)
print("Vocabulary size:", len(vocab))
print("Top-5 genes by non-zero cell count:")
for gene, count in list(vocab.items())[:5]:
    print(f"  {gene}: {count} cells")

# %%
sentences = generate_sentences(adata, vocab)
print("Generated", len(sentences), "cell sentences")
for i, sent in enumerate(sentences[:3]):
    genes = sent.split()
    print(f"  Cell{i}: {len(genes)} genes expressed | first 5: {genes[:5]}")

# %% [markdown]
# ## 4. C2SProcessor: adata_to_arrow

# %%
from interp_pipeline.c2s_local.processor import C2SProcessor

processor = handle.processor
processor.normalize_adata(adata)  # in-place normalization
arrow_ds, vocab_built = processor.adata_to_arrow(
    adata,
    sentence_delimiter=" ",
    label_col_names=["cell_type", "organism"],
)
prompts = processor.prompts_generation(arrow_ds, task="cell_type_prediction", n_genes=200)

# %% [markdown]
# ## 5. make_batches, forward_and_capture, process_captured

# %%
from interp_pipeline.types.dataset import StandardDataset

BATCH_SIZE   = 4
MAX_LENGTH   = 200   # n_genes per cell sentence
target_layers = ["layer_0", "layer_23"]   # first and last block

dataset = StandardDataset(adata=adata, obs_key_map={})

# ── 5a. make_batches ──────────────────────────────────────────────────────────
batch = None
for batch in adapter.make_batches(dataset, handle, BATCH_SIZE, MAX_LENGTH):
    break   # grab first batch only

print("batch keys:", list(batch.keys()))
print("cell_ids:", batch["cell_ids"][:3])
print("input_ids shape:", batch["tokenized"]["input_ids"].shape)     # [B, T]
print("attention_mask shape:", batch["tokenized"]["attention_mask"].shape)  # [B, T]
# attention_mask: 1 = real token, 0 = padding

# ── 5b. forward_and_capture ───────────────────────────────────────────────────
# %%
captured = adapter.forward_and_capture(
    model_handle=handle,
    batch=batch,
    layers=target_layers,
    capture_cfg={},
)

print("\ncaptured layers:", list(captured.keys()))
for lname, acts in captured.items():
    print(f"  {lname}: {acts.shape}")   # expect [B, T, H] or nested

# ── 5c. process_captured ──────────────────────────────────────────────────────
# %%
buf_entries = adapter.process_captured(captured, batch)

print("\nbuf_entries keys per layer:", {k: list(v.keys()) for k, v in buf_entries.items()})
for lname, entry in buf_entries.items():
    print(f"\n[{lname}]")
    print(f"  acts:       {entry['acts'].shape}")   # [N, H]
    print(f"  tok[:5]:    {entry['tok'][:5]}")
    print(f"  ex[:5]:     {entry['ex'][:5]}")
    print(f"  token_unit: {entry['token_unit']}")
    N = entry["acts"].shape[0]
    assert len(entry["tok"]) == N, f"tok length mismatch: {len(entry['tok'])} != {N}"
    assert len(entry["ex"])  == N, f"ex length mismatch:  {len(entry['ex'])}  != {N}"

print("\nAll process_captured checks passed.")