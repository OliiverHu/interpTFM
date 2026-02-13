import os
import scanpy as sc

from interp_pipeline.adapters.datasets.cosmx import CosMxDatasetAdapter
from interp_pipeline.adapters.models.scgpt import ScGPTAdapter
from interp_pipeline.adapters.model_base import ModelSpec
from interp_pipeline.extraction.extractor import extract_activations
from interp_pipeline.io.activation_store import ActivationStore, ActivationStoreSpec
from interp_pipeline.types.dataset import StandardDataset

# === CONFIG ===
ADATA_PATH = "/maiziezhou_lab2/yunfei/Projects/FM_temp/InterPLM/interplm/ge_shards/cosmx_human_lung_sec8.h5ad"
SCKPT = "/maiziezhou_lab/zihang/SpatialFoundationModel/eval/scGPT/whole-human-pretrain"
OUT_DIR = "debug_acts"
LAYER = "layer_0.norm2"

# Load dataset
ds = CosMxDatasetAdapter().load({"path": ADATA_PATH, "obs_key_map": {}})

# Load model
adapter = ScGPTAdapter()
handle = adapter.load(ModelSpec(name="scgpt", checkpoint=SCKPT, device="cuda", options={}))

# Activation store
store = ActivationStore(ActivationStoreSpec(root=OUT_DIR))

# Extract (limit to small batch size)
_, layers = extract_activations(
    dataset=ds,
    model_handle=handle,
    model_adapter=adapter,
    layers=[LAYER],
    store=store,
    extraction_cfg={
        "batch_size": 8,      # small
        "max_length": 512,    # reduce memory
        "model_name": "scgpt",
        "start_shard": 0,
    },
)

print("Extracted layers:", layers)

# Inspect output files
import glob
print("Shards found:", glob.glob(f"{OUT_DIR}/activations/{LAYER}/shard_*"))

# Load one shard to check shapes
import torch
shard_path = glob.glob(f"{OUT_DIR}/activations/{LAYER}/shard_*")[0]
acts = torch.load(os.path.join(shard_path, "activations.pt"))
index = torch.load(os.path.join(shard_path, "index.pt"))

print("Activation shape:", acts.shape)
print("Index lengths:", len(index["example_ids"]), len(index["token_ids"]))
