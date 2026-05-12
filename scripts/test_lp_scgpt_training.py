"""
Train linear probes on pre-extracted scGPT activations.

Assumes activations have already been written to an ActivationStore by
test_lp_scgpt_acts.py.  This script:

  1. Loads the scGPT tokenizer vocab (no model weights needed).
  2. Reads the binary concept matrix CSV produced by test_lp_scgpt_acts.py.
  3. Discovers all layers in the store and trains one probe per layer.
  4. Saves probes to PROBE_OUT_DIR/probe_{layer}.pt.
"""
# %%
import os

import pandas as pd
from dotenv import load_dotenv

from interp_pipeline.scgpt_local.tokenizer import Tokenizer
from interp_pipeline.io.activation_store import ActivationStore, ActivationStoreSpec
from interp_pipeline.linear_probe import (
    build_id_to_gene,
    LinearProbeSpec,
    train_probe_for_layer,
)

# ── Config ────────────────────────────────────────────────────────────────────

# Path to the scGPT checkpoint directory (only vocab.json is read).
SCGPT_CKPT = "/maiziezhou_lab/zihang/SpatialFoundationModel/eval/scGPT/whole-human-pretrain"

# Root of the ActivationStore produced by test_lp_scgpt_acts.py.
STORE_ROOT = "runs/lp_scgpt_norman"

# Binary concept matrix CSV produced by test_lp_scgpt_acts.py.
# Rows = GO terms, columns = gene symbols, values = 0/1.
CONCEPT_CSV = "runs/lp_scgpt_norman/gprofiler/concept_matrix.csv"

# Where trained probes will be saved.
PROBE_OUT_DIR = "runs/lp_scgpt_norman/linear_probes"

# Device for probe training (probes are small — CPU is fine, cuda is faster).
DEVICE = "cuda"

# scGPT hidden dimension.
HIDDEN_SIZE = 512

# Which layers to train on. Set to None to train all layers found in the store.
LAYERS = ["layer_4"]  # e.g. None to train all

# W&B logging.
USE_WANDB     = True
WANDB_PROJECT = "linear-probe-norman"
WANDB_ENTITY  = "yunfei-hu-vanderbilt-university"
# Run name per layer will be "probe-{layer}" unless overridden here.
WANDB_NAME    = None   # e.g. "trial-L4" to override

# LinearProbe training hyperparameters.
PROBE_SPEC_KWARGS = dict(
    epochs=10,
    batch_size=8192,
    lr=1e-4,
    betas=(0.9, 0.99),
    weight_decay=0.1,
    test_fraction=0.2,
    seed=42,
)

# ── W&B login ─────────────────────────────────────────────────────────────────
# %%
if USE_WANDB:
    import wandb
    load_dotenv()
    assert (wandb_api_key := os.getenv("WANDB_API_KEY")), "WANDB_API_KEY not set in environment or .env"
    wandb.login(key=wandb_api_key)

# ── 1. Tokenizer vocab → id_to_gene ──────────────────────────────────────────
# %%
tokenizer = Tokenizer(os.path.join(SCGPT_CKPT, "vocab.json"), device="cpu")
id_to_gene = build_id_to_gene(tokenizer.vocab)
print(f"Vocab size: {len(tokenizer.vocab):,} genes")

# ── 2. Concept matrix ─────────────────────────────────────────────────────────
# %%
# concept_df: [n_terms, n_genes] binary DataFrame produced by test_lp_scgpt_acts.py.
concept_df = pd.read_csv(CONCEPT_CSV, index_col=0)
n_concepts = concept_df.shape[0]
print(f"Concept matrix: {n_concepts:,} terms × {concept_df.shape[1]:,} genes")

# ── 3. Discover layers ────────────────────────────────────────────────────────
# %%
store = ActivationStore(ActivationStoreSpec(root=STORE_ROOT))

acts_dir = os.path.join(STORE_ROOT, "activations")
available_layers = sorted(
    name for name in os.listdir(acts_dir)
    if os.path.isdir(os.path.join(acts_dir, name))
)
print(f"Layers in store: {available_layers}")

target_layers = LAYERS if LAYERS is not None else available_layers
print(f"Training probes for: {target_layers}")

# ── 4. Train one probe per layer ──────────────────────────────────────────────
# %%
os.makedirs(PROBE_OUT_DIR, exist_ok=True)

for layer in target_layers:
    n_shards = len(store.list_shards(layer))
    print(f"\n── {layer}  ({n_shards} shards) ──")

    spec = LinearProbeSpec(
        n_concepts=n_concepts,
        hidden_size=HIDDEN_SIZE,
        **PROBE_SPEC_KWARGS,
    )

    probe = train_probe_for_layer(
        store=store,
        layer=layer,
        concept_matrix=concept_df,
        id_to_gene=id_to_gene,
        spec=spec,
        output_dir=PROBE_OUT_DIR,
        device=DEVICE,
        use_wandb=USE_WANDB,
        wandb_project=WANDB_PROJECT,
        wandb_entity=WANDB_ENTITY,
        wandb_name=WANDB_NAME,
    )

    print(f"Saved probe for {layer} → {PROBE_OUT_DIR}/probe_{layer}.pt")

print("\nAll probes trained.")