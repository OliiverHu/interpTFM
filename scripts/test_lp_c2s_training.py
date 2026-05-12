"""
Train linear probes on pre-extracted C2S (Pythia-410m) activations.

Assumes activations have already been written to an ActivationStore by
test_lp_c2s_acts.py.  This script:

  1. Builds an identity id_to_gene map from the concept matrix columns.
     Unlike scGPT (which stores integer token IDs), C2S stores gene name
     strings directly as token_ids in the ActivationStore — so the mapping
     is gene_name → gene_name.
  2. Reads the binary concept matrix CSV produced by test_lp_c2s_acts.py.
  3. Discovers all layers in the store and trains one probe per layer.
  4. Saves probes to PROBE_OUT_DIR/probe_{layer}.pt.
"""
# %%
import os

import pandas as pd
from dotenv import load_dotenv

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_PROJECT_ROOT)

from interp_pipeline.io.activation_store import ActivationStore, ActivationStoreSpec
from interp_pipeline.linear_probe import (
    LinearProbeSpec,
    train_probe_for_layer,
)

# ── Config ────────────────────────────────────────────────────────────────────

# Root of the ActivationStore produced by test_lp_c2s_acts.py.
STORE_ROOT = "runs/lp_c2s_norman"

# Binary concept matrix CSV produced by test_lp_c2s_acts.py.
# Rows = GO terms, columns = gene symbols, values = 0/1.
CONCEPT_CSV = "runs/lp_c2s_norman/gprofiler/concept_matrix.csv"

# Where trained probes will be saved.
PROBE_OUT_DIR = "runs/lp_c2s_norman/linear_probes"

DEVICE = "cuda"

# C2S Pythia-410m hidden dimension.
HIDDEN_SIZE = 1024

# Which layers to train on.  Set to None to train all layers found in the store.
LAYERS = ["layer_12"]

# W&B logging.
USE_WANDB     = True
WANDB_PROJECT = "linear-probe-norman-c2s"
WANDB_ENTITY  = "yunfei-hu-vanderbilt-university"
WANDB_NAME    = None

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

# ── 1. Build identity id_to_gene ──────────────────────────────────────────────
# %%
# C2S stores gene names directly as token_ids in the ActivationStore (not
# integer IDs like scGPT).  ConceptFilteredDataset calls id_to_gene.get(str(tid))
# to resolve each token; the identity map makes that work transparently.
concept_df = pd.read_csv(CONCEPT_CSV, index_col=0)
n_concepts = concept_df.shape[0]
print(f"Concept matrix: {n_concepts:,} terms × {concept_df.shape[1]:,} genes")

id_to_gene = {g: g for g in concept_df.columns}
print(f"id_to_gene: identity map over {len(id_to_gene):,} genes")

# ── 2. Discover layers ────────────────────────────────────────────────────────
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

# ── 3. Train one probe per layer ──────────────────────────────────────────────
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
