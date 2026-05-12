"""
Extract scGPT token activations on the Norman perturbation dataset and build
a binary GO concept matrix via g:Profiler.

No SAE training.  Output is consumed by test_lp_training.py.

Steps:
  1. Load Norman AnnData; map var["gene_name"] → var["feature_name"] so the
     scGPT adapter uses gene symbols (its tokenizer vocab key format).
  2. Call g:Profiler for all genes in the panel; build a binary concept matrix
     [n_terms, n_genes] and save it as concept_matrix.csv.
  3. Load scGPT and extract activations for all N_LAYERS simultaneously
     (one forward pass per batch — 12× cheaper than the per-layer loop in
     run_scgpt_full_12layers.py).
  4. Print a handoff summary for test_lp_training.py.
"""
# %%
import os
import glob
from typing import Any, List

import numpy as np
import pandas as pd
import scanpy as sc

from interp_pipeline.adapters.models.scgpt import ScGPTAdapter
from interp_pipeline.adapters.model_base import ModelSpec
from interp_pipeline.extraction.extractor import extract_activations
from interp_pipeline.get_annotation.gprofiler_client import GProfilerClient, GProfilerSpec
from interp_pipeline.io.activation_store import ActivationStore, ActivationStoreSpec
from interp_pipeline.scgpt_local.tokenizer import Tokenizer
from interp_pipeline.types.dataset import StandardDataset

# ── Config ────────────────────────────────────────────────────────────────────

ADATA_PATH = "/maiziezhou_lab2/zihang/interpTFM/temp/norman/perturb_processed.h5ad"
SCGPT_CKPT = "/maiziezhou_lab/zihang/SpatialFoundationModel/eval/scGPT/whole-human-pretrain"
OUT_ROOT   = "runs/lp_scgpt_norman"
DEVICE     = "cuda"

# Layers to extract. None = all layers from the model.
LAYERS   = ["layer_4"]   # e.g. ["layer_4"] or None for all
BATCH_SIZE  = 8
MAX_LENGTH  = 512

# g:Profiler
GPROF_SOURCES = ["GO:BP", "GO:MF", "GO:CC"]
GPROF_ALPHA   = 0.05

# Skip steps if outputs already exist on disk.
REUSE_CONCEPT = True
REUSE_ACTS    = True

GPROF_DIR    = os.path.join(OUT_ROOT, "gprofiler")
CONCEPT_CSV  = os.path.join(GPROF_DIR, "concept_matrix.csv")

EXTRACT_CFG = {
    "batch_size": BATCH_SIZE,
    "max_length": MAX_LENGTH,
    "model_name": "scgpt",
    "max_shards": None,          # auto-compute from dataset size
    # "target_tokens_per_shard": 50_000,  # override if you want fixed shard size
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _as_list(x: Any) -> List[str]:
    """Parse a g:Profiler intersections cell to a list of gene symbols."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, list):
        return [str(v["name"] if isinstance(v, dict) else v) for v in x]
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("["):
            import ast
            try:
                return [str(v) for v in ast.literal_eval(s)]
            except Exception:
                pass
        return [p.strip() for p in s.split(",") if p.strip()]
    return []


def _shards_exist(store_root: str, layer: str) -> bool:
    pat = os.path.join(store_root, "activations", layer, "shard_*", "activations.pt")
    return len(glob.glob(pat)) > 0


# ── 1. Load AnnData ───────────────────────────────────────────────────────────
# %%
print("[1] Loading Norman AnnData...")
adata = sc.read_h5ad(ADATA_PATH)
print(f"    {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")
print(f"    obs cols: {list(adata.obs.columns)}")

# scGPT adapter reads var["feature_name"] for gene symbols; Norman stores
# gene symbols in var["gene_name"] with Ensembl IDs as var_names.
adata.var["feature_name"] = adata.var["gene_name"]

# Filter to genes present in the scGPT tokenizer vocab — genes outside the
# vocab get OOB embedding indices that crash the CUDA gather kernel.
tokenizer = Tokenizer(os.path.join(SCGPT_CKPT, "vocab.json"), device="cpu")
vocab_genes = set(tokenizer.vocab.keys())
keep = adata.var["gene_name"].isin(vocab_genes)
adata = adata[:, keep].copy()
gene_symbols = list(adata.var["gene_name"])
print(f"    After vocab filter: {len(gene_symbols):,} genes (dropped {(~keep).sum():,} unknown)")
print(f"    Gene symbols sample: {gene_symbols[:5]}")

dataset = StandardDataset(adata=adata, obs_key_map={})
dataset.validate()

# ── 2. Build GO concept matrix ────────────────────────────────────────────────
# %%
os.makedirs(GPROF_DIR, exist_ok=True)

if REUSE_CONCEPT and os.path.exists(CONCEPT_CSV):
    print(f"[2] Reusing concept matrix: {CONCEPT_CSV}")
    concept_df = pd.read_csv(CONCEPT_CSV, index_col=0)
else:
    print(f"[2] Querying g:Profiler for {len(gene_symbols):,} genes...")
    gp   = GProfilerClient(cache_dir=os.path.join(GPROF_DIR, "cache"))
    spec = GProfilerSpec(
        organism="hsapiens",
        sources=GPROF_SOURCES,
        user_threshold=GPROF_ALPHA,
        significance_threshold_method="fdr",
        return_dataframe=True,
    )
    res = gp.profile(gene_symbols, spec=spec, query_name="norman_panel", force=False)
    if not isinstance(res, pd.DataFrame):
        res = pd.DataFrame(res)

    # Filter to significant terms.
    qcol = next(
        (c for c in ["p_value", "p_value_fdr", "p_value_adj", "p_value_corrected"]
         if c in res.columns),
        None,
    )
    if qcol:
        res = res[res[qcol] <= GPROF_ALPHA].copy()
    print(f"    {len(res):,} significant terms before deduplication")

    # Build binary matrix: rows = terms, cols = gene symbols, values = 0/1.
    gene_set = set(gene_symbols)
    gene_idx = {g: i for i, g in enumerate(gene_symbols)}

    rows, term_ids = [], []
    for _, row in res.iterrows():
        term_name  = str(row.get("name", row.get("term_name", "")))
        intersect  = _as_list(row.get("intersections", row.get("intersection", [])))
        hit_genes  = [g for g in intersect if g in gene_set]
        if not hit_genes:
            continue
        vec = np.zeros(len(gene_symbols), dtype=np.int8)
        for g in hit_genes:
            vec[gene_idx[g]] = 1
        rows.append(vec)
        term_ids.append(term_name)

    concept_df = pd.DataFrame(rows, index=term_ids, columns=gene_symbols)
    concept_df.to_csv(CONCEPT_CSV)
    print(f"    Saved concept matrix: {len(term_ids):,} terms × {len(gene_symbols):,} genes → {CONCEPT_CSV}")

print(f"    Concept matrix shape: {concept_df.shape}")

# ── 3. Load scGPT ─────────────────────────────────────────────────────────────
# %%
print("[3] Loading scGPT model...")
adapter = ScGPTAdapter()
handle  = adapter.load(ModelSpec(name="scgpt", checkpoint=SCGPT_CKPT, device=DEVICE, options={}))
layers  = LAYERS if LAYERS is not None else adapter.list_layers(handle)
print(f"    Layers: {layers}")

# ── 4. Extract activations (all layers in one pass) ───────────────────────────
# %%
store = ActivationStore(ActivationStoreSpec(root=OUT_ROOT))

layers_needed = [l for l in layers if not (REUSE_ACTS and _shards_exist(OUT_ROOT, l))]

if not layers_needed:
    print("[4] All activation shards exist — skipping extraction.")
else:
    print(f"[4] Extracting {len(layers_needed)} layer(s) in one forward-pass sweep...")
    print(f"    Layers: {layers_needed}")
    extract_activations(
        dataset=dataset,
        model_handle=handle,
        model_adapter=adapter,
        layers=layers_needed,
        store=store,
        extraction_cfg=EXTRACT_CFG,
    )
    print("    Extraction complete.")

# ── 5. Handoff summary ────────────────────────────────────────────────────────
# %%
print("\n── Outputs ──────────────────────────────────────────────")
for layer in layers:
    n = len(store.list_shards(layer))
    print(f"  {layer:20s}  {n:4d} shards")
print(f"\n  Concept matrix : {concept_df.shape[0]:,} terms × {concept_df.shape[1]:,} genes")
print(f"  Concept CSV    : {CONCEPT_CSV}")
print(f"\nNext: run test_lp_training.py with")
print(f"  STORE_ROOT   = '{OUT_ROOT}'")
print(f"  CONCEPT_CSV  = '{CONCEPT_CSV}'")
