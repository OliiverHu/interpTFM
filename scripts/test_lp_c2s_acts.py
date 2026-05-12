"""
Extract C2S (Pythia-410m) gene-level activations on the Norman perturbation
dataset and build a binary GO concept matrix via g:Profiler.

No SAE training.  Output is consumed by test_lp_c2s_training.py.

Key differences vs. test_lp_scgpt_acts.py:
  - C2S uses an NLP subword tokenizer (BPE) so each gene name becomes a
    variable-length sub-token span.  Gene-level representations are obtained
    by pooling over the span (default: last token).
  - token_ids stored in the ActivationStore are gene name strings, not integer
    vocab IDs.  test_lp_c2s_training.py uses an identity id_to_gene map.
  - Extraction is a two-step process:
      (a) extract_c2s_dataset → C2S raw format (batch_*.pt + cell_gene_pairs.txt)
      (b) convert_c2s_layer_to_activation_store → generic ActivationStore shards
  - Concept matrix is built from all gene symbols in the Norman panel
    (C2S has no fixed vocab restriction).

Steps:
  1. Load (or prepare) Norman AnnData with shard labels, feature_name,
     author_cell_type, and organism baked in.  Saved to NORMAN_PREPPED_PATH
     so shard assignment is stable across re-runs.
  2. Call g:Profiler; build binary concept matrix [n_terms × n_genes].
  3. Extract C2S activations → C2S raw format  (skipped if shards exist).
  4. Convert raw format → generic ActivationStore  (skipped if shards exist).
  5. Handoff summary for test_lp_c2s_training.py.
"""
# %%
import os
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import scanpy as sc

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_PROJECT_ROOT)

from interp_pipeline.extraction.c2s_extraction import (
    extract_c2s_dataset,
    convert_c2s_layer_to_activation_store,
)
from interp_pipeline.get_annotation.gprofiler_client import GProfilerClient, GProfilerSpec
from interp_pipeline.io.activation_store import ActivationStore, ActivationStoreSpec

# ── Config ────────────────────────────────────────────────────────────────────

ADATA_PATH = "/maiziezhou_lab2/zihang/interpTFM/temp/norman/perturb_processed.h5ad"

# Prepared h5ad with shard labels + C2S-required columns baked in.
# Written once; reused on subsequent runs so shard assignment is stable.
NORMAN_PREPPED_PATH = "/maiziezhou_lab2/zihang/interpTFM/temp/norman/perturb_c2s_prepped.h5ad"

# Current C2S model (Pythia-410m, from HuggingFace).
C2S_MODEL = "vandijklab/C2S-Pythia-410m-cell-type-conditioned-cell-generation"
CACHE_DIR = "/maiziezhou_lab2/zihang/interpTFM/cache/c2s"

DEVICE = "cuda"

OUT_ROOT         = "runs/lp_c2s_norman"          # generic ActivationStore + probes
C2S_EXTRACT_ROOT = "runs/lp_c2s_norman/c2s_raw"  # C2S raw format (intermediate)

# Layers to extract.  Pythia-410m has 24 layers (layer_0 … layer_23).
LAYERS = ["layer_12"]

# g:Profiler.
GPROF_SOURCES = ["GO:BP", "GO:MF", "GO:CC"]
GPROF_ALPHA   = 0.05

GPROF_DIR   = os.path.join(OUT_ROOT, "gprofiler")
CONCEPT_CSV = os.path.join(GPROF_DIR, "concept_matrix.csv")


@dataclass(frozen=True)
class ExtractionConfig:
    shards:    int           = 60
    shard_key: str           = "shards"
    batch_size: int          = 4
    max_genes: int           = 256
    device:    str           = DEVICE
    pooling:   str           = "last"
    save_dtype: str          = "fp16"
    pool_dtype: str          = "fp32"
    normalize: bool          = False   # Norman is already processed
    cache_dir: Optional[str] = CACHE_DIR


EXTRACT_CFG = ExtractionConfig()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _as_list(x: Any) -> List[str]:
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


def _c2s_raw_exist(extract_root: str, layer: str) -> bool:
    pat = os.path.join(extract_root, "activations", layer, "shard_*", "batch_*_gene_acts.pt")
    return len(glob.glob(pat)) > 0


def _generic_shards_exist(store_root: str, layer: str) -> bool:
    pat = os.path.join(store_root, "activations", layer, "shard_*", "activations.pt")
    return len(glob.glob(pat)) > 0


# ── 1. Load / prepare Norman AnnData ─────────────────────────────────────────
# %%
if Path(NORMAN_PREPPED_PATH).exists():
    print(f"[1] Loading pre-prepared Norman AnnData: {NORMAN_PREPPED_PATH}")
    adata = sc.read_h5ad(NORMAN_PREPPED_PATH)
else:
    print("[1] Preparing Norman AnnData for C2S (saved for stable shard assignment)...")
    adata = sc.read_h5ad(ADATA_PATH)
    print(f"    {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    # C2S make_batches requires:
    #   var["feature_name"]      — gene symbols for cell sentence construction
    #   obs["author_cell_type"]  — copied to obs["cell_type"] inside make_batches
    #   obs["organism"]          — fills {organism} slot in the prompt template
    adata.var["feature_name"]     = adata.var["gene_name"]
    adata.obs["author_cell_type"] = adata.obs["condition"]
    adata.obs["organism"]         = "human"

    rng = np.random.default_rng(seed=42)
    adata.obs[EXTRACT_CFG.shard_key] = rng.choice(
        [f"shard_{i}" for i in range(EXTRACT_CFG.shards)], size=adata.n_obs
    )

    Path(NORMAN_PREPPED_PATH).parent.mkdir(parents=True, exist_ok=True)
    adata.write(NORMAN_PREPPED_PATH)
    print(f"    Saved prepared AnnData: {NORMAN_PREPPED_PATH}")

gene_symbols = list(adata.var["gene_name"])
print(f"    {adata.n_obs:,} cells × {adata.n_vars:,} genes  |  {EXTRACT_CFG.shards} shards")

# ── 2. Build GO concept matrix ────────────────────────────────────────────────
# %%
os.makedirs(GPROF_DIR, exist_ok=True)

if os.path.exists(CONCEPT_CSV):
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
    res = gp.profile(gene_symbols, spec=spec, query_name="norman_c2s_panel", force=False)
    if not isinstance(res, pd.DataFrame):
        res = pd.DataFrame(res)

    qcol = next(
        (c for c in ["p_value", "p_value_fdr", "p_value_adj", "p_value_corrected"]
         if c in res.columns),
        None,
    )
    if qcol:
        res = res[res[qcol] <= GPROF_ALPHA].copy()
    print(f"    {len(res):,} significant terms")

    gene_set = set(gene_symbols)
    gene_idx = {g: i for i, g in enumerate(gene_symbols)}

    rows, term_ids = [], []
    for _, row in res.iterrows():
        term_name = str(row.get("name", row.get("term_name", "")))
        intersect = _as_list(row.get("intersections", row.get("intersection", [])))
        hit_genes = [g for g in intersect if g in gene_set]
        if not hit_genes:
            continue
        vec = np.zeros(len(gene_symbols), dtype=np.int8)
        for g in hit_genes:
            vec[gene_idx[g]] = 1
        rows.append(vec)
        term_ids.append(term_name)

    concept_df = pd.DataFrame(rows, index=term_ids, columns=gene_symbols)
    concept_df.to_csv(CONCEPT_CSV)
    print(f"    Saved: {len(term_ids):,} terms × {len(gene_symbols):,} genes → {CONCEPT_CSV}")

print(f"    Concept matrix shape: {concept_df.shape}")

# ── 3. Extract C2S activations → C2S raw format ───────────────────────────────
# %%
missing_layers = [l for l in LAYERS if not _c2s_raw_exist(C2S_EXTRACT_ROOT, l)]

if not missing_layers:
    print("\n[3] C2S raw shards exist for all layers — skipping extraction.")
else:
    print(f"\n[3] Extracting layers {missing_layers}  (model: {C2S_MODEL})")
    extract_c2s_dataset(
        adata=adata,
        model_path=C2S_MODEL,
        output_dir=C2S_EXTRACT_ROOT,
        layers=missing_layers,
        shards=EXTRACT_CFG.shards,
        shard_key=EXTRACT_CFG.shard_key,
        batch_size=EXTRACT_CFG.batch_size,
        max_genes=EXTRACT_CFG.max_genes,
        device=EXTRACT_CFG.device,
        pooling=EXTRACT_CFG.pooling,
        save_dtype=EXTRACT_CFG.save_dtype,
        pool_dtype=EXTRACT_CFG.pool_dtype,
        normalize=EXTRACT_CFG.normalize,
        cache_dir=EXTRACT_CFG.cache_dir,
    )
    print("    Extraction complete.")

# ── 4. Convert C2S raw → generic ActivationStore ─────────────────────────────
# %%
missing_generic = [l for l in LAYERS if not _generic_shards_exist(OUT_ROOT, l)]

if not missing_generic:
    print("[4] Generic shards exist for all layers — skipping conversion.")
else:
    print(f"\n[4] Converting C2S raw → ActivationStore for {missing_generic}...")
    for layer in missing_generic:
        n_written = convert_c2s_layer_to_activation_store(
            c2s_root=C2S_EXTRACT_ROOT,
            out_root=OUT_ROOT,
            layer=layer,
            overwrite=False,
        )
        print(f"    {layer}: {n_written} shards written")

# ── 5. Handoff summary ────────────────────────────────────────────────────────
# %%
store = ActivationStore(ActivationStoreSpec(root=OUT_ROOT))
print("\n── Outputs ──────────────────────────────────────────────")
for layer in LAYERS:
    n = len(store.list_shards(layer))
    print(f"  {layer:20s}  {n:4d} shards")
print(f"\n  Concept matrix : {concept_df.shape[0]:,} terms × {concept_df.shape[1]:,} genes")
print(f"  Concept CSV    : {CONCEPT_CSV}")
print(f"\n  token_ids in the store are gene name strings — use identity id_to_gene in training.")
print(f"\nNext: run test_lp_c2s_training.py with")
print(f"  STORE_ROOT   = '{OUT_ROOT}'")
print(f"  CONCEPT_CSV  = '{CONCEPT_CSV}'")
