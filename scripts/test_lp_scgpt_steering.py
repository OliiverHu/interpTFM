"""
Test script: linear probe evaluation and activation steering on Norman scGPT activations.

Assumes the following have already been run:
  - scripts/test_lp_scgpt_acts.py  (activations + concept_matrix.csv)
  - scripts/test_lp_training.py    (probe_{LAYER}.pt)

Sections (run as a notebook with # %% or top-to-bottom):
  1. LP evaluation  — load probe + store → AUROC + top-k accuracy
  2. Steering setup — select gene, find concept directions
  3. Data split     — load Norman, partition ctrl / pert
  4. Load scGPT
  5. Collect unsteered CLS activations
  6. Run intervention
  7. Score steered activations (SGD regression)
  8. UMAP
  9. Per-layer probe activation violin plots
"""
# %%
import os

# Resolve all relative paths from the project root regardless of CWD.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_PROJECT_ROOT)

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch.utils.data import DataLoader

from interp_pipeline.adapters.model_base import ModelSpec
from interp_pipeline.adapters.models.scgpt import ScGPTAdapter
from interp_pipeline.io.activation_store import ActivationStore, ActivationStoreSpec
from interp_pipeline.linear_probe import (
    ConceptFilteredDataset,
    build_id_to_gene,
    evaluate_probe,
    load_probe,
)
from interp_pipeline.scgpt_local.tokenizer import Tokenizer
from interp_pipeline.steering.analysis import (
    analyze_probe_activations,
    plot_steering_umap,
    score_steering_regression,
)
from interp_pipeline.steering.collect import (
    collect_condition_activations,
    collect_per_layer_cls_activations,
)
from interp_pipeline.steering.intervene import (
    InterventionConfig,
    find_gene_position,
    run_intervention,
)

# ── Config ────────────────────────────────────────────────────────────────────

SCGPT_CKPT  = "/maiziezhou_lab/zihang/SpatialFoundationModel/eval/scGPT/whole-human-pretrain"
ADATA_PATH  = "/maiziezhou_lab2/zihang/interpTFM/temp/norman/perturb_processed.h5ad"
STORE_ROOT  = "runs/lp_scgpt_norman"
CONCEPT_CSV = "runs/lp_scgpt_norman/gprofiler/concept_matrix.csv"
PROBE_DIR   = "runs/lp_scgpt_norman/linear_probes"
OUT_DIR     = "runs/lp_scgpt_norman/eval_steering"
DEVICE      = "cuda"

# Layer to evaluate and steer (must have a trained probe + activation shards).
LAYER     = "layer_4"
LAYER_IDX = 4  # integer index for steering/collect APIs

# Norman obs column holding condition labels ("ctrl" or perturbed gene symbol).
CONDITION_COL = "condition"

# Gene to steer toward.  Must be present in the Norman panel and scGPT vocab.
GENE_SELECT = "CEBPA"

# Perturbation condition label in adata.obs[CONDITION_COL] for single-gene
# GENE_SELECT experiments.  Norman uses "GENE+ctrl" or "ctrl+GENE" format.
# Set to None to auto-detect (tries both orderings, picks the first match).
CONDITION_PERT = None

# Intervention scales (negative = suppress, positive = amplify).
SCALE_LIST = [-2.0, -1.0, 0.0, 1.0, 2.0]

# Cells per forward pass (steering + collection).
BATCH_SIZE = 128

# Max test samples for AUROC + top-k (avoids materialising the full test set).
MAX_EVAL = 4096

# Max GO terms to show in violin plots (caps seaborn output).
MAX_VIOLIN_CONCEPTS = 5

# ── 1. LP Evaluation ──────────────────────────────────────────────────────────
# %%
print("[1] Loading tokenizer vocab...")
tokenizer = Tokenizer(os.path.join(SCGPT_CKPT, "vocab.json"), device="cpu")
id_to_gene = build_id_to_gene(tokenizer.vocab)

print("[1] Loading concept matrix...")
concept_df = pd.read_csv(CONCEPT_CSV, index_col=0)
n_concepts = concept_df.shape[0]
print(f"    {n_concepts:,} terms × {concept_df.shape[1]:,} genes")

print("[1] Loading activation store and probe...")
store = ActivationStore(ActivationStoreSpec(root=STORE_ROOT))
probe = load_probe(os.path.join(PROBE_DIR, f"probe_{LAYER}.pt"), device=DEVICE)
print(f"    Probe weight shape: {probe.weight.shape}")  # [H, n_concepts]

print("[1] Building test dataset...")
test_ds = ConceptFilteredDataset(
    store=store,
    layer=LAYER,
    concept_matrix=concept_df,
    id_to_gene=id_to_gene,
    split="test",
    test_fraction=0.2,
    seed=42,
)
test_loader = DataLoader(test_ds, batch_size=8192, shuffle=False, num_workers=0)

print("[1] Evaluating probe...")
metrics = evaluate_probe(probe, test_loader, device=DEVICE, max_eval_samples=MAX_EVAL)
print(f"    macro-AUROC = {metrics['auroc']:.4f}")
print(f"    top-k acc   = {metrics['top_k_acc']:.4f}")

# ── 2. Steering setup ─────────────────────────────────────────────────────────
# %%
print(f"\n[2] Steering setup — gene: {GENE_SELECT}")

if GENE_SELECT not in concept_df.columns:
    available = list(concept_df.columns[:10])
    raise ValueError(
        f"'{GENE_SELECT}' not found in concept matrix columns.\n"
        f"Sample available: {available}"
    )

gene_col = concept_df[GENE_SELECT]                    # binary Series, index = GO terms
concept_idx_union = list(np.where(gene_col.values == 1)[0])
concept_names     = list(concept_df.index[concept_idx_union])
print(f"    {len(concept_idx_union)} GO terms associated with {GENE_SELECT}")
print(f"    Sample terms: {concept_names[:5]}")

# Probe weight matrix for the steering layer (CPU — NNsight moves it in-loop).
probes_for_steer = {LAYER_IDX: probe.weight.detach().cpu()}

# ── 3. Load AnnData, split ctrl / pert ───────────────────────────────────────
# %%
print("\n[3] Loading Norman AnnData...")
adata = sc.read_h5ad(ADATA_PATH)
adata.var["feature_name"] = adata.var["gene_name"]

vocab_genes = set(tokenizer.vocab.keys())
keep = adata.var["gene_name"].isin(vocab_genes)
adata = adata[:, keep].copy()
gene_names = adata.var["gene_name"].to_numpy()
print(f"    {adata.n_obs:,} cells × {adata.n_vars:,} genes after vocab filter")
print(f"    Condition column: '{CONDITION_COL}'")
print(f"    Unique conditions (first 10): {list(adata.obs[CONDITION_COL].unique()[:10])}")

adata_ctrl = adata[adata.obs[CONDITION_COL] == "ctrl"].copy()

# Resolve the perturbation condition string (Norman uses "GENE+ctrl" / "ctrl+GENE").
_cond_pert = CONDITION_PERT
if _cond_pert is None:
    all_conds = set(adata.obs[CONDITION_COL].unique())
    for candidate in (f"{GENE_SELECT}+ctrl", f"ctrl+{GENE_SELECT}"):
        if candidate in all_conds:
            _cond_pert = candidate
            break
if _cond_pert is None:
    matching = [c for c in adata.obs[CONDITION_COL].unique() if GENE_SELECT in c]
    raise ValueError(
        f"Cannot find a single-gene condition for '{GENE_SELECT}'. "
        f"Conditions containing that gene: {matching}. "
        f"Set CONDITION_PERT explicitly."
    )
print(f"    Using perturbation condition: '{_cond_pert}'")

adata_pert = adata[adata.obs[CONDITION_COL] == _cond_pert].copy()

# Subsample ctrl to match pert count so the two sets are balanced.
n_pert = adata_pert.n_obs
if adata_ctrl.n_obs > n_pert:
    rng = np.random.default_rng(seed=42)
    idx = np.sort(rng.choice(adata_ctrl.n_obs, size=n_pert, replace=False))
    adata_ctrl = adata_ctrl[idx].copy()
print(f"    ctrl: {adata_ctrl.n_obs:,}  |  {_cond_pert}: {adata_pert.n_obs:,}")

gene_position = find_gene_position(gene_names, GENE_SELECT, add_cls=True)
print(f"    Token position of {GENE_SELECT}: {gene_position}")

# ── 4. Load scGPT ─────────────────────────────────────────────────────────────
# %%
print("\n[4] Loading scGPT...")
adapter = ScGPTAdapter()
handle  = adapter.load(ModelSpec(name="scgpt", checkpoint=SCGPT_CKPT, device=DEVICE, options={}))
print(f"    n_layers={handle.n_layers}  device={handle.device}")

# ── 5. Collect unsteered CLS activations (cached) ────────────────────────────
# %%
_acts_dir = os.path.join(OUT_DIR, "activations", GENE_SELECT)
_safe = lambda s: s.replace("/", "_").replace("+", "_")
_ctrl_cache = os.path.join(_acts_dir, f"{_safe('ctrl')}_activations.pt")
_pert_cache  = os.path.join(_acts_dir, f"{_safe(_cond_pert)}_activations.pt")

if os.path.exists(_ctrl_cache) and os.path.exists(_pert_cache):
    print("\n[5] Loading cached CLS activations...")
    ctrl_act = torch.load(_ctrl_cache, map_location="cpu")
    pert_act  = torch.load(_pert_cache,  map_location="cpu")
else:
    print("\n[5] Collecting unsteered CLS activations...")
    adata_balanced = sc.concat([adata_ctrl, adata_pert])
    # sc.concat drops all var columns; restore gene_name for the tokenizer.
    adata_balanced.var["gene_name"] = adata_ctrl.var["gene_name"].values
    cond_acts = collect_condition_activations(
        handle=handle,
        adata=adata_balanced,
        condition_col=CONDITION_COL,
        conditions=["ctrl", _cond_pert],
        batch_size=BATCH_SIZE,
        include_zero_genes=True,
        normalize=False,
        output_dir=_acts_dir,
    )
    ctrl_act = cond_acts["ctrl"]
    pert_act  = cond_acts[_cond_pert]

print(f"    ctrl: {ctrl_act.shape}  |  pert: {pert_act.shape}")

# ── 6. Run intervention (cached) ─────────────────────────────────────────────
# %%
_intv_cache = os.path.join(_acts_dir, f"intv_acts_{GENE_SELECT}.pt")

if os.path.exists(_intv_cache):
    print(f"\n[6] Loading cached intervention activations...")
    intv_acts = torch.load(_intv_cache, map_location="cpu")
else:
    print(f"\n[6] Running intervention at scales {SCALE_LIST}...")
    cfg = InterventionConfig(
        gene_select=GENE_SELECT,
        scale_list=SCALE_LIST,
        batch_size=BATCH_SIZE,
        include_zero_genes=True,
        normalize=False,
    )
    intv_acts = run_intervention(
        handle=handle,
        adata_ctrl=adata_ctrl,
        probes=probes_for_steer,
        concept_idx_union=concept_idx_union,
        gene_position=gene_position,
        cfg=cfg,
        gene_names=gene_names,
        steering_mode="amplify",
    )
    os.makedirs(_acts_dir, exist_ok=True)
    torch.save(intv_acts, _intv_cache)
    print(f"    Saved: {_intv_cache}")

for scale, act in sorted(intv_acts.items()):
    print(f"    scale={scale:+.1f}  →  {act.shape}")

# ── 7. Score steered activations ──────────────────────────────────────────────
# %%
print("\n[7] Scoring steered activations (SGD regression: 0=ctrl-like, 1=pert-like)...")
scores = score_steering_regression(ctrl_act, pert_act, intv_acts)
for scale in sorted(scores):
    print(f"    scale={scale:+.1f}  score={scores[scale]:.4f}")

# ── 8. UMAP ───────────────────────────────────────────────────────────────────
# %%
print("\n[8] Plotting UMAP...")
os.makedirs(OUT_DIR, exist_ok=True)
umap_path = os.path.join(OUT_DIR, f"umap_{GENE_SELECT}.png")
plot_steering_umap(
    ctrl_act=ctrl_act,
    pert_act=pert_act,
    intv_acts=intv_acts,
    gene_select=_cond_pert,
    save_path=umap_path,
)
print(f"    Saved: {umap_path}")

# ── 9. Per-layer probe activation violin plots ────────────────────────────────
# %%
print("\n[9] Collecting per-layer CLS for probe activation analysis...")
n_vis = min(200, adata_ctrl.n_obs, adata_pert.n_obs)
adata_vis = sc.concat([adata_ctrl[:n_vis], adata_pert[:n_vis]])
condition_labels = ["ctrl"] * n_vis + [_cond_pert] * n_vis

per_layer_cls = collect_per_layer_cls_activations(
    handle=handle,
    adata=adata_vis,
    gene_names=gene_names,
    batch_size=BATCH_SIZE,
    include_zero_genes=True,
)
print(f"    Captured {len(per_layer_cls)} layers, {n_vis * 2} cells each")

n_terms = min(MAX_VIOLIN_CONCEPTS, len(concept_idx_union))
violin_dir = os.path.join(OUT_DIR, "violins")
df_proj = analyze_probe_activations(
    per_layer_cls=per_layer_cls,
    probes={LAYER_IDX: probe.weight.detach().cpu()},
    concept_idx_union=concept_idx_union[:n_terms],
    concept_names=concept_names[:n_terms],
    condition_labels=condition_labels,
    save_dir=violin_dir,
)
print(f"    Projection DataFrame: {df_proj.shape}")
print(df_proj.head())

print("\nDone.")

# %%