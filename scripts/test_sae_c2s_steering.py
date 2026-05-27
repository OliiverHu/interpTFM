#!/usr/bin/env python3
# %%
from __future__ import annotations

import json
import os
import sys
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from interp_pipeline.sae.sae_base import AutoEncoder

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH   = "/maiziezhou_lab2/yunfei/Projects/interpTFM-legacy/c2sscale/models/C2S-Scale-Gemma-2-2B"
SAE_PATH     = (
    "/maiziezhou_lab2/yunfei/Projects/interpTFM/runs/full_c2sscale_cosmx"
    "/sae/layer_17/nr_on__steps_6000__l1_3e-3/sae_layer_17_best_normalized.pt"
)
GPROFILER_DIR = "/maiziezhou_lab2/yunfei/Projects/interpTFM/runs/full_c2sscale_cosmx/gprofiler"
ADATA_PATH   = "/maiziezhou_lab2/yunfei/Projects/FM_temp/interGFM/ge_shards/cosmx_human_lung_sec8.h5ad"
F1_PATH      = "/maiziezhou_lab2/yunfei/Projects/interpTFM/runs/full_c2sscale_cosmx/f1_analysis/f1_long.parquet"
OUT_DIR      = "runs/sae_c2s_steering"
DEVICE       = "cuda"

SAE_LAYER       = 17          # hook layer index
TARGET_FEATURE  = 8208        # neutrophil-specific (neut_pct=91.8%, fibro_mean=0.016)

# GO terms are selected automatically from f1_long.parquet for TARGET_FEATURE:
#   1. rank all terms by F1, keep only those above MIN_F1_CUTOFF
#   2. from those, take at most TOP_K_TERMS
MIN_F1_CUTOFF = 0.7           # hard threshold — exclude terms with F1 below this
TOP_K_TERMS   = 10            # cap on how many terms to use after the cutoff

# Clamp values in normalized units (1.0 = max natural activation of this feature)
CLAMP_VALUES  = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]

ORGANISM          = "Homo sapiens"
CELL_TYPE         = "lung fibroblast"   # negative control: feat=8208 near-zero (mean=0.016)
POS_CTRL_CELL_TYPE = "mature neutrophil" # positive control: feat=8208 highly active (pct=91.8%)
NUM_GENES     = 128
MAX_NEW_TOKENS = NUM_GENES + 10   # extra headroom for BPE fragments
N_GENERATIONS = 20
TOP_K_SCORE   = 50
TEMPERATURE   = 1.0
DO_SAMPLE     = True
SEED_BASE     = 42
CACHE_FILE    = os.path.join(OUT_DIR, f"sae_steering_feat{TARGET_FEATURE}_n{N_GENERATIONS}.json")

# ---------------------------------------------------------------------------
# 1. Select GO terms + build target gene set for TARGET_FEATURE
# ---------------------------------------------------------------------------
def select_go_terms(f1_path: str, feature_id: int,
                    min_f1: float, top_k: int) -> List[tuple[str, str, float]]:
    """
    Rank all GO terms for feature_id by F1, apply min_f1 hard cutoff,
    then return at most top_k terms.
    """
    f1_df = pd.read_parquet(f1_path)
    sub = (f1_df[f1_df["latent"] == feature_id]
           .sort_values("f1", ascending=False)
           .drop_duplicates("term_id"))
    above = sub[sub["f1"] >= min_f1].head(top_k)
    terms = list(zip(above["term_id"], above["term_name"], above["f1"]))
    print(f"[f1] feat={feature_id}: {len(sub)} total terms, "
          f"{len(above)} with F1 >= {min_f1} (capped at {top_k})")
    for tid, tname, f1v in terms:
        print(f"     {tid}  f1={f1v:.3f}  {tname}")
    if not terms:
        print(f"[f1] WARNING: no GO terms pass F1 >= {min_f1} for feat={feature_id}. "
              f"Best F1 = {sub['f1'].iloc[0]:.3f} ({sub['term_name'].iloc[0]})")
    return terms


def build_target_genes(gprofiler_dir: str, adata_path: str,
                       go_terms: List[tuple[str, str, float]]) -> tuple[set, float]:
    import scanpy as sc
    bin_df = pd.read_csv(
        os.path.join(gprofiler_dir, "gprofiler_binary_gene_by_term.csv"), index_col=0
    )
    adata = sc.read_h5ad(adata_path)
    ens_to_sym = dict(zip(adata.var.index, adata.var["feature_name"]))

    term_ids = [tid for tid, _, _ in go_terms]
    avail = [t for t in term_ids if t in bin_df.columns]
    print(f"[genes] {len(avail)}/{len(term_ids)} GO terms found in gprofiler matrix")

    gene_mask = bin_df[avail].any(axis=1)
    sym_genes = {ens_to_sym[e] for e in bin_df.index[gene_mask] if e in ens_to_sym}
    all_syms  = {ens_to_sym[e] for e in bin_df.index if e in ens_to_sym}
    background = len(sym_genes) / max(len(all_syms), 1)
    print(f"[genes] target genes: {len(sym_genes)} / panel: {len(all_syms)} "
          f"(background rate = {background:.1%})")
    return sym_genes, background

# ---------------------------------------------------------------------------
# 2. Load model + SAE
# ---------------------------------------------------------------------------
def load_model(model_path: str, device: str):
    print(f"[model] loading {model_path}")
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device
    )
    model.eval()
    print(f"[model] hidden_size={model.config.hidden_size}, "
          f"layers={model.config.num_hidden_layers}")
    return model, tok

def load_sae(sae_path: str, device: str) -> AutoEncoder:
    print(f"[sae] loading {sae_path}")
    ae = AutoEncoder.from_pretrained(sae_path, device=device)
    ae.eval()
    ae.to(device)
    print(f"[sae] d_in={ae.activation_dim}, n_latents={ae.dict_size}")
    return ae

# ---------------------------------------------------------------------------
# 3. Hook factory — InterPLM decompose-clamp-recompose
# ---------------------------------------------------------------------------
def _make_sae_hook(sae: AutoEncoder, feature_id: int, clamp_val: float,
                   _diag: dict | None = None):
    """
    Gemma-2 uses KV cache: decode steps have hs.shape[1] == 1.
    Prefill has hs.shape[1] > 1 — we skip it.
    _diag: optional dict to accumulate pre-clamp activation values for diagnostics.
    """
    def hook(module, inp, out):
        hs = out[0] if isinstance(out, tuple) else out   # [B, T, H]
        if hs.shape[1] != 1:                              # skip prefill
            return out
        h = hs[:, 0].float()                              # [B, H]
        z   = sae.encode(h)                               # [B, n_latents]
        err = h - sae.decode(z)                           # [B, H]
        if _diag is not None:
            _diag.setdefault("pre_clamp", []).append(float(z[:, feature_id].mean()))
        z[:, feature_id] = clamp_val                      # clamp feature
        h_new = sae.decode(z) + err                       # recompose
        hs[:, 0] = h_new.to(hs.dtype)
        return (hs,) + out[1:] if isinstance(out, tuple) else hs
    return hook

# ---------------------------------------------------------------------------
# 4. Single generation run
# ---------------------------------------------------------------------------
def run_generation(model, tokenizer, sae, prompt_ids: torch.Tensor,
                   clamp_val: float, seed: int,
                   diag: dict | None = None,
                   use_hook: bool = True) -> List[str]:
    diag_dict = diag if diag is not None else {}
    if use_hook:
        h = model.model.layers[SAE_LAYER].register_forward_hook(
            _make_sae_hook(sae, TARGET_FEATURE, clamp_val, _diag=diag_dict)
        )

    torch.manual_seed(seed)
    with torch.no_grad():
        out_ids = model.generate(
            prompt_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    if use_hook:
        h.remove()

    new_ids = out_ids[0, prompt_ids.shape[1]:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True).strip().rstrip(".")
    return text.split()

# ---------------------------------------------------------------------------
# 5. Score generation
# ---------------------------------------------------------------------------
def score_genes(genes: List[str], target_set: set, top_k: int) -> float:
    top = genes[:top_k]
    hits = sum(1 for g in top if g in target_set)
    return hits / max(len(top), 1)

# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)

go_terms = select_go_terms(F1_PATH, TARGET_FEATURE, MIN_F1_CUTOFF, TOP_K_TERMS)
target_genes, background_rate = build_target_genes(GPROFILER_DIR, ADATA_PATH, go_terms)

model, tokenizer = load_model(MODEL_PATH, DEVICE)
sae = load_sae(SAE_PATH, DEVICE)

def _make_prompt(cell_type: str) -> torch.Tensor:
    text = (
        f"Generate a list of {NUM_GENES} genes in order of descending expression "
        f"which represent a {ORGANISM} cell of cell type {cell_type}.\nCell sentence:"
    )
    return tokenizer(text, return_tensors="pt")["input_ids"].to(DEVICE)

neg_prompt_ids = _make_prompt(CELL_TYPE)
pos_prompt_ids = _make_prompt(POS_CTRL_CELL_TYPE)
print(f"[gen] neg ctrl prompt ({CELL_TYPE}): {neg_prompt_ids.shape[1]} tokens")
print(f"[gen] pos ctrl prompt ({POS_CTRL_CELL_TYPE}): {pos_prompt_ids.shape[1]} tokens")
print(f"[gen] steering feature {TARGET_FEATURE} with clamp_values={CLAMP_VALUES}")

# ---- run or load cache ------------------------------------------------
if os.path.exists(CACHE_FILE):
    print(f"[cache] loading from {CACHE_FILE}")
    with open(CACHE_FILE) as f:
        results = json.load(f)
else:
    results = {}

    # steering experiment: lung fibroblast prompt + hook at each clamp value
    for cv in CLAMP_VALUES:
        key = str(cv)
        seqs = []
        for i in tqdm(range(N_GENERATIONS),
                        desc=f"clamp={cv:.2f}", leave=False):
            diag: dict = {}
            genes = run_generation(model, tokenizer, sae, neg_prompt_ids,
                                    clamp_val=cv, seed=SEED_BASE + i,
                                    diag=diag, use_hook=True)
            seqs.append(genes)
            if i == 0:
                pre = diag.get("pre_clamp", [])
                print(f"  clamp={cv:.2f}: feat{TARGET_FEATURE} pre-clamp "
                        f"mean={np.mean(pre):.4f} pct_active={np.mean([v>0 for v in pre]):.3f} "
                        f"sample[0][:8]={genes[:8]}")
        results[key] = seqs

    # positive control: mature neutrophil prompt, no hook (natural activation)
    pos_seqs = []
    for i in tqdm(range(N_GENERATIONS), desc="pos_ctrl", leave=False):
        genes = run_generation(model, tokenizer, sae, pos_prompt_ids,
                                clamp_val=0.0, seed=SEED_BASE + i,
                                use_hook=False)
        pos_seqs.append(genes)
    results["positive_control"] = pos_seqs
    print(f"  pos_ctrl sample[:8]={pos_seqs[0][:8]}")

    with open(CACHE_FILE, "w") as f:
        json.dump(results, f)
    print(f"[cache] saved to {CACHE_FILE}")

# ---- score -------------------------------------------------------------
rows = []
for cv in CLAMP_VALUES:
    key = str(cv)
    seqs = results[key]
    scores = [score_genes(g, target_genes, TOP_K_SCORE) for g in seqs]
    rows.append({
        "clamp": cv,
        "mean_score": np.mean(scores),
        "std_score": np.std(scores),
        "fold": np.mean(scores) / max(background_rate, 1e-8),
        "n_valid": sum(1 for g in seqs if len(g) >= 3),
    })

df = pd.DataFrame(rows)

# baseline scores
neg_scores = [score_genes(g, target_genes, TOP_K_SCORE) for g in results[str(CLAMP_VALUES[0])]]
pos_scores = [score_genes(g, target_genes, TOP_K_SCORE) for g in results.get("positive_control", [])]
neg_mean = np.mean(neg_scores)
pos_mean = np.mean(pos_scores) if pos_scores else None

print("\n=== Baseline Scores ===")
print(f"  negative ctrl ({CELL_TYPE}, clamp=0.0): {neg_mean:.3f}")
if pos_mean is not None:
    print(f"  positive ctrl ({POS_CTRL_CELL_TYPE}, no hook): {pos_mean:.3f}")
print("\n=== Steering Results ===")
print(df.to_string(index=False))
df.to_csv(os.path.join(OUT_DIR, "scores.csv"), index=False)

# ---- plots -------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# (A) mean score +/- std
ax = axes[0]
ax.errorbar(df["clamp"], df["mean_score"], yerr=df["std_score"],
            fmt="o-", capsize=4, color="steelblue", label="steered (lung fibroblast)")
ax.axhline(background_rate, color="gray", ls="--", lw=1.2,
            label=f"background ({background_rate:.1%})")
ax.axhline(neg_mean, color="tomato", ls=":", lw=1.5,
            label=f"neg ctrl: {CELL_TYPE} ({neg_mean:.3f})")
if pos_mean is not None:
    ax.axhline(pos_mean, color="green", ls=":", lw=1.5,
                label=f"pos ctrl: {POS_CTRL_CELL_TYPE} ({pos_mean:.3f})")
ax.set_xlabel("SAE clamp value (normalized units, 1.0 = natural max)")
ax.set_ylabel(f"Fraction top-{TOP_K_SCORE} in target genes")
ax.set_title("A. Enrichment vs. clamp value")
ax.legend(fontsize=7)

# (B) fold enrichment bar
ax = axes[1]
ax.bar(df["clamp"].astype(str), df["fold"], color="steelblue", alpha=0.8)
ax.axhline(1.0, color="gray", ls="--", lw=1.2, label="no enrichment")
ax.axhline(neg_mean / max(background_rate, 1e-8), color="tomato", ls=":", lw=1.5,
            label=f"neg ctrl")
if pos_mean is not None:
    ax.axhline(pos_mean / max(background_rate, 1e-8), color="green", ls=":", lw=1.5,
                label=f"pos ctrl")
ax.set_xlabel("SAE clamp value")
ax.set_ylabel("Fold enrichment over background")
ax.set_title("B. Fold enrichment")
ax.legend(fontsize=7)

# (C) gene frequency shift: baseline vs. max clamp
ax = axes[2]
cv_max = max(CLAMP_VALUES)
from collections import Counter
freq_base  = Counter(g for genes in results[str(CLAMP_VALUES[0])]
                        for g in genes[:TOP_K_SCORE])
freq_steer = Counter(g for genes in results[str(cv_max)]
                        for g in genes[:TOP_K_SCORE])
all_genes = set(freq_base) | set(freq_steer)
diffs = {g: freq_steer.get(g, 0) - freq_base.get(g, 0) for g in all_genes}
top_gained = sorted(diffs, key=diffs.get, reverse=True)[:15]
top_lost   = sorted(diffs, key=diffs.get)[:10]
g_list = top_lost + top_gained
d_list = [diffs[g] for g in g_list]
colors = ["tomato" if d < 0 else "steelblue" for d in d_list]
ax.barh(g_list, d_list, color=colors)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel(f"Freq change (clamp={cv_max} vs baseline)")
ax.set_title(f"C. Gene freq shift (top-{TOP_K_SCORE})")
# mark granulocyte genes with star
for i, g in enumerate(g_list):
    if g in target_genes:
        ax.text(0, i, " ★", va="center", fontsize=7, color="darkgreen")

fig.suptitle(
    f"SAE feature {TARGET_FEATURE} (neutrophil-specific, neut_pct=91.8%) clamping\n"
    f"C2S-Gemma-2-2B | layer {SAE_LAYER} | prompt: {CELL_TYPE} → steer toward neutrophil",
    fontsize=11
)
fig.tight_layout()
out_fig = os.path.join(OUT_DIR, "steering_results.pdf")
fig.savefig(out_fig, bbox_inches="tight")
print(f"\n[plot] saved -> {out_fig}")

# ---- coherence check --------------------------------------------------
print("\n=== Coherence check (sample genes per condition) ===")
if "positive_control" in results:
    print(f"  pos ctrl ({POS_CTRL_CELL_TYPE}): {results['positive_control'][0][:12]}")
for cv in CLAMP_VALUES:
    sample = results[str(cv)][0][:12]
    print(f"  clamp={cv:.2f} ({CELL_TYPE}): {sample}")


# %%