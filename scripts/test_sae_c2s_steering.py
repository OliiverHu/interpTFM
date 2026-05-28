#!/usr/bin/env python3
# %%
from __future__ import annotations

import json
import os
import sys
from collections import Counter
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
MODEL_PATH    = "/maiziezhou_lab2/yunfei/Projects/interpTFM-legacy/c2sscale/models/C2S-Scale-Gemma-2-2B"
SAE_PATH      = (
    "/maiziezhou_lab2/yunfei/Projects/interpTFM/runs/full_c2sscale_cosmx"
    "/sae/layer_17/nr_on__steps_6000__l1_3e-3/sae_layer_17_best_normalized.pt"
)
GPROFILER_DIR = "/maiziezhou_lab2/yunfei/Projects/interpTFM/runs/full_c2sscale_cosmx/gprofiler"
ADATA_PATH    = "/maiziezhou_lab2/yunfei/Projects/FM_temp/interGFM/ge_shards/cosmx_human_lung_sec8.h5ad"
# F1_PATH: no longer needed — go_term_id is read directly from the pairs JSON
PAIRS_PATH    = os.path.join(os.path.dirname(__file__), "..",
                             "runs/f1_celltype_analysis/feature_celltype_pairs.json")
PROMPTS_PATH  = os.path.join(os.path.dirname(__file__), "..",
                             "src/interp_pipeline/c2s_local/prompts/"
                             "single_cell_cell_type_conditional_generation_prompts.json")
BASE_OUT_DIR  = "runs/sae_c2s_steering"
DEVICE        = "cuda"

SAE_LAYER     = 17

# MIN_F1_CUTOFF = 0.7   # replaced by go_term field in pairs JSON
# TOP_K_TERMS   = 10    # replaced by go_term field in pairs JSON

# CLAMP_VALUES  = [0.0, 1.0, 2.0, 3.0, 5.0, 10.0]
CLAMP_VALUES  = [0.0, 1.0, 2.0, 3.0, 5.0]

with open(PROMPTS_PATH) as _pf:
    PROMPT_TEMPLATES = json.load(_pf)["model_input"]

ORGANISM       = "Homo sapiens"
NUM_GENES      = 128
MAX_NEW_TOKENS = NUM_GENES * 3  # gene names avg 2.64 tokens each in Gemma-2 BPE
N_GENERATIONS  = 20
TOP_K_SCORE    = NUM_GENES
TEMPERATURE    = 1.0
DO_SAMPLE      = True
SEED_BASE      = 42

# ---------------------------------------------------------------------------
# 1. Select GO terms + build target gene set
# ---------------------------------------------------------------------------
# select_go_terms: replaced by get_go_term_from_pair below, which uses the
# go_term name stored in the pairs JSON instead of re-ranking from f1_long.
#
# def select_go_terms(f1_path: str, feature_id: int,
#                     min_f1: float, top_k: int) -> List[tuple[str, str, float]]:
#     f1_df = pd.read_parquet(f1_path)
#     sub = (f1_df[f1_df["latent"] == feature_id]
#            .sort_values("f1", ascending=False)
#            .drop_duplicates("term_id"))
#     above = sub[sub["f1"] >= min_f1].head(top_k)
#     terms = list(zip(above["term_id"], above["term_name"], above["f1"]))
#     print(f"[f1] feat={feature_id}: {len(sub)} total terms, "
#           f"{len(above)} with F1 >= {min_f1} (capped at {top_k})")
#     for tid, tname, f1v in terms:
#         print(f"     {tid}  f1={f1v:.3f}  {tname}")
#     if not terms:
#         best_f1 = sub["f1"].iloc[0] if len(sub) else 0.0
#         best_name = sub["term_name"].iloc[0] if len(sub) else "N/A"
#         print(f"[f1] WARNING: no GO terms pass F1 >= {min_f1} for feat={feature_id}. "
#               f"Best F1 = {best_f1:.3f} ({best_name})")
#     return terms


# get_go_term_from_pair: replaced by inline construction in run_pair using the
# go_term name and f1 stored directly in the pairs JSON.
#
# def get_go_term_from_pair(pair: dict, f1_path: str) -> List[tuple[str, str, float]]:
#     feature_id = pair["latent"]
#     go_name    = pair["go_term"]
#     f1_val     = pair.get("f1", float("nan"))
#     f1_df = pd.read_parquet(f1_path)
#     match = f1_df[
#         (f1_df["latent"] == feature_id) & (f1_df["term_name"] == go_name)
#     ].drop_duplicates("term_id")
#     if match.empty:
#         print(f"[f1] WARNING: go_term '{go_name}' not found in parquet for feat={feature_id}")
#         return []
#     term_id = match["term_id"].iloc[0]
#     f1_resolved = float(match["f1"].iloc[0]) if "f1" in match.columns else f1_val
#     print(f"[f1] feat={feature_id}: using go_term '{go_name}' -> {term_id}  f1={f1_resolved:.3f}")
#     return [(term_id, go_name, f1_resolved)]


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
# 3. Hook factory
# ---------------------------------------------------------------------------
def _make_sae_hook(
    sae: AutoEncoder,
    feature_id: int,
    clamp_val: float,
    _diag: dict | None = None,
):
    def hook(module, input, output):
        # HuggingFace transformer blocks often return either:
        #   hidden_states
        # or
        #   (hidden_states, ...)
        hs = output[0] if isinstance(output, tuple) else output

        # Clone to avoid in-place modification issues.
        hs_new = hs.clone()

        # Intervene on the final sequence position.
        # During prefill: this is the last prompt token.
        # During decoding: this is the current generated token.
        h = hs_new[:, -1, :].float()

        # Encode into SAE feature space.
        z = sae.encode(h)

        # Preserve SAE reconstruction residual.
        recon = sae.decode(z)
        err = h - recon

        if _diag is not None:
            _diag.setdefault("pre_clamp", []).append(
                float(z[:, feature_id].mean().detach().cpu())
            )

        # Clamp selected SAE feature.
        z_new = z.clone()
        z_new[:, feature_id] = clamp_val

        # Decode back to hidden-state space.
        h_new = sae.decode(z_new) + err

        # Put modified hidden state back.
        hs_new[:, -1, :] = h_new.to(dtype=hs.dtype, device=hs.device)

        if isinstance(output, tuple):
            return (hs_new,) + output[1:]
        else:
            return hs_new

    return hook

# ---------------------------------------------------------------------------
# 4. Generation
# ---------------------------------------------------------------------------
def run_generation(model, tokenizer, sae, prompt_ids: torch.Tensor,
                   feature_id: int, clamp_val: float, seed: int,
                   diag: dict | None = None,
                   use_hook: bool = True) -> tuple[List[str], str]:
    diag_dict = diag if diag is not None else {}
    if use_hook:
        h = model.model.layers[SAE_LAYER].register_forward_hook(
            _make_sae_hook(sae, feature_id, clamp_val, _diag=diag_dict)
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
    cell_sentence = tokenizer.decode(new_ids, skip_special_tokens=True).strip().rstrip(".")
    return cell_sentence.split(), cell_sentence

def _make_prompt(tokenizer, cell_type: str, template_idx: int = 0) -> tuple[torch.Tensor, str]:
    template = PROMPT_TEMPLATES[template_idx % len(PROMPT_TEMPLATES)]
    text = template.format(num_genes=NUM_GENES, organism=ORGANISM, cell_type=cell_type)
    ids = tokenizer(text, return_tensors="pt")["input_ids"].to(DEVICE)
    return ids, text

# ---------------------------------------------------------------------------
# 5. Hook verification
# ---------------------------------------------------------------------------
def verify_hook(model, tokenizer, sae, feature_id: int, n_tokens: int = 30):
    """
    Single greedy run per clamp value with output_scores=True.
    Directly compares output logits step-by-step to verify the hook changes predictions.
    """
    prompt_ids, _ = _make_prompt(tokenizer, "lung fibroblast", template_idx=0)

    results = {}
    for cv in [0.0, 20.0]:
        diag: dict = {}
        handle = model.model.layers[SAE_LAYER].register_forward_hook(
            _make_sae_hook(sae, feature_id, cv, _diag=diag)
        )
        torch.manual_seed(SEED_BASE)
        with torch.no_grad():
            out = model.generate(
                prompt_ids, max_new_tokens=n_tokens, do_sample=False,
                output_scores=True, return_dict_in_generate=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        handle.remove()
        pre = diag.get("pre_clamp", [])
        results[cv] = {
            "hook_fires": len(pre),
            "pre_mean": float(np.mean(pre)) if pre else float("nan"),
            "scores": torch.stack(out.scores).cpu().float(),  # [n_tokens, batch, vocab]
            "tokens": out.sequences[0, prompt_ids.shape[1]:].cpu(),
        }

    return results


# ---------------------------------------------------------------------------
# 5b. Score
# ---------------------------------------------------------------------------
def score_genes(genes: List[str], target_set: set, top_k: int) -> float:
    top = genes[:top_k]
    hits = sum(1 for g in top if g in target_set)
    return hits / max(len(top), 1)

# ---------------------------------------------------------------------------
# 6. Per-pair run
# ---------------------------------------------------------------------------
def run_pair(pair: dict, model, tokenizer, sae) -> dict | None:
    feature_id = pair["latent"]
    pos_cell   = pair["up_cell"]
    neg_cell   = pair["down_cell"]

    out_dir = os.path.join(BASE_OUT_DIR, f"feat{feature_id}")
    os.makedirs(out_dir, exist_ok=True)
    cache_file = os.path.join(out_dir, f"sae_steering_feat{feature_id}_n{N_GENERATIONS}.json")

    print(f"\n{'='*70}")
    print(f"[pair] feat={feature_id}  up={pos_cell}  down={neg_cell}")
    print(f"{'='*70}")

    go_terms = [(pair["go_term_id"], pair["go_term"], pair["f1"])]
    print(f"[f1] feat={feature_id}: {go_terms[0][0]}  f1={pair['f1']:.3f}  {pair['go_term']}")

    target_genes, background_rate = build_target_genes(GPROFILER_DIR, ADATA_PATH, go_terms)

    print(f"[gen] {len(PROMPT_TEMPLATES)} prompt templates, cycling per generation")

    # ---- run or load cache ------------------------------------------------
    if os.path.exists(cache_file):
        print(f"[cache] loading from {cache_file}")
        with open(cache_file) as f:
            results = json.load(f)
    else:
        results = {}

        for cv in CLAMP_VALUES:
            key = str(cv)
            seqs = []
            for i in tqdm(range(N_GENERATIONS),
                          desc=f"feat{feature_id} clamp={cv:.2f}", leave=False):
                diag: dict = {}
                prompt_ids, prompt_text = _make_prompt(tokenizer, neg_cell, template_idx=i)
                genes, cell_sentence = run_generation(model, tokenizer, sae, prompt_ids,
                                                      feature_id=feature_id, clamp_val=cv,
                                                      seed=SEED_BASE + i, diag=diag, use_hook=True)
                seqs.append({"genes": genes, "n_genes": len(genes),
                             "n_prompt_tokens": int(prompt_ids.shape[1]),
                             "prompt": prompt_text, "cell_sentence": cell_sentence})
                if i == 0:
                    pre = diag.get("pre_clamp", [])
                    print(f"  clamp={cv:.2f}: feat{feature_id} pre-clamp "
                          f"mean={np.mean(pre):.4f} "
                          f"pct_active={np.mean([v>0 for v in pre]):.3f} "
                          f"sample[:8]={genes[:8]}")
            results[key] = seqs

        pos_seqs = []
        for i in tqdm(range(N_GENERATIONS), desc="pos_ctrl", leave=False):
            prompt_ids, prompt_text = _make_prompt(tokenizer, pos_cell, template_idx=i)
            genes, cell_sentence = run_generation(model, tokenizer, sae, prompt_ids,
                                                  feature_id=feature_id, clamp_val=0.0,
                                                  seed=SEED_BASE + i, use_hook=False)
            pos_seqs.append({"genes": genes, "n_genes": len(genes),
                             "n_prompt_tokens": int(prompt_ids.shape[1]),
                             "prompt": prompt_text, "cell_sentence": cell_sentence})
        results["positive_control"] = pos_seqs
        print(f"  pos_ctrl sample[:8]={pos_seqs[0]['genes'][:8]}")

        with open(cache_file, "w") as f:
            json.dump(results, f)
        print(f"[cache] saved to {cache_file}")

    # ---- score ------------------------------------------------------------
    rows = []
    for cv in CLAMP_VALUES:
        key = str(cv)
        seqs = results[key]
        scores = [score_genes(e["genes"], target_genes, TOP_K_SCORE) for e in seqs]
        rows.append({
            "clamp": cv,
            "mean_score": np.mean(scores),
            "std_score": np.std(scores),
            "fold": np.mean(scores) / max(background_rate, 1e-8),
            "n_valid": sum(1 for e in seqs if len(e["genes"]) >= 3),
        })
    df = pd.DataFrame(rows)

    neg_scores = [score_genes(e["genes"], target_genes, TOP_K_SCORE)
                  for e in results[str(CLAMP_VALUES[0])]]
    pos_scores = [score_genes(e["genes"], target_genes, TOP_K_SCORE)
                  for e in results.get("positive_control", [])]
    neg_mean = np.mean(neg_scores)
    pos_mean = np.mean(pos_scores) if pos_scores else None

    print(f"\n=== Baseline Scores (feat={feature_id}) ===")
    print(f"  negative ctrl ({neg_cell}, clamp=0.0): {neg_mean:.3f}")
    if pos_mean is not None:
        print(f"  positive ctrl ({pos_cell}, no hook): {pos_mean:.3f}")
    print("\n=== Steering Results ===")
    print(df.to_string(index=False))
    df.to_csv(os.path.join(out_dir, "scores.csv"), index=False)

    # ---- plots ------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    # ax.errorbar(df["clamp"], df["mean_score"], yerr=df["std_score"],
    #             fmt="o-", capsize=4, color="steelblue", label=f"steered ({neg_cell})")
    ax.errorbar(df["clamp"], df["mean_score"],
                fmt="o-", capsize=4, color="steelblue", label=f"steered ({neg_cell})")
    # ax.axhline(background_rate, color="gray", ls="--", lw=1.2,
    #            label=f"background ({background_rate:.1%})")
    ax.axhline(neg_mean, color="tomato", ls=":", lw=1.5,
               label=f"neg ctrl: {neg_cell} ({neg_mean:.3f})")
    if pos_mean is not None:
        ax.axhline(pos_mean, color="green", ls=":", lw=1.5,
                   label=f"pos ctrl: {pos_cell} ({pos_mean:.3f})")
    ax.set_xlabel("SAE clamp value (normalized units, 1.0 = natural max)")
    ax.set_ylabel(f"Fraction top-{TOP_K_SCORE} in target genes")
    ax.set_title("A. Enrichment vs. clamp value")
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.bar(df["clamp"].astype(str), df["fold"], color="steelblue", alpha=0.8)
    # ax.axhline(1.0, color="gray", ls="--", lw=1.2, label="no enrichment")
    ax.axhline(neg_mean / max(background_rate, 1e-8), color="tomato", ls=":", lw=1.5, label="neg ctrl")
    if pos_mean is not None:
        ax.axhline(pos_mean / max(background_rate, 1e-8), color="green", ls=":", lw=1.5, label="pos ctrl")
    ax.set_xlabel("SAE clamp value")
    ax.set_ylabel("Fold enrichment over background")
    ax.set_title("B. Fold enrichment")
    ax.legend(fontsize=7)

    ax = axes[2]
    cv_max = max(CLAMP_VALUES)
    freq_base  = Counter(g for e in results[str(CLAMP_VALUES[0])]
                         for g in e["genes"][:TOP_K_SCORE])
    freq_steer = Counter(g for e in results[str(cv_max)]
                         for g in e["genes"][:TOP_K_SCORE])
    all_genes_seen = set(freq_base) | set(freq_steer)
    diffs = {g: freq_steer.get(g, 0) - freq_base.get(g, 0) for g in all_genes_seen}
    top_gained = sorted(diffs, key=diffs.get, reverse=True)[:15]
    top_lost   = sorted(diffs, key=diffs.get)[:10]
    g_list = top_lost + top_gained
    d_list = [diffs[g] for g in g_list]
    colors = ["tomato" if d < 0 else "steelblue" for d in d_list]
    ax.barh(g_list, d_list, color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel(f"Freq change (clamp={cv_max} vs baseline)")
    ax.set_title(f"C. Gene freq shift (top-{TOP_K_SCORE})")
    for i, g in enumerate(g_list):
        if g in target_genes:
            ax.text(0, i, " ★", va="center", fontsize=7, color="darkgreen")

    go_label = go_terms[0][1] if go_terms else "N/A"
    fig.suptitle(
        f"SAE feature {feature_id} | up={pos_cell} | down={neg_cell}\n"
        f"top GO: {go_label} | C2S-Gemma-2-2B | layer {SAE_LAYER}",
        fontsize=10
    )
    fig.tight_layout()
    out_fig = os.path.join(out_dir, "steering_results.pdf")
    fig.savefig(out_fig, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved -> {out_fig}")

    return {
        "feature_id": feature_id,
        "pos_cell": pos_cell,
        "neg_cell": neg_cell,
        "neg_mean": float(neg_mean),
        "pos_mean": float(pos_mean) if pos_mean is not None else None,
        "background_rate": float(background_rate),
        "scores": df.to_dict(orient="records"),
    }

# %%

# ---------------------------------------------------------------------------
# 7. Sweep
# ---------------------------------------------------------------------------
os.makedirs(BASE_OUT_DIR, exist_ok=True)

with open(PAIRS_PATH) as f:
    pairs = json.load(f)
print(f"[sweep] {len(pairs)} pairs loaded from {PAIRS_PATH}")

model, tokenizer = load_model(MODEL_PATH, DEVICE)
sae = load_sae(SAE_PATH, DEVICE)

# v_results = verify_hook(model, tokenizer, sae, feature_id=pairs[0]["latent"])

# %%
summary_path = os.path.join(BASE_OUT_DIR, "sweep_summary.json")
if os.path.exists(summary_path):
    with open(summary_path) as f:
        summary = json.load(f)
    done_feats = {r["feature_id"] for r in summary}
    print(f"[sweep] resuming — {len(done_feats)} pairs already done: {sorted(done_feats)}")
else:
    summary = []
    done_feats = set()

for pair in pairs:
    # if pair["latent"] in done_feats:
    #     print(f"[skip] feat={pair['latent']} already in summary, skipping.")
    #     continue
    result = run_pair(pair, model, tokenizer, sae)
    if result is not None:
        summary.append(result)
        done_feats.add(pair["latent"])
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

print(f"\n[sweep] done. {len(summary)} pairs completed. Summary -> {summary_path}")

# %%
