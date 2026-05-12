"""
Design A — Steered autoregressive cell sentence generation with C2S-Scale.

Registers forward hooks on GPT-NeoX transformer layers during model.generate(),
injecting alpha * probe_dir at the last token position every decoding step.
Compares generated gene sequences with/without steering to measure
CEBPA GO-term gene enrichment in the top-K generated genes.

Requires:
  - scripts/test_lp_c2s_acts.py     : activations + concept matrix
  - scripts/test_lp_c2s_training.py : trained probe at each PROBE_LAYER
"""
# %% ── Config ─────────────────────────────────────────────────────────────────
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_PROJECT_ROOT)

PROBE_PATH_TMPL = "runs/lp_c2s_norman/linear_probes/probe_layer_{layer}.pt"
CONCEPT_CSV   = "runs/lp_c2s_norman/gprofiler/concept_matrix.csv"
C2S_MODEL     = "vandijklab/C2S-Pythia-410m-cell-type-conditioned-cell-generation"
CACHE_DIR     = "/maiziezhou_lab2/zihang/interpTFM/cache/c2s"
OUT_DIR       = "runs/lp_c2s_norman/eval_generation"
DEVICE        = "cuda"

GENE_SELECT    = "CEBPA"
PROBE_LAYERS   = [12, 17]       # both probes injected simultaneously
CELL_TYPE      = "Type II alveolar epithelial cells"         # Norman dataset cell line
ORGANISM       = "Homo sapiens"

SCALE_LIST     = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
N_GENERATIONS  = 20             # per alpha; temperature sampling for diversity
MAX_NEW_TOKENS = 128            # gene tokens to generate (top-K genes for enrichment)
TOP_K_SCORE    = 50             # enrichment scored on top-K generated genes
TEMPERATURE    = 1.0
DO_SAMPLE      = True
SEED_BASE      = 42
STEERING_MODE  = "amplify"      # amplify scales each direction's existing projection

# %% ── 1. Imports and probe/concept setup ────────────────────────────────────
import json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from interp_pipeline.adapters.model_base import ModelSpec
from interp_pipeline.adapters.models.c2s_scale import C2SScaleAdapter
from interp_pipeline.linear_probe import load_probe

concept_df = pd.read_csv(CONCEPT_CSV, index_col=0)
print(f"Concept matrix: {concept_df.shape[0]} terms × {concept_df.shape[1]} genes")

if GENE_SELECT not in concept_df.columns:
    raise ValueError(
        f"'{GENE_SELECT}' not in concept matrix. Sample cols: {list(concept_df.columns[:10])}"
    )

gene_col          = concept_df[GENE_SELECT]
concept_idx_union = list(np.where(gene_col.values == 1)[0])
concept_names     = list(concept_df.index[concept_idx_union])
print(f"{GENE_SELECT}: {len(concept_idx_union)} GO terms  sample: {concept_names[:5]}")

# All genes that share any GO term with CEBPA — the enrichment target set.
cebpa_go_genes = set(
    concept_df.columns[(concept_df.iloc[concept_idx_union] == 1).any(axis=0)]
)
print(f"CEBPA GO gene set: {len(cebpa_go_genes)} genes")

# Background enrichment rate: fraction of the total gene vocabulary that is in CEBPA GO set.
total_genes     = len(concept_df.columns)
background_rate = len(cebpa_go_genes) / total_genes
print(f"Background rate: {background_rate:.4f} ({len(cebpa_go_genes)}/{total_genes})")

# %% ── 2. Load probes ─────────────────────────────────────────────────────────

probe_dirs_list: list[torch.Tensor] = []
for pl in PROBE_LAYERS:
    _probe = load_probe(PROBE_PATH_TMPL.format(layer=pl), device=DEVICE)
    _dirs  = _probe.weight[:, concept_idx_union].detach().to(DEVICE)  # [H, n_G_concepts]
    probe_dirs_list.append(_dirs)
    print(f"  Layer {pl}: dirs={_dirs.shape}  n_concepts={_dirs.shape[1]}")

# %% ── 3. Load C2S model ──────────────────────────────────────────────────────

adapter  = C2SScaleAdapter()
handle   = adapter.load(ModelSpec(
    name="c2s-scale",
    checkpoint=C2S_MODEL,
    device=DEVICE,
    options={"max_genes": MAX_NEW_TOKENS, "cache_dir": CACHE_DIR},
))
hf_model     = handle.model._model          # GPTNeoXForCausalLM
hf_tokenizer = handle.tokenizer.hf_tokenizer
n_layers     = len(hf_model.gpt_neox.layers)
H            = hf_model.config.hidden_size
print(f"Model loaded — {n_layers} transformer layers, H={H}")
assert all(pl < n_layers for pl in PROBE_LAYERS), \
    f"All PROBE_LAYERS must be < {n_layers}"

# %% ── 4. Build generation prompt ────────────────────────────────────────────

# Fixed template from C2S cell_type_generation prompts.
GEN_PROMPT = (
    f"Generate a list of {MAX_NEW_TOKENS} genes in order of descending expression "
    f"which represent a {ORGANISM} cell of cell type {CELL_TYPE}.\nCell sentence:"
)
print(f"\nGeneration prompt:\n  {GEN_PROMPT}")

prompt_ids = hf_tokenizer(GEN_PROMPT, return_tensors="pt")["input_ids"].to(DEVICE)
print(f"Prompt length: {prompt_ids.shape[1]} tokens")

# %% ── 5. Generation helper with steering hooks ───────────────────────────────

def _make_hook(dirs: torch.Tensor, alpha: float, prompt_len: int, mode: str = "amplify"):
    """
    Forward hook that steers each newly generated gene token.

    GPT-NeoX generate() runs full-sequence forward passes (no KV-cache):
      pass 1: T = prompt_len  (initial prefill — first token predicted)
      pass 2: T = prompt_len + 1  (first gene token in context)
      pass k: T = prompt_len + k - 1  (k-th gene in context)

    We skip the initial prefill (T == prompt_len) because the prompt's last
    token (":") has very large spurious projections onto probe directions.
    Injection at T > prompt_len targets each generated gene's hidden state,
    which is geometrically meaningful for the probe directions trained on
    cell-sentence activations.

    Injection always applies to [:, -1, :] — the most recently added token.
    """
    dirs_n = dirs / dirs.norm(dim=0, keepdim=True).clamp(min=1e-8)  # [H, n_dirs]

    def hook(module, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        if hs.shape[1] <= prompt_len:
            # Initial prefill — skip to preserve first-token prediction.
            return out
        if mode == "add":
            unit_dir = dirs_n.sum(dim=1)
            unit_dir = unit_dir / unit_dir.norm().clamp(min=1e-8)
            hs[:, -1] = hs[:, -1] + alpha * unit_dir
        else:
            # Amplify: scale existing projections onto each concept direction.
            proj = hs[:, -1] @ dirs_n              # [B, n_dirs]
            hs[:, -1] = hs[:, -1] + alpha * (proj @ dirs_n.T)
        if isinstance(out, tuple):
            return (hs,) + out[1:]
        return hs

    return hook


def run_generation(alpha: float, seed: int, mode: str = "amplify") -> list[str]:
    """
    Generate one cell sentence with/without steering.
    Returns the decoded gene tokens as a list of strings.
    """
    hooks = []
    if alpha != 0.0:
        _plen = prompt_ids.shape[1]
        for pl, dirs in zip(PROBE_LAYERS, probe_dirs_list):
            h = hf_model.gpt_neox.layers[pl].register_forward_hook(
                _make_hook(dirs, alpha, prompt_len=_plen, mode=mode)
            )
            hooks.append(h)

    torch.manual_seed(seed)
    with torch.no_grad():
        out_ids = hf_model.generate(
            prompt_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
            pad_token_id=hf_tokenizer.pad_token_id,
            eos_token_id=hf_tokenizer.eos_token_id,
        )

    for h in hooks:
        h.remove()

    new_ids   = out_ids[0, prompt_ids.shape[1]:]
    generated = hf_tokenizer.decode(new_ids, skip_special_tokens=True).strip().rstrip(".")
    return generated.split()


# %% ── 6. Run all generations (cached) ───────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)
_layer_tag  = "l" + "_".join(str(pl) for pl in PROBE_LAYERS)
cache_path  = os.path.join(
    OUT_DIR,
    f"generations_{GENE_SELECT}_{_layer_tag}_{STEERING_MODE}_n{N_GENERATIONS}.json",
)

if os.path.exists(cache_path):
    print(f"\n[6] Loading cached generations from {cache_path}")
    with open(cache_path) as f:
        _raw = json.load(f)
    all_generations: dict[float, list[list[str]]] = {float(k): v for k, v in _raw.items()}
else:
    print(f"\n[6] Running generations (N={N_GENERATIONS} per alpha, mode={STEERING_MODE})...")
    all_generations = {}

    for alpha in SCALE_LIST:
        seqs = []
        for i in tqdm(range(N_GENERATIONS), desc=f"alpha={alpha:+.2f}"):
            genes = run_generation(alpha, seed=SEED_BASE + i, mode=STEERING_MODE)
            seqs.append(genes)
        all_generations[alpha] = seqs
        sample = seqs[0][:10]
        print(f"  alpha={alpha:+.2f}  sample[0]: {sample}")

    with open(cache_path, "w") as f:
        json.dump({str(k): v for k, v in all_generations.items()}, f, indent=2)
    print(f"  Saved: {cache_path}")

# %% ── 7. Score enrichment ────────────────────────────────────────────────────

print(f"\n[7] Enrichment of CEBPA GO genes in top-{TOP_K_SCORE} generated genes:")
print(f"    Background rate: {background_rate:.4f}")

records = []
per_gen_records = []

for alpha in sorted(all_generations.keys()):
    seqs = all_generations[alpha]
    enrich_scores = []
    for genes in seqs:
        top_k = genes[:TOP_K_SCORE]
        n_hit = sum(1 for g in top_k if g in cebpa_go_genes)
        score = n_hit / max(len(top_k), 1)
        enrich_scores.append(score)
        per_gen_records.append({"alpha": alpha, "enrichment": score, "n_hit": n_hit,
                                 "n_top_k": len(top_k)})

    mean_e = float(np.mean(enrich_scores))
    std_e  = float(np.std(enrich_scores))
    fold   = mean_e / max(background_rate, 1e-8)
    records.append({"alpha": alpha, "mean": mean_e, "std": std_e, "fold": fold})
    print(f"  alpha={alpha:+.2f}  mean={mean_e:.4f} ± {std_e:.4f}  fold={fold:.2f}x  n={len(seqs)}")

df_scores  = pd.DataFrame(records)
df_per_gen = pd.DataFrame(per_gen_records)

scores_path = os.path.join(OUT_DIR, f"enrichment_scores_{GENE_SELECT}_{_layer_tag}_{STEERING_MODE}.csv")
df_scores.to_csv(scores_path, index=False)
print(f"  Saved scores: {scores_path}")

# %% ── 8. Plots ───────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import seaborn as sns

print("\n[8] Plotting...")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 8a. Mean enrichment ± std ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.errorbar(df_scores["alpha"], df_scores["mean"], yerr=df_scores["std"],
            marker="o", capsize=4, linewidth=1.5, color="steelblue")
ax.axhline(background_rate, color="red", linestyle="--",
           label=f"Background ({background_rate:.3f})")
ax.set_xlabel("Steering alpha")
ax.set_ylabel(f"Fraction of top-{TOP_K_SCORE} genes in {GENE_SELECT} GO set")
ax.set_title(f"Steered generation enrichment — {GENE_SELECT} (mode={STEERING_MODE})")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.bar(df_scores["alpha"], df_scores["fold"], color="steelblue", alpha=0.7)
ax.axhline(1.0, color="red", linestyle="--", label="Background (1×)")
ax.set_xlabel("Steering alpha")
ax.set_ylabel("Fold enrichment over background")
ax.set_title(f"Fold enrichment — {GENE_SELECT} GO genes")
ax.legend()
ax.grid(True, alpha=0.3)

fig.tight_layout()
plot_path = os.path.join(OUT_DIR, f"enrichment_plot_{GENE_SELECT}_{_layer_tag}_{STEERING_MODE}.png")
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {plot_path}")

# ── 8b. Violin plot of per-generation enrichment distribution ────────────────
fig2, ax2 = plt.subplots(figsize=(10, 5))
sns.violinplot(
    data=df_per_gen,
    x="alpha",
    y="enrichment",
    ax=ax2,
    inner="box",
)
ax2.axhline(background_rate, color="red", linestyle="--",
            label=f"Background ({background_rate:.3f})")
ax2.set_xlabel("Steering alpha")
ax2.set_ylabel(f"Fraction of top-{TOP_K_SCORE} genes in {GENE_SELECT} GO set")
ax2.set_title(f"Per-generation enrichment distribution — {GENE_SELECT}")
ax2.legend()
violin_path = os.path.join(OUT_DIR, f"enrichment_violin_{GENE_SELECT}_{_layer_tag}_{STEERING_MODE}.png")
fig2.savefig(violin_path, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"  Saved: {violin_path}")

# ── 8c. Top genes appearing more often when steered ──────────────────────────
# Compare how often CEBPA GO genes appear in top-K for alpha=0 vs max alpha.
_alpha_0   = 0.0
_alpha_max = max(all_generations.keys())

def _gene_freq(seqs, top_k=TOP_K_SCORE):
    """Count frequency of each gene appearing in top-K across all sequences."""
    from collections import Counter
    c = Counter()
    for genes in seqs:
        c.update(genes[:top_k])
    total = len(seqs)
    return {g: cnt / total for g, cnt in c.items()}

freq_0   = _gene_freq(all_generations[_alpha_0])
freq_max = _gene_freq(all_generations[_alpha_max])

# CEBPA GO genes with largest frequency increase.
go_genes_in_both = cebpa_go_genes & (set(freq_0.keys()) | set(freq_max.keys()))
delta = {
    g: freq_max.get(g, 0.0) - freq_0.get(g, 0.0)
    for g in go_genes_in_both
}
top_enriched = sorted(delta.items(), key=lambda x: -x[1])[:20]
print(f"\n  Top CEBPA GO genes most enriched by steering (alpha={_alpha_max:+.1f} vs 0):")
for g, d in top_enriched:
    print(f"    {g:12s}  freq_0={freq_0.get(g, 0):.3f}  freq_max={freq_max.get(g, 0):.3f}  Δ={d:+.3f}")

# Save frequency table.
_max_col = f"freq_alpha{_alpha_max}"
freq_df = pd.DataFrame({
    "gene": sorted(go_genes_in_both),
    "freq_alpha0.0": [freq_0.get(g, 0.0) for g in sorted(go_genes_in_both)],
    _max_col: [freq_max.get(g, 0.0) for g in sorted(go_genes_in_both)],
    "delta": [freq_max.get(g, 0.0) - freq_0.get(g, 0.0) for g in sorted(go_genes_in_both)],
    "in_cebpa_go": [g in cebpa_go_genes for g in sorted(go_genes_in_both)],
})
freq_path = os.path.join(OUT_DIR, f"gene_freq_{GENE_SELECT}_{_layer_tag}_{STEERING_MODE}.csv")
freq_df.to_csv(freq_path, index=False)
print(f"  Saved frequency table: {freq_path}")

print("\nDone.")

# %%