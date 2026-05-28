"""
Explore the joint structure of SAE feature F1 scores (GO concept alignment) and
cell-type-specific activation patterns.

Goal: find features that both
  (a) have high F1 with some GO biological concept, and
  (b) activate differentially across cell types (high in some, low in others).

Pipeline:
  1. Load high-F1 feature list from f1_analysis/high_f1_features.csv
  2. Load SAE encoder weights for only those candidate latents
  3. Stream through all activation shards, compute per-cell mean latent activation
  4. Map cells to cell types (from prepared cosmx h5ad)
  5. Compute per-cell-type mean activation and a cell-type-variance score
  6. Rank features by F1 × cell-type variance and visualize
"""
# %%

import os
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import anndata as ad
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

# ── Paths ────────────────────────────────────────────────────────────────────
BASE      = Path('/maiziezhou_lab2/yunfei/Projects/interpTFM/runs/full_c2sscale_cosmx')
F1_DIR    = BASE / 'f1_analysis'
ACT_DIR   = BASE / 'activations' / 'layer_17'
SAE_PATH  = BASE / 'sae' / 'layer_17' / 'sae_layer_17_best.pt'
ADATA_PATH = '/maiziezhou_lab2/yunfei/Projects/interpTFM/acts_extraction_geneformer_cosmx/prepared/cosmx.prepared.h5ad'
OUT_DIR   = Path('/maiziezhou_lab2/zihang/interpTFM/runs/f1_celltype_analysis')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Candidate features ─────────────────────────────────────────────────────
print("=== 1. Loading high-F1 features ===")
high_f1 = pd.read_csv(F1_DIR / 'high_f1_features.csv')
print(f"  {len(high_f1)} features with best_f1 >= threshold")
print(high_f1[['latent', 'best_term_name', 'best_f1']].sort_values('best_f1', ascending=False).head(10).to_string(index=False))

candidate_latents = sorted(high_f1['latent'].tolist())
latent_to_idx = {l: i for i, l in enumerate(candidate_latents)}  # latent id → column index

# ── 2. Cell type map ─────────────────────────────────────────────────────────
print("\n=== 2. Loading cell type annotations ===")
adata = ad.read_h5ad(ADATA_PATH)
cell_id_to_type = dict(zip(adata.obs['cell_id'].tolist(), adata.obs['author_cell_type'].tolist()))
print(f"  {adata.n_obs} cells, {len(set(cell_id_to_type.values()))} cell types")
print("  Cell type counts:")
for ct, n in adata.obs['author_cell_type'].value_counts().items():
    print(f"    {ct:30s}  {n:6d}")

# ── 3. Load SAE encoder weights (candidate latents only) ─────────────────────
print("\n=== 3. Loading SAE encoder ===")
sae_ckpt = torch.load(SAE_PATH, map_location='cpu')
state    = sae_ckpt['state_dict']
bias_pre = state['bias']           # (d_in,)  — centering: encode(x) = ReLU(W_enc(x - bias_pre) + b_enc)
W_enc    = state['encoder.weight'] # (n_latents, d_in)
b_enc    = state['encoder.bias']   # (n_latents,)

lat_idx_t  = torch.tensor(candidate_latents, dtype=torch.long)
W_enc_sub  = W_enc[lat_idx_t].float()   # (n_cand, d_in)
b_enc_sub  = b_enc[lat_idx_t].float()   # (n_cand,)
bias_pre_f = bias_pre.float()            # (d_in,)

n_cand = len(candidate_latents)
print(f"  SAE d_in={sae_ckpt['d_in']}, n_latents={sae_ckpt['n_latents']}")
print(f"  Using {n_cand} candidate latents")

# ── 4. Stream shards → per-cell accumulation ──────────────────────────────────
print("\n=== 4. Processing activation shards ===")
cell_sum   = defaultdict(lambda: np.zeros(n_cand, dtype=np.float64))
cell_count = defaultdict(int)

shards = sorted(
    [s for s in ACT_DIR.iterdir() if s.is_dir() and (s / 'activations.pt').exists()],
    key=lambda p: int(p.name.split('_')[1])
)
print(f"  {len(shards)} shards found")

for si, shard_dir in enumerate(shards):
    idx_data = torch.load(shard_dir / 'index.pt',       map_location='cpu')
    acts     = torch.load(shard_dir / 'activations.pt', map_location='cpu').float()

    example_ids = idx_data['example_ids']   # list[str], length = n_tokens

    # SAE encode: ReLU(W_enc (x - bias_pre) + b_enc)
    x_centered  = acts - bias_pre_f.unsqueeze(0)          # (n_tokens, d_in)
    latent_acts = torch.relu(x_centered @ W_enc_sub.T + b_enc_sub)  # (n_tokens, n_cand)
    lat_np      = latent_acts.numpy()

    for j, cell_id in enumerate(example_ids):
        cell_sum[cell_id]   += lat_np[j]
        cell_count[cell_id] += 1

    if (si + 1) % 10 == 0 or si == len(shards) - 1:
        print(f"  [{si+1}/{len(shards)}] processed, {len(cell_sum)} unique cells so far", flush=True)

# ── 5. Per-cell mean activation ───────────────────────────────────────────────
print("\n=== 5. Aggregating per-cell mean activation ===")
cell_ids_list  = list(cell_sum.keys())
cell_means_mat = np.stack(
    [cell_sum[c] / cell_count[c] for c in cell_ids_list]
)  # (n_cells, n_cand)
print(f"  {len(cell_ids_list)} cells processed")

cell_types_list = [cell_id_to_type.get(c, 'unknown') for c in cell_ids_list]
unmatched = sum(1 for ct in cell_types_list if ct == 'unknown')
print(f"  {unmatched} cells without a cell-type match")

df_cells = pd.DataFrame(
    cell_means_mat,
    columns=[f'lat_{l}' for l in candidate_latents]
)
df_cells['cell_id']   = cell_ids_list
df_cells['cell_type'] = cell_types_list
df_cells.to_csv(OUT_DIR / 'per_cell_mean_acts.csv', index=False)
print(f"  Saved per_cell_mean_acts.csv")

# ── 6. Per-cell-type mean and variance ───────────────────────────────────────
print("\n=== 6. Computing per-cell-type statistics ===")
lat_cols = [f'lat_{l}' for l in candidate_latents]
ct_mean  = df_cells.groupby('cell_type')[lat_cols].mean()    # (n_types, n_cand)
ct_std   = df_cells.groupby('cell_type')[lat_cols].std()

# Variance of per-cell-type means (how much the feature differs across types)
ct_var   = ct_mean.var(axis=0)   # Series indexed by lat_col
ct_var.index = candidate_latents  # map back to latent id

ct_mean.to_csv(OUT_DIR / 'celltype_mean_acts.csv')
ct_std.to_csv(OUT_DIR  / 'celltype_std_acts.csv')
print("  Saved celltype_mean/std_acts.csv")

# ── 7. Rank features ─────────────────────────────────────────────────────────
print("\n=== 7. Ranking features ===")
results = high_f1.copy()
results['celltype_var']   = results['latent'].map(ct_var.to_dict())
# normalized score: F1 * sqrt(celltype_var)  – weights both axes
results['score'] = results['best_f1'] * np.sqrt(results['celltype_var'])
results = results.sort_values('score', ascending=False).reset_index(drop=True)

top20 = results.head(20)
print("\nTop 20 features (high F1 AND high cell-type variance):")
print(top20[['latent', 'best_term_name', 'best_f1', 'celltype_var', 'score']].to_string(index=False))

results.to_csv(OUT_DIR / 'f1_celltype_scores.csv', index=False)
print(f"\nSaved f1_celltype_scores.csv")

# ── 8. Visualizations ────────────────────────────────────────────────────────
print("\n=== 8. Generating plots ===")

# ── 8a. Scatter: F1 vs cell-type variance ─────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(results['best_f1'], results['celltype_var'], alpha=0.6, s=20)
for _, row in results.head(10).iterrows():
    ax.annotate(
        f"{int(row['latent'])}: {row['best_term_name'][:30]}",
        (row['best_f1'], row['celltype_var']),
        fontsize=6, xytext=(4, 4), textcoords='offset points'
    )
ax.set_xlabel('Best F1 (GO concept alignment)')
ax.set_ylabel('Cell-type variance of mean activation')
ax.set_title('SAE features: GO alignment vs cell-type specificity')
plt.tight_layout()
plt.savefig(OUT_DIR / 'f1_vs_celltype_var_scatter.png', dpi=150)
plt.close()
print("  Saved f1_vs_celltype_var_scatter.png")

# ── 8b. Heatmap of top-N features × cell types ───────────────────────────
top_n  = 30
top_df = results.head(top_n)
top_cols = [f'lat_{l}' for l in top_df['latent'].tolist()]

heatmap_df = ct_mean[top_cols].copy()
heatmap_df.columns = [
    f"{row['latent']}: {row['best_term_name'][:35]}"
    for _, row in top_df.iterrows()
]

# Z-score across cell types for each feature (highlight relative differences)
heatmap_z = heatmap_df.apply(lambda c: (c - c.mean()) / (c.std() + 1e-8), axis=0)

fig, ax = plt.subplots(figsize=(max(14, top_n * 0.5), 8))
sns.heatmap(
    heatmap_z,
    ax=ax,
    cmap='RdBu_r',
    center=0,
    linewidths=0.3,
    xticklabels=True,
    yticklabels=True,
    cbar_kws={'label': 'Z-score (mean act. across cell types)'}
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=7)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)
ax.set_title(f'Top {top_n} features (F1 × cell-type specificity): cell-type activation heatmap')
plt.tight_layout()
plt.savefig(OUT_DIR / 'top_features_celltype_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved top_features_celltype_heatmap.png")

# ── 8c. Bar chart grid: top-20 features, activation per cell type ─────────
ncols = 4
nrows = (min(top_n, 20) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3.5))
axes = axes.flatten()

for i, (_, row) in enumerate(results.head(20).iterrows()):
    latent = row['latent']
    col    = f'lat_{latent}'
    vals   = ct_mean[col].sort_values(ascending=False)
    ax     = axes[i]
    bars   = ax.bar(range(len(vals)), vals.values, color='steelblue', alpha=0.8)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(vals.index, rotation=55, ha='right', fontsize=6)
    ax.set_title(
        f"Latent {latent}\n{row['best_term_name'][:45]}\nF1={row['best_f1']:.3f}  var={row['celltype_var']:.4f}",
        fontsize=7
    )
    ax.set_ylabel('Mean activation', fontsize=7)
    ax.tick_params(axis='y', labelsize=6)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Top-20 features: mean SAE latent activation by cell type', fontsize=10, y=1.01)
plt.tight_layout()
plt.savefig(OUT_DIR / 'top20_features_celltype_bars.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved top20_features_celltype_bars.png")

print(f"\nDone. All outputs in: {OUT_DIR}")

# %%
# ── 9. Best feature × cell-type pair (non-tumor, top-20 only) ────────────────
print("\n=== 9. Best high-vs-low cell-type pair (non-tumor, top-20 features) ===")

non_tumor_cts = [ct for ct in ct_mean.index if 'tumor' not in ct]
top20_results = results.head(20)

pairs = []
for _, row in top20_results.iterrows():
    lat  = row['latent']
    col  = f'lat_{lat}'
    vals = ct_mean.loc[non_tumor_cts, col].sort_values(ascending=False)
    top_ct  = vals.index[0];  top_val = vals.iloc[0]
    bot_ct  = vals.index[-1]; bot_val = vals.iloc[-1]
    pairs.append({
        'latent': lat,
        'best_term_name': row['best_term_name'],
        'best_f1': row['best_f1'],
        'high_ct': top_ct, 'high_val': top_val,
        'low_ct':  bot_ct, 'low_val':  bot_val,
        'fold':    (top_val + 1e-6) / (bot_val + 1e-6),
        'abs_diff': top_val - bot_val,
    })

pairs_df = pd.DataFrame(pairs).sort_values('abs_diff', ascending=False)
print("\nTop-20 features ranked by absolute activation difference (no tumor cell types):")
print(pairs_df[['latent','best_term_name','best_f1','high_ct','high_val','low_ct','low_val','fold','abs_diff']].to_string(index=False))
pairs_df.to_csv(OUT_DIR / 'top20_nontumor_pairs.csv', index=False)
print("  Saved top20_nontumor_pairs.csv")

# Pick the best feature: highest absolute difference
best = pairs_df.iloc[0]
best_lat   = int(best['latent'])
best_col   = f'lat_{best_lat}'
best_high  = best['high_ct']
best_low   = best['low_ct']
best_term  = best['best_term_name']
best_f1    = best['best_f1']

print(f"\nBest feature: Latent {best_lat} — {best_term}")
print(f"  F1 = {best_f1:.3f}")
print(f"  HIGH: {best_high} = {best['high_val']:.4f}")
print(f"  LOW:  {best_low}  = {best['low_val']:.4f}")
print(f"  Fold = {best['fold']:.2f}x,  Abs diff = {best['abs_diff']:.4f}")

# ── 9a. Per-cell violin plot for the best feature ────────────────────────────
df_best = df_cells[df_cells['cell_type'].isin(non_tumor_cts)][['cell_type', best_col]].copy()
df_best = df_best.rename(columns={best_col: 'activation'})

# Order cell types by median activation descending
ct_order = (
    df_best.groupby('cell_type')['activation']
    .median()
    .sort_values(ascending=False)
    .index.tolist()
)

palette = {ct: ('#e05c5c' if ct == best_high else
                '#5c7be0' if ct == best_low  else
                '#aaaaaa')
           for ct in ct_order}

fig, ax = plt.subplots(figsize=(12, 5))
sns.violinplot(
    data=df_best, x='cell_type', y='activation',
    order=ct_order, hue='cell_type', palette=palette, legend=False,
    inner='box', cut=0, ax=ax
)
ax.set_xticks(range(len(ct_order)))
ax.set_xticklabels(ct_order, rotation=40, ha='right', fontsize=9)
ax.set_xlabel('')
ax.set_ylabel('Mean SAE latent activation (per cell)', fontsize=10)
ax.set_title(
    f'Latent {best_lat}: {best_term}\n'
    f'F1={best_f1:.3f}  |  {best_high} (red) vs {best_low} (blue)',
    fontsize=10
)
plt.tight_layout()
plt.savefig(OUT_DIR / f'best_feature_lat{best_lat}_violin.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved best_feature_lat{best_lat}_violin.png")

# ── 9b. Bar chart — mean ± std for all non-tumor cell types ──────────────────
ct_stats = df_best.groupby('cell_type')['activation'].agg(['mean', 'std']).loc[ct_order]

colors = ['#e05c5c' if ct == best_high else
          '#5c7be0' if ct == best_low  else
          '#aaaaaa'
          for ct in ct_order]

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(range(len(ct_order)), ct_stats['mean'], yerr=ct_stats['std'],
       color=colors, alpha=0.85, capsize=4, error_kw={'linewidth': 1.2})
ax.set_xticks(range(len(ct_order)))
ax.set_xticklabels(ct_order, rotation=40, ha='right', fontsize=9)
ax.set_ylabel('Mean SAE latent activation ± std', fontsize=10)
ax.set_title(
    f'Latent {best_lat}: {best_term}\n'
    f'F1={best_f1:.3f}  |  {best_high} (red) vs {best_low} (blue)',
    fontsize=10
)
plt.tight_layout()
plt.savefig(OUT_DIR / f'best_feature_lat{best_lat}_bar.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved best_feature_lat{best_lat}_bar.png")

# %%