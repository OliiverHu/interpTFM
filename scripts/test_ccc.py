from __future__ import annotations

import os
import numpy as np
import pandas as pd
import scanpy as sc

from interp_pipeline.downstream.interaction.edges import EdgeConfig, build_edges_radius, distance_weights, tile_ids
from interp_pipeline.downstream.interaction.grouping import encode_groups
from interp_pipeline.downstream.interaction.preprocess import preprocess_variant_b, PreprocessBConfig
from interp_pipeline.downstream.interaction.aggregate import AggConfig, aggregate_S1_S2_chunked, null_tile_shuffle_S1
from interp_pipeline.downstream.interaction.score import neff_from_s1_s2, exposure_normalize, apply_null_std_floor
from interp_pipeline.downstream.interaction.report import pair_table, decode_pair
from interp_pipeline.downstream.interaction.plot import heatmap_masked, vec_to_square
from interp_pipeline.downstream.interaction.adjacency import adjacency_counts, adjacency_null, adjacency_z_and_fdr


# =========================
# CONFIG (match notebook)
# =========================
INTERP_H5AD = "runs/full_scgpt_cosmx/interpretable_adata/layer_4.norm2/sae_thr_0.6/adata_interpretable_layer_4.norm2_saeThr0.6_bestonly_conceptdedup_f1cut0.6_topall.h5ad"
NICHE_LABELS_CSV = "runs/full_scgpt_cosmx/niche_discovery/layer_4.norm2/validation_k3_r120_m/global_labels.csv"

CELLTYPE_KEY = "author_cell_type"
SPATIAL_KEY = "spatial"

R = 138.0
TILE_SIZE = 400.0

WEIGHT_MODE = "exp"
SIGMA_FRAC_OF_R = 0.7
EPS_W = 1e-3

SEED = 0
N_PERM_ADJ = 100
N_PERM_CROSSTALK = 50
CHUNK_SIZE = 40_000

EPS = 1e-6
MIN_NEFF = 50.0
MIN_PAIR_EDGES = 1000
MIN_PAIR_WSUM = 5.0
TOPK_INTENSITY = 10

B_CLIP = 10.0
MAD_EPS = 1e-6

ALPHA_FDR = 0.05
KEEP_PROP = 0.005   # plot-only

OUTDIR = "runs/full_scgpt_cosmx/ccc_interdomain/layer_4.norm2"
# =========================


def attach_niche_labels(adata: sc.AnnData, niche_csv: str) -> np.ndarray:
    df = pd.read_csv(niche_csv)
    m = dict(zip(df["cell"].astype(str), df["niche"].astype(int)))
    out = np.array([m.get(x, -1) for x in adata.obs_names.astype(str)], dtype=int)
    if (out < 0).any():
        # keep them but they won't appear in niche subsets
        pass
    return out


def keep_types_for_plot(celltype_labels: np.ndarray, group_names: np.ndarray) -> tuple[set[str], np.ndarray]:
    ct_counts = pd.Series(celltype_labels.astype(str)).value_counts()
    ct_prop = ct_counts / ct_counts.sum()
    keep_names = ct_prop[ct_prop >= KEEP_PROP].index.astype(str).tolist()
    keep_set = set(keep_names)
    keep_idx = np.array([i for i, n in enumerate(group_names.tolist()) if n in keep_set], dtype=int)
    return keep_set, keep_idx


def subset_square(vec_or_mat, keep_idx, G):
    arr = np.asarray(vec_or_mat)
    if arr.ndim == 1:
        arr = arr.reshape(int(G), int(G))
    return arr[np.ix_(keep_idx, keep_idx)]


def run_one_domain(domain_name: str, coords: np.ndarray, X: np.ndarray, celltype_labels: np.ndarray) -> dict:
    print(f"\n==== [{domain_name}] r={R:.1f} tile={TILE_SIZE:.1f} ====")

    cell_codes, cell_names = encode_groups(pd.Series(celltype_labels.astype(str)))
    cell_names = np.asarray(cell_names, dtype=str)
    G = int(len(cell_names))

    tiles = tile_ids(coords, tile_size=TILE_SIZE)

    sigma = float(max(1.0, SIGMA_FRAC_OF_R * R))
    ei, ej, ed = build_edges_radius(coords, R)
    w = distance_weights(ed, EdgeConfig(radius=R, weight_mode=WEIGHT_MODE, sigma=sigma, eps=EPS_W))

    # --- adjacency Z (wsum) ---
    adj_obs = adjacency_counts(ei, ej, w, cell_codes, G, use_weight=True)
    adj_null_mean, adj_null_std = adjacency_null(
        ei, ej, w, cell_codes, tiles, G, n_perm=N_PERM_ADJ, seed=SEED, use_weight=True
    )
    adj_Z, adj_q, adj_sig = adjacency_z_and_fdr(adj_obs, adj_null_mean, adj_null_std, eps=EPS, alpha_fdr=ALPHA_FDR)

    # --- crosstalk Variant B ---
    F_B = preprocess_variant_b(X, PreprocessBConfig(clip=B_CLIP, eps=MAD_EPS))
    S1, S2, edge_count, weight_sum = aggregate_S1_S2_chunked(
        F_B, ei, ej, w, cell_codes, G, AggConfig(chunk_size=CHUNK_SIZE, ordered_pairs=False)
    )
    n_eff = (S1 * S1) / (S2 + EPS)

    S1_norm = exposure_normalize(S1, weight_sum, eps=EPS)

    null_mean, null_std = null_tile_shuffle_S1(
        F_B, ei, ej, w, cell_codes, G, tiles,
        n_perm=N_PERM_CROSSTALK, seed=SEED,
        cfg=AggConfig(chunk_size=CHUNK_SIZE, ordered_pairs=False),
    )
    null_mean_norm = exposure_normalize(null_mean, weight_sum, eps=EPS)
    null_std_norm = exposure_normalize(null_std, weight_sum, eps=EPS)

    null_std_eff = apply_null_std_floor(null_std_norm, floor_q=0.10, eps=EPS)
    Z = (S1_norm - null_mean_norm) / (null_std_eff + EPS)

    Z_stable = Z.copy()
    Z_stable[n_eff < MIN_NEFF] = np.nan

    # intensity = median of top-k positive Z (same as notebook)
    intensity = np.full((G * G,), np.nan, dtype=np.float32)
    for p in range(G * G):
        z = Z_stable[p]
        z = z[np.isfinite(z)]
        pos = z[z > 0]
        if pos.size:
            k = min(TOPK_INTENSITY, pos.size)
            top = np.partition(pos, -k)[-k:]
            intensity[p] = float(np.median(top))

    pair_mask = (edge_count >= MIN_PAIR_EDGES) & (weight_sum >= MIN_PAIR_WSUM)
    intensity_masked = intensity.copy()
    intensity_masked[~pair_mask] = np.nan

    # proxy p/q for intensity (same as notebook)
    from scipy.stats import norm
    inten_p = 2.0 * (1.0 - norm.cdf(np.abs(np.nan_to_num(intensity_masked, nan=0.0))))
    from interp_pipeline.downstream.interaction.adjacency import bh_fdr
    inten_q = bh_fdr(inten_p, alpha=ALPHA_FDR)
    inten_sig = (inten_q < ALPHA_FDR) & np.isfinite(intensity_masked) & pair_mask

    flags = (adj_sig.astype(bool) & inten_sig.astype(bool))

    return dict(
        cell_names=cell_names,
        cell_codes=cell_codes,
        G=G,
        edges=(ei, ej, w),
        adj_Z=adj_Z, adj_q=adj_q, adj_sig=adj_sig,
        intensity=intensity_masked, intensity_q=inten_q, inten_sig=inten_sig,
        flags=flags,
        edge_count=edge_count, weight_sum=weight_sum,
    )


def build_pair_lookup(res: dict) -> dict:
    cn = res["cell_names"]
    G = res["G"]
    out = {}
    for p in range(G * G):
        a, b = decode_pair(p, G)
        if a == b:
            continue
        ga, gb = cn[a], cn[b]
        key = (ga, gb) if ga <= gb else (gb, ga)
        out[key] = dict(
            adj_Z=float(res["adj_Z"][p]),
            adj_q=float(res["adj_q"][p]),
            intensity=float(res["intensity"][p]) if np.isfinite(res["intensity"][p]) else np.nan,
            intensity_q=float(res["intensity_q"][p]) if np.isfinite(res["intensity_q"][p]) else np.nan,
            flag=int(bool(res["flags"][p])),
        )
    return out


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    plot_dir = os.path.join(OUTDIR, "plots")
    tab_dir = os.path.join(OUTDIR, "tables")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)

    print("[1] Load interpretable h5ad")
    adata = sc.read_h5ad(INTERP_H5AD)
    coords_full = np.asarray(adata.obsm[SPATIAL_KEY], dtype=np.float32)

    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    celltype_labels_full = np.asarray(adata.obs[CELLTYPE_KEY]).astype(str)
    niche_labels_full = attach_niche_labels(adata, NICHE_LABELS_CSV)

    # ---- GLOBAL ----
    global_res = run_one_domain("GLOBAL", coords_full, X, celltype_labels_full)

    keep_set_glob, keep_idx_glob = keep_types_for_plot(celltype_labels_full, global_res["cell_names"])

    adjZ_plot = subset_square(global_res["adj_Z"], keep_idx_glob, global_res["G"])
    inten_plot = subset_square(global_res["intensity"], keep_idx_glob, global_res["G"])
    plot_names = global_res["cell_names"][keep_idx_glob].tolist()

    heatmap_masked(
        adjZ_plot,
        title=f"[GLOBAL] Adjacency Z (wsum) r={R:.0f} tile={TILE_SIZE:.0f}",
        names=plot_names,
        outpath=os.path.join(plot_dir, "global_adjacencyZ_maskdiag.png"),
        fmt="{:.1f}",
        mask_diag=True,
    )
    heatmap_masked(
        inten_plot,
        title=f"[GLOBAL] Crosstalk intensity (Variant B) r={R:.0f} tile={TILE_SIZE:.0f}",
        names=plot_names,
        outpath=os.path.join(plot_dir, "global_intensity_maskdiag.png"),
        fmt="{:.2f}",
        mask_diag=True,
    )

    cell_names = global_res["cell_names"].tolist()
    G = global_res["G"]

    adj_df = pair_table(global_res["adj_Z"], cell_names, G, score_name="adj_Z", mask_diag=True)
    adj_df["adj_q"] = global_res["adj_q"][adj_df["pair_id"].values]
    adj_df = adj_df.sort_values("adj_Z", ascending=False)
    adj_df.to_csv(os.path.join(tab_dir, "global_top_pairs_by_adjacencyZ.csv"), index=False)

    inten_df = pair_table(global_res["intensity"], cell_names, G, score_name="intensity", mask_diag=True)
    inten_df["intensity_q"] = global_res["intensity_q"][inten_df["pair_id"].values]
    inten_df = inten_df.sort_values("intensity", ascending=False)
    inten_df.to_csv(os.path.join(tab_dir, "global_top_pairs_by_intensity.csv"), index=False)

    # flagged pairs (adj AND intensity significant)
    pairs_flag = adj_df.merge(inten_df, on=["pair_id", "group_a", "group_b"], how="inner")
    pairs_flag["flag"] = global_res["flags"][pairs_flag["pair_id"].values].astype(int)
    pairs_flag = pairs_flag[pairs_flag["flag"] == 1].sort_values("intensity", ascending=False)
    pairs_flag.to_csv(os.path.join(tab_dir, "global_flagged_pairs.csv"), index=False)

    # ---- NICHES ----
    niche_results = {}
    for k in sorted(np.unique(niche_labels_full[niche_labels_full >= 0]).tolist()):
        idx = np.where(niche_labels_full == k)[0]
        niche_res = run_one_domain(f"NICHE_{k}", coords_full[idx], X[idx], celltype_labels_full[idx])
        niche_results[int(k)] = niche_res

        # niche plots: keep globally common types AND present in niche encoding
        cn_k = niche_res["cell_names"]
        keep_idx_k = np.array([ii for ii, n in enumerate(cn_k.tolist()) if n in keep_set_glob], dtype=int)
        if keep_idx_k.size >= 3:
            adjZ_k_plot = subset_square(niche_res["adj_Z"], keep_idx_k, niche_res["G"])
            inten_k_plot = subset_square(niche_res["intensity"], keep_idx_k, niche_res["G"])
            plot_names_k = cn_k[keep_idx_k].tolist()

            heatmap_masked(
                adjZ_k_plot,
                title=f"[NICHE {k}] Adjacency Z (wsum) r={R:.0f} tile={TILE_SIZE:.0f}",
                names=plot_names_k,
                outpath=os.path.join(plot_dir, f"niche_{k}_adjacencyZ_maskdiag.png"),
                fmt="{:.1f}",
                mask_diag=True,
            )
            heatmap_masked(
                inten_k_plot,
                title=f"[NICHE {k}] Crosstalk intensity (Variant B) r={R:.0f} tile={TILE_SIZE:.0f}",
                names=plot_names_k,
                outpath=os.path.join(plot_dir, f"niche_{k}_intensity_maskdiag.png"),
                fmt="{:.2f}",
                mask_diag=True,
            )

    # ---- Compare niche vs global ----
    glob_lookup = build_pair_lookup(global_res)
    rows = []
    for k, res_k in niche_results.items():
        look_k = build_pair_lookup(res_k)
        for key, g in glob_lookup.items():
            ga, gb = key
            n = look_k.get(key, None)
            rows.append(
                dict(
                    niche=int(k),
                    group_a=ga,
                    group_b=gb,
                    adjZ_global=g["adj_Z"],
                    adjZ_niche=(n["adj_Z"] if n else np.nan),
                    inten_global=g["intensity"],
                    inten_niche=(n["intensity"] if n else np.nan),
                    flag_global=g["flag"],
                    flag_niche=(n["flag"] if n else 0),
                )
            )

    cmp = pd.DataFrame(rows)
    cmp["delta_inten"] = cmp["inten_niche"] - cmp["inten_global"]
    cmp["delta_adjZ"] = cmp["adjZ_niche"] - cmp["adjZ_global"]
    cmp.to_csv(os.path.join(tab_dir, "niche_vs_global_comparison.csv"), index=False)

    print("[OK] wrote:", OUTDIR)


if __name__ == "__main__":
    main()