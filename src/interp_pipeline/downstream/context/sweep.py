from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .neighbors import NeighborConfig
from .microenv import microenv_embedding
from .cluster import ClusterConfig, build_cluster_matrix, pca_embed, cluster_labels
from .metrics import (
    stability_ari_over_seeds,
    mean_neighbor_purity,
    tile_shuffle_null_purity,
    separability_cosine,
    min_cluster_size,
    fragmentation_score_from_neigh,
)


@dataclass(frozen=True)
class SweepConfig:
    radii: List[float]
    n_clusters_list: List[int]
    method_list: List[str]          # ["kmeans","gmm"]
    space_list: List[str]           # ["m","xm"]
    kernel: str = "uniform"         # uniform/gaussian
    sigma_frac: float = 0.25

    pca_components: int = 50
    seeds: List[int] = (0, 1, 2, 3, 4)

    min_cluster_size: int = 150

    # null model
    tile_mult: float = 4.0
    n_null: int = 20
    null_seed: int = 0

    # composite score weights (same spirit as notebook)
    w_stability: float = 1.5
    w_purityz: float = 0.1
    w_sep: float = 1.0
    purityz_clip: float = 10.0


def run_niche_sweep(
    X: np.ndarray,
    coords: np.ndarray,
    cfg: SweepConfig,
) -> pd.DataFrame:
    rows: List[Dict] = []

    for r in cfg.radii:
        ncfg = NeighborConfig(radius=float(r), kernel=cfg.kernel, sigma_frac=cfg.sigma_frac)
        m, neigh = microenv_embedding(X, coords, ncfg)

        for space in cfg.space_list:
            Z = build_cluster_matrix(X, m, space=space)
            E0 = pca_embed(Z, n_components=cfg.pca_components, seed=0)  # PCA seed fixed like notebook

            for method in cfg.method_list:
                for k in cfg.n_clusters_list:
                    labels_by_seed = []
                    for sd in cfg.seeds:
                        ccfg = ClusterConfig(
                            space=space,
                            pca_components=cfg.pca_components,
                            method=method,
                            n_clusters=int(k),
                            seed=int(sd),
                        )
                        # clustering done in PCA space
                        lab = cluster_labels(E0, ccfg)
                        labels_by_seed.append(lab)

                    ari_mean, ari_std = stability_ari_over_seeds(labels_by_seed)
                    labels0 = labels_by_seed[0]

                    mcs = min_cluster_size(labels0)
                    if mcs < cfg.min_cluster_size:
                        continue

                    purity = mean_neighbor_purity(labels0, neigh)
                    tile_size = float(cfg.tile_mult) * float(r)
                    null_mu, null_sd = tile_shuffle_null_purity(
                        labels0, coords, neigh, tile_size=tile_size, n_null=cfg.n_null, seed=cfg.null_seed
                    )
                    purity_z = (purity - null_mu) / null_sd if null_sd > 0 else 0.0
                    purity_z_clip = float(np.clip(purity_z, -cfg.purityz_clip, cfg.purityz_clip))

                    sep = separability_cosine(E0, labels0)
                    frag = fragmentation_score_from_neigh(labels0, neigh)

                    score = cfg.w_stability * ari_mean + cfg.w_purityz * purity_z_clip + cfg.w_sep * sep

                    rows.append(
                        dict(
                            radius=float(r),
                            space=space,
                            kernel=cfg.kernel,
                            method=method,
                            n_clusters=int(k),
                            ari_mean=float(ari_mean),
                            ari_std=float(ari_std),
                            min_cluster_size=int(mcs),
                            purity=float(purity),
                            purity_null_mean=float(null_mu),
                            purity_null_std=float(null_sd),
                            purity_z=float(purity_z),
                            separability=float(sep),
                            fragmentation=float(frag),
                            score=float(score),
                        )
                    )

    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
    return df