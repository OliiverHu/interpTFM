from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class ClusterConfig:
    space: str = "m"          # "m" | "xm"
    pca_components: int = 50
    method: str = "kmeans"    # "kmeans" | "gmm"
    n_clusters: int = 12
    seed: int = 0


def build_cluster_matrix(X: np.ndarray, m: np.ndarray, space: str) -> np.ndarray:
    if space == "m":
        return m
    if space == "xm":
        return np.concatenate([X, m], axis=1)
    raise ValueError(f"Unknown cluster space: {space}")


def pca_embed(Z: np.ndarray, n_components: int, seed: int) -> np.ndarray:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    Zs = StandardScaler(with_mean=True, with_std=True).fit_transform(Z).astype(np.float32)
    pca = PCA(n_components=int(n_components), random_state=int(seed))
    return pca.fit_transform(Zs).astype(np.float32)


def cluster_labels(E: np.ndarray, cfg: ClusterConfig) -> np.ndarray:
    if cfg.method == "kmeans":
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=int(cfg.n_clusters), random_state=int(cfg.seed), n_init="auto")
        return km.fit_predict(E).astype(np.int32)

    if cfg.method == "gmm":
        from sklearn.mixture import GaussianMixture
        gm = GaussianMixture(n_components=int(cfg.n_clusters), random_state=int(cfg.seed), covariance_type="full")
        return gm.fit_predict(E).astype(np.int32)

    raise ValueError(f"Unknown cluster method: {cfg.method}")