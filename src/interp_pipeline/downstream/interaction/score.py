from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ScoreConfig:
    min_edges: int = 200
    min_weight_sum: float = 1e-3
    min_neff: float = 25.0

    std_floor_q: float = 0.10
    topk: int = 10
    eps: float = 1e-8


def neff_from_s1_s2(S1: np.ndarray, S2: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return (S1 * S1) / (S2 + eps)


def exposure_normalize(M: np.ndarray, weight_sum: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return M / (weight_sum.astype(np.float32)[:, None] + float(eps))


def apply_null_std_floor(null_std_norm: np.ndarray, floor_q: float, eps: float = 1e-8) -> np.ndarray:
    """
    Notebook: floor per concept using quantile across pairs.
    """
    floor = np.quantile(null_std_norm, float(floor_q), axis=0)
    floor = floor.astype(np.float32)
    return np.maximum(null_std_norm, floor[None, :] + float(eps)).astype(np.float32)


def compute_z_from_norm(
    S1: np.ndarray,
    null_mean: np.ndarray,
    null_std: np.ndarray,
    weight_sum: np.ndarray,
    cfg: ScoreConfig,
) -> np.ndarray:
    """
    Notebook ordering:
      S1_norm = S1 / weight_sum
      null_mean_norm/std_norm same
      floor std
      Z = (obs - mu) / sd
    """
    obs = exposure_normalize(S1, weight_sum, eps=cfg.eps)
    mu = exposure_normalize(null_mean, weight_sum, eps=cfg.eps)
    sd = exposure_normalize(null_std, weight_sum, eps=cfg.eps)
    sd_eff = apply_null_std_floor(sd, cfg.std_floor_q, eps=cfg.eps)
    return ((obs - mu) / (sd_eff + float(cfg.eps))).astype(np.float32)


def intensity_median_topk(Z: np.ndarray, neff: np.ndarray, edge_count: np.ndarray, weight_sum: np.ndarray, cfg: ScoreConfig) -> np.ndarray:
    """
    Notebook: mask unstable concepts by neff, then median of top-k positive.
    Uses np.partition (not full sort).
    """
    P, T = Z.shape
    out = np.full((P,), np.nan, dtype=np.float32)

    for p in range(P):
        if edge_count[p] < cfg.min_edges:
            continue
        if weight_sum[p] < cfg.min_weight_sum:
            continue

        m = neff[p] >= float(cfg.min_neff)
        if not np.any(m):
            continue

        z = Z[p][m]
        z = z[np.isfinite(z)]
        z = z[z > 0]
        if z.size == 0:
            continue

        k = min(int(cfg.topk), int(z.size))
        top = np.partition(z, -k)[-k:]
        out[p] = float(np.median(top))

    return out