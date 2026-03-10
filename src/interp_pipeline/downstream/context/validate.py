from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# -------------------------
# Block split (3x3)
# -------------------------
def split_into_3x3_blocks(coords: np.ndarray) -> np.ndarray:
    """
    Split points into 9 blocks by tertiles of x and y.
    Returns block_id in [0..8] where block = bx*3 + by.
    """
    x = coords[:, 0]
    y = coords[:, 1]
    qx = np.quantile(x, [1/3, 2/3])
    qy = np.quantile(y, [1/3, 2/3])

    bx = (x > qx[0]).astype(int) + (x > qx[1]).astype(int)
    by = (y > qy[0]).astype(int) + (y > qy[1]).astype(int)
    return (bx * 3 + by).astype(np.int32)


# -------------------------
# Signatures + Hungarian matching
# -------------------------
def cluster_signatures(Z: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (centroids, cluster_ids) where centroids shape (K, D) aligned to cluster_ids.
    """
    labs = np.unique(labels)
    cents = []
    for k in labs:
        idx = np.where(labels == k)[0]
        if idx.size == 0:
            continue
        cents.append(Z[idx].mean(axis=0))
    C = np.stack(cents, axis=0).astype(np.float32)
    return C, labs.astype(np.int32)


def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Cosine similarity between rows of A and rows of B.
    """
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + eps)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + eps)
    return An @ Bn.T


def hungarian_match(sim: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Max-similarity Hungarian matching on sim matrix (nA x nB).
    Returns (row_idx, col_idx, mean_sim_of_matches)
    """
    from scipy.optimize import linear_sum_assignment
    # Hungarian minimizes cost, so use -sim
    r, c = linear_sum_assignment(-sim)
    mean_sim = float(sim[r, c].mean()) if len(r) else float("nan")
    return r, c, mean_sim


def block_signature_matching(
    Z: np.ndarray,
    labels: np.ndarray,
    block_id: np.ndarray,
    *,
    ref_block: int = 4,
) -> pd.DataFrame:
    """
    For each block b != ref_block:
      - compute cluster centroids in block b and in ref block
      - cosine similarity matrix
      - Hungarian match
      - report mean matched similarity and per-pair matches
    """
    rows = []
    ref_mask = block_id == ref_block
    if ref_mask.sum() == 0:
        raise RuntimeError(f"Reference block {ref_block} has 0 cells")

    C_ref, labs_ref = cluster_signatures(Z[ref_mask], labels[ref_mask])

    for b in sorted(np.unique(block_id).tolist()):
        if b == ref_block:
            continue
        m = block_id == b
        if m.sum() == 0:
            continue
        C_b, labs_b = cluster_signatures(Z[m], labels[m])
        sim = cosine_similarity_matrix(C_b, C_ref)

        r, c, mean_sim = hungarian_match(sim)

        # record matched pairs
        for rr, cc in zip(r, c):
            rows.append(
                dict(
                    ref_block=int(ref_block),
                    block=int(b),
                    block_cluster=int(labs_b[rr]),
                    ref_cluster=int(labs_ref[cc]),
                    cosine_sim=float(sim[rr, cc]),
                    mean_cosine_sim=float(mean_sim),
                    n_cells_block=int(m.sum()),
                    n_cells_ref=int(ref_mask.sum()),
                    n_clusters_block=int(len(labs_b)),
                    n_clusters_ref=int(len(labs_ref)),
                )
            )

    return pd.DataFrame(rows)


# -------------------------
# Leave-one-block-out LR
# -------------------------
def leave_one_block_out_lr(
    Z: np.ndarray,
    y: np.ndarray,
    block_id: np.ndarray,
    *,
    seed: int = 0,
    max_iter: int = 2000,
    C: float = 1.0,
) -> pd.DataFrame:
    """
    Train multinomial logistic regression to predict global labels y from Z,
    leaving one block out each time.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    rows = []
    uniq_blocks = sorted(np.unique(block_id).tolist())

    for b in uniq_blocks:
        test = block_id == b
        train = ~test
        if test.sum() == 0 or train.sum() == 0:
            continue

        clf = LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=int(max_iter),
            C=float(C),
            random_state=int(seed),
            n_jobs=None,
        )
        clf.fit(Z[train], y[train])
        yp = clf.predict(Z[test])

        rows.append(
            dict(
                heldout_block=int(b),
                n_train=int(train.sum()),
                n_test=int(test.sum()),
                acc=float(accuracy_score(y[test], yp)),
                bal_acc=float(balanced_accuracy_score(y[test], yp)),
                macro_f1=float(f1_score(y[test], yp, average="macro")),
            )
        )

    return pd.DataFrame(rows)


# -------------------------
# Cell-type enrichment
# -------------------------
def niche_celltype_table(
    labels: np.ndarray,
    celltypes: pd.Series,
) -> pd.DataFrame:
    """
    Crosstab of niche label x celltype (counts + row-normalized fractions).
    """
    df = pd.DataFrame({"niche": labels.astype(int), "celltype": celltypes.astype(str)})
    ct = pd.crosstab(df["niche"], df["celltype"])
    frac = ct.div(ct.sum(axis=1), axis=0)
    out = ct.copy()
    out.columns = [f"count::{c}" for c in out.columns]
    frac.columns = [f"frac::{c}" for c in frac.columns]
    return pd.concat([out, frac], axis=1).reset_index().rename(columns={"niche": "niche"})


def celltype_chi2_residuals(
    labels: np.ndarray,
    celltypes: pd.Series,
) -> pd.DataFrame:
    """
    Chi-square standardized residuals for niche x celltype contingency.
    Useful to spot enriched/depleted celltypes per niche.
    """
    from scipy.stats import chi2_contingency

    df = pd.DataFrame({"niche": labels.astype(int), "celltype": celltypes.astype(str)})
    tab = pd.crosstab(df["niche"], df["celltype"]).astype(float)

    # chi2
    chi2, p, dof, expected = chi2_contingency(tab.values)
    expected = np.asarray(expected, dtype=float)
    # standardized residuals
    resid = (tab.values - expected) / np.sqrt(expected + 1e-12)

    out = []
    niches = tab.index.tolist()
    cts = tab.columns.tolist()
    for i, n in enumerate(niches):
        for j, c in enumerate(cts):
            out.append(
                dict(
                    niche=int(n),
                    celltype=str(c),
                    count=float(tab.values[i, j]),
                    expected=float(expected[i, j]),
                    std_resid=float(resid[i, j]),
                    chi2=float(chi2),
                    p_value=float(p),
                    dof=int(dof),
                )
            )
    return pd.DataFrame(out).sort_values("std_resid", ascending=False).reset_index(drop=True)