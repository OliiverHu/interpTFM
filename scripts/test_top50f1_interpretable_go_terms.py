# scripts/top50_interpretable_go_terms.py
from __future__ import annotations

import os
import pandas as pd

# =========================
# EDIT HERE
# =========================
RUNS_ROOT = "runs/full_scgpt_cosmx"
LAYER = "layer_4.norm2"
THRESHOLD = 0.6

PER_FEATURE_BEST = os.path.join(RUNS_ROOT, "f1_analysis", "per_feature_best.csv")
TERM_META = os.path.join(RUNS_ROOT, "gprofiler", "gprofiler_terms.tsv")

OUT_PATH = os.path.join(RUNS_ROOT, "f1_analysis", f"top50_interpretable_go_{LAYER}_thr{THRESHOLD}.csv")
TOPK = 50
# =========================


def load_term_meta():
    meta = pd.read_csv(TERM_META, sep="\t")
    if "term_name" not in meta.columns and "name" in meta.columns:
        meta = meta.rename(columns={"name": "term_name"})
    if "term_name" not in meta.columns:
        meta["term_name"] = ""
    return meta[["term_id", "term_name"]].drop_duplicates("term_id")


def main():
    df = pd.read_csv(PER_FEATURE_BEST)
    df = df[(df["layer"] == LAYER) & (df["threshold"] == float(THRESHOLD))].copy()
    if df.empty:
        raise RuntimeError(f"No rows for layer={LAYER} threshold={THRESHOLD} in {PER_FEATURE_BEST}")

    df = df.rename(columns={"best_term_id": "term_id", "best_f1": "best_f1"})

    # Only GO terms
    df = df[df["term_id"].astype(str).str.startswith("GO:")].copy()

    meta = load_term_meta()
    df = df.merge(meta, on="term_id", how="left")

    agg = (
        df.groupby(["term_id", "term_name"], as_index=False)
        .agg(
            n_features=("latent", "nunique"),
            mean_best_f1=("best_f1", "mean"),
            median_best_f1=("best_f1", "median"),
            max_best_f1=("best_f1", "max"),
        )
    )
    agg["score"] = agg["n_features"] * agg["mean_best_f1"]

    top = agg.sort_values(["score", "n_features", "max_best_f1"], ascending=False).head(int(TOPK))
    top.to_csv(OUT_PATH, index=False)

    print("[OK] wrote:", OUT_PATH)
    print(top[["term_id", "term_name", "n_features", "mean_best_f1", "max_best_f1", "score"]].to_string(index=False))


if __name__ == "__main__":
    main()