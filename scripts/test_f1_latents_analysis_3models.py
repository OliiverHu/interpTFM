#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import pandas as pd

from interp_pipeline.downstream.f1.metrics import (
    F1AnalysisConfig,
    load_f1_long_from_root,
)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _load_term_meta(path: str | None) -> pd.DataFrame:
    if path is None or not os.path.exists(path):
        print(f"[warn] term meta not found: {path}")
        return pd.DataFrame(columns=["term_id", "term_name"])
    meta = pd.read_csv(path, sep="\t")
    if "term_id" not in meta.columns:
        raise ValueError(f"term meta missing term_id: {list(meta.columns)}")
    if "term_name" not in meta.columns and "name" in meta.columns:
        meta = meta.rename(columns={"name": "term_name"})
    if "term_name" not in meta.columns:
        meta["term_name"] = ""
    return meta[["term_id", "term_name"]].drop_duplicates("term_id")


def attach_term_names(f1_long: pd.DataFrame, term_meta: pd.DataFrame) -> pd.DataFrame:
    out = f1_long.copy()
    if "term_name" not in out.columns:
        out["term_name"] = None
    if len(term_meta):
        out = out.merge(term_meta, on="term_id", how="left", suffixes=("", "_meta"))
        if "term_name_meta" in out.columns:
            m = (
                out["term_name"].isna()
                | (out["term_name"].astype(str) == "None")
                | (out["term_name"].astype(str) == "")
            )
            out.loc[m, "term_name"] = out.loc[m, "term_name_meta"]
            out = out.drop(columns=["term_name_meta"])
    out["term_name"] = out["term_name"].fillna(out["term_id"].astype(str))
    return out


def analyze_one(
    *,
    label: str,
    heldout_root: str,
    out_dir: str,
    patterns: List[str],
    term_meta: pd.DataFrame,
    cfg: F1AnalysisConfig,
    min_true_pos: int | None,
    make_plots: bool,
) -> None:
    out_dir_p = ensure_dir(out_dir)

    print("=" * 100)
    print(f"[SAE F1 analysis] {label}")
    print(f"  heldout_root={heldout_root}")
    print(f"  out_dir={out_dir_p}")
    print("=" * 100)

    f1_long = load_f1_long_from_root(heldout_root, patterns=patterns, skip_empty=True)
    f1_long = attach_term_names(f1_long, term_meta)
    if min_true_pos is not None and "true_pos" in f1_long.columns and f1_long["true_pos"].notna().any():
        f1_long = f1_long[pd.to_numeric(f1_long["true_pos"], errors="coerce") >= int(min_true_pos)].copy()

    try:
        f1_long.to_parquet(out_dir_p / "f1_long.parquet", index=False)
    except Exception as e:
        print(f"[warn] parquet unavailable; writing CSV instead: {e}")
        f1_long.to_csv(out_dir_p / "f1_long.csv", index=False)

    sort_cols = ["layer", "threshold", "latent", "f1"]
    ascending = [True, True, True, False]
    if "tp" in f1_long.columns:
        sort_cols.append("tp")
        ascending.append(False)

    per_feature_best = (
        f1_long.sort_values(sort_cols, ascending=ascending)
        .groupby(["layer", "threshold", "latent"], as_index=False, dropna=False)
        .first()
        .rename(columns={"term_id": "best_term_id", "term_name": "best_term_name", "f1": "best_f1"})
    )
    per_feature_best.to_csv(out_dir_p / "per_feature_best.csv", index=False)

    per_feature_best_overall = (
        per_feature_best.sort_values(["layer", "latent", "best_f1"], ascending=[True, True, False])
        .groupby(["layer", "latent"], as_index=False)
        .first()
        .rename(columns={"threshold": "best_threshold"})
    )
    per_feature_best_overall.to_csv(out_dir_p / "per_feature_best_overall.csv", index=False)

    high = per_feature_best_overall[per_feature_best_overall["best_f1"] >= cfg.high_f1_latent_cutoff].copy()
    high.to_csv(out_dir_p / "high_f1_features.csv", index=False)

    layer_thr_summary = (
        per_feature_best.groupby(["layer", "threshold"], as_index=False, dropna=False)
        .agg(
            n_features=("latent", "nunique"),
            mean_best_f1=("best_f1", "mean"),
            median_best_f1=("best_f1", "median"),
            p90_best_f1=("best_f1", lambda s: s.quantile(0.90)),
            p95_best_f1=("best_f1", lambda s: s.quantile(0.95)),
            max_best_f1=("best_f1", "max"),
            n_high=("best_f1", lambda s: int((s >= cfg.high_f1_latent_cutoff).sum())),
            n_ge_0p5=("best_f1", lambda s: int((s >= 0.5).sum())),
            n_ge_0p6=("best_f1", lambda s: int((s >= 0.6).sum())),
            n_ge_0p7=("best_f1", lambda s: int((s >= 0.7).sum())),
        )
        .sort_values(["layer", "threshold"])
    )
    layer_thr_summary.to_csv(out_dir_p / "layer_threshold_summary.csv", index=False)

    concept_support = (
        f1_long[f1_long["f1"] >= float(cfg.f1_cutoff)]
        .groupby(["layer", "threshold", "term_id"], as_index=False, dropna=False)
        .agg(
            support=("latent", "nunique"),
            max_f1=("f1", "max"),
            sum_f1=("f1", "sum"),
            mean_f1=("f1", "mean"),
        )
        .merge(term_meta, on="term_id", how="left")
        .sort_values(["layer", "threshold", "support", "max_f1"], ascending=[True, True, False, False])
    )
    if "term_name" not in concept_support.columns:
        concept_support["term_name"] = concept_support["term_id"]
    concept_support["term_name"] = concept_support["term_name"].fillna(concept_support["term_id"].astype(str))
    concept_support.to_csv(out_dir_p / "concept_support.csv", index=False)

    best_feature_per_concept = (
        f1_long.sort_values(["layer", "threshold", "term_id", "f1"], ascending=[True, True, True, False])
        .groupby(["layer", "threshold", "term_id"], as_index=False, dropna=False)
        .first()
        .rename(columns={"latent": "best_feature", "f1": "best_f1"})
    )
    best_feature_per_concept.to_csv(out_dir_p / "best_feature_per_concept.csv", index=False)

    if make_plots:
        import matplotlib.pyplot as plt
        plot_dir = ensure_dir(out_dir_p / "plots")

        for layer in sorted(per_feature_best_overall["layer"].dropna().unique()):
            safe_layer = str(layer).replace("/", "_").replace(".", "p")
            dfL = per_feature_best_overall[per_feature_best_overall["layer"] == layer]
            plt.figure(figsize=(6, 4))
            plt.hist(dfL["best_f1"].dropna(), bins=40)
            plt.xlabel("Best F1 per SAE feature across thresholds")
            plt.ylabel("Count")
            plt.title(f"{label} | {layer}: SAE best F1 distribution")
            plt.tight_layout()
            plt.savefig(plot_dir / f"hist_best_f1_{safe_layer}.png", dpi=200)
            plt.close()

        agg = (
            layer_thr_summary.groupby("threshold", as_index=False)
            .agg(
                median_best_f1=("median_best_f1", "median"),
                mean_best_f1=("mean_best_f1", "mean"),
                max_best_f1=("max_best_f1", "max"),
                n_ge_0p5=("n_ge_0p5", "sum"),
                n_ge_0p6=("n_ge_0p6", "sum"),
                n_ge_0p7=("n_ge_0p7", "sum"),
            )
            .sort_values("threshold")
        )
        plt.figure(figsize=(6, 4))
        plt.plot(agg["threshold"], agg["median_best_f1"], marker="o", label="median")
        plt.plot(agg["threshold"], agg["mean_best_f1"], marker="o", label="mean")
        plt.plot(agg["threshold"], agg["max_best_f1"], marker="o", label="max")
        plt.xlabel("Threshold")
        plt.ylabel("F1")
        plt.title(f"{label}: SAE F1 vs threshold")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / "trend_f1_vs_threshold.png", dpi=200)
        plt.close()

    print(f"[OK] {label}")
    print(f"  f1_long rows: {len(f1_long):,}")
    print(f"  per_feature_best rows: {len(per_feature_best):,}")
    print(f"  out: {out_dir_p / 'per_feature_best.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="3-model SAE latent F1 analysis.")
    ap.add_argument("--labels", nargs=3, required=True)
    ap.add_argument("--heldout-roots", nargs=3, required=True)
    ap.add_argument("--out-dirs", nargs=3, required=True)
    ap.add_argument("--term-meta-path", required=True)
    ap.add_argument("--patterns", nargs="+", default=["**/test_concept_f1_scores.csv"])
    ap.add_argument("--f1-cutoff", type=float, default=0.20)
    ap.add_argument("--high-f1-latent-cutoff", type=float, default=0.35)
    ap.add_argument("--top-n-concepts", type=int, default=20)
    ap.add_argument("--min-true-pos", type=int, default=None)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    term_meta = _load_term_meta(args.term_meta_path)
    cfg = F1AnalysisConfig(
        f1_cutoff=float(args.f1_cutoff),
        high_f1_latent_cutoff=float(args.high_f1_latent_cutoff),
        top_n_concepts=int(args.top_n_concepts),
    )

    for label, heldout_root, out_dir in zip(args.labels, args.heldout_roots, args.out_dirs):
        analyze_one(
            label=label,
            heldout_root=heldout_root,
            out_dir=out_dir,
            patterns=list(args.patterns),
            term_meta=term_meta,
            cfg=cfg,
            min_true_pos=args.min_true_pos,
            make_plots=not args.no_plots,
        )


if __name__ == "__main__":
    main()


# python test_f1_latents_analysis_3models.py \
#   --labels scgpt c2sscale geneformer \
#   --heldout-roots \
#     /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/heldout_3models_l1_3e-3_geneheldout/scgpt/layer_4.norm2 \
#     /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/heldout_3models_l1_3e-3_geneheldout/c2sscale/layer_17 \
#     /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/heldout_3models_l1_3e-3_geneheldout/geneformer/layer_4 \
#   --out-dirs \
#     /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/full_scgpt_cosmx/f1_analysis \
#     /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/full_c2sscale_cosmx/f1_analysis \
#     /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/full_geneformer_cosmx/f1_analysis \
#   --term-meta-path /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/heldout_3models_l1_3e-3_geneheldout/gprofiler/gprofiler_terms.tsv \
#   --patterns "**/test_concept_f1_scores.csv" \
#   --min-true-pos 3