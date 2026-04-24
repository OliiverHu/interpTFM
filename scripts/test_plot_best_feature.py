#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def infer_columns(df: pd.DataFrame) -> tuple[str, str, str]:
    concept_candidates = ["concept", "term_id", "concept_id", "best_term_id"]
    feature_candidates = ["feature", "latent", "feature_id", "latent_id"]
    f1_candidates = ["f1", "best_f1"]

    concept_col = next((c for c in concept_candidates if c in df.columns), None)
    feature_col = next((c for c in feature_candidates if c in df.columns), None)
    f1_col = next((c for c in f1_candidates if c in df.columns), None)

    missing = []
    if concept_col is None:
        missing.append(f"concept column among {concept_candidates}")
    if feature_col is None:
        missing.append(f"feature/latent column among {feature_candidates}")
    if f1_col is None:
        missing.append(f"F1 column among {f1_candidates}")

    if missing:
        raise ValueError(
            "Could not infer required columns. Missing: "
            + ", ".join(missing)
            + f". Available columns: {list(df.columns)}"
        )

    return concept_col, feature_col, f1_col


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot histogram of best associated feature F1 per concept from valid/test score CSV."
    )
    parser.add_argument("--input", required=True, help="Path to valid_concept_f1_scores.csv or test_concept_f1_scores.csv")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--split-name", default=None, help="valid/test label for output names/title")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--min-true-pos", type=int, default=1, help="Drop concepts with true_pos below this, if true_pos column exists")
    parser.add_argument("--bins", type=int, default=40)
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    if args.threshold is not None:
        thr_col = "threshold_pct" if "threshold_pct" in df.columns else ("threshold" if "threshold" in df.columns else None)
        if thr_col is None:
            raise ValueError(f"--threshold was passed, but no threshold column exists in {in_path}")
        df = df[df[thr_col].astype(float) == float(args.threshold)].copy()

    if "true_pos" in df.columns and args.min_true_pos is not None:
        df = df[pd.to_numeric(df["true_pos"], errors="coerce") >= int(args.min_true_pos)].copy()

    if df.empty:
        raise ValueError("No rows remain after filtering. Check --threshold / --min-true-pos / input path.")

    concept_col, feature_col, f1_col = infer_columns(df)

    df[f1_col] = pd.to_numeric(df[f1_col], errors="coerce")
    df = df.dropna(subset=[concept_col, feature_col, f1_col]).copy()

    sort_cols = [concept_col, f1_col]
    ascending = [True, False]
    if "tp" in df.columns:
        sort_cols.append("tp")
        ascending.append(False)

    best = (
        df.sort_values(sort_cols, ascending=ascending)
        .groupby(concept_col, as_index=False, dropna=False)
        .first()
        .rename(
            columns={
                concept_col: "concept",
                feature_col: "best_feature",
                f1_col: "best_f1",
            }
        )
    )

    keep_cols = ["concept", "best_feature", "best_f1"]
    for c in ["threshold_pct", "threshold", "precision", "recall", "tp", "fp", "fn", "pred_pos", "true_pos"]:
        if c in best.columns and c not in keep_cols:
            keep_cols.append(c)

    best = best[keep_cols].sort_values("best_f1", ascending=False)

    prefix = f"{args.split_name}_" if args.split_name else ""
    if args.threshold is not None:
        prefix += f"thr_{str(args.threshold).replace('.', 'p')}_"
    prefix += f"mintrue_{args.min_true_pos}_"

    csv_path = out_dir / f"{prefix}best_feature_per_concept_f1.csv"
    png_path = out_dir / f"{prefix}best_feature_per_concept_f1_hist.png"
    json_path = out_dir / f"{prefix}best_feature_per_concept_f1_summary.json"

    best.to_csv(csv_path, index=False)

    summary = {
        "input": str(in_path),
        "split_name": args.split_name,
        "threshold_filter": args.threshold,
        "min_true_pos_filter": args.min_true_pos,
        "n_input_rows_after_filter": int(len(df)),
        "n_concepts": int(best["concept"].nunique()),
        "n_unique_best_features": int(best["best_feature"].nunique()),
        "mean_best_f1": float(best["best_f1"].mean()),
        "median_best_f1": float(best["best_f1"].median()),
        "p90_best_f1": float(best["best_f1"].quantile(0.90)),
        "p95_best_f1": float(best["best_f1"].quantile(0.95)),
        "max_best_f1": float(best["best_f1"].max()),
        "n_f1_eq_1": int((best["best_f1"] >= 0.999999).sum()),
        "n_f1_ge_0_9": int((best["best_f1"] >= 0.9).sum()),
        "n_f1_ge_0_75": int((best["best_f1"] >= 0.75).sum()),
        "n_f1_ge_0_5": int((best["best_f1"] >= 0.5).sum()),
    }
    if "true_pos" in best.columns:
        summary.update(
            {
                "true_pos_min": int(best["true_pos"].min()),
                "true_pos_median": float(best["true_pos"].median()),
                "true_pos_max": int(best["true_pos"].max()),
            }
        )

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    title_bits = []
    if args.model_name:
        title_bits.append(args.model_name)
    if args.split_name:
        title_bits.append(args.split_name)
    if args.threshold is not None:
        title_bits.append(f"threshold={args.threshold}")
    if args.min_true_pos is not None:
        title_bits.append(f"true_pos≥{args.min_true_pos}")

    plt.figure(figsize=(7, 4.5))
    plt.hist(best["best_f1"], bins=args.bins)
    plt.xlabel("Best F1 per concept")
    plt.ylabel("Number of concepts")
    plt.title(" | ".join(title_bits) if title_bits else "Best associated feature per concept")
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()

    print("[OK] wrote:")
    print(f"  {csv_path}")
    print(f"  {png_path}")
    print(f"  {json_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


# python test_plot_best_feature.py \
#   --input /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/heldout_3models_l1_3e-3_geneheldout/scgpt/layer_4.norm2/test_concept_f1_scores.csv \
#   --out-dir /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/heldout_3models_l1_3e-3_geneheldout/scgpt/layer_4.norm2/best_per_concept \
#   --model-name scGPT \
#   --split-name test \
#   --threshold 0.15 \
#   --min-true-pos 1

# python test_plot_best_feature.py \
#   --input /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/heldout_3models_l1_3e-3_geneheldout/geneformer/layer_4/test_concept_f1_scores.csv \
#   --out-dir /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/heldout_3models_l1_3e-3_geneheldout/geneformer/layer_4/best_per_concept \
#   --model-name geneformer \
#   --split-name test \
#   --threshold 0.15 \
#   --min-true-pos 1

# python test_plot_best_feature.py \
#   --input /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/heldout_3models_l1_3e-3_geneheldout/c2sscale/layer_17/test_concept_f1_scores.csv \
#   --out-dir /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/heldout_3models_l1_3e-3_geneheldout/c2sscale/layer_17/best_per_concept \
#   --model-name c2sscale \
#   --split-name test \
#   --threshold 0.15 \
#   --min-true-pos 1