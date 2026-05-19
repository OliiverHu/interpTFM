#!/usr/bin/env python3
from __future__ import annotations

"""
Step 5: 3-model GO parent-reduction / NMI analysis for F1 results.

This is a post-hoc GO hierarchy analysis on existing F1 score tables.
It does NOT build a new GT matrix.

For each model:
  - reads an existing F1 table
  - keeps GO concepts only
  - optionally filters by latent threshold and min true_pos
  - sweeps max_descendants × f1_min
  - maps detailed GO terms to broader parents with Strategy C
  - computes NMI between feature labels and reduced GO-parent labels
  - writes tables and plots

Typical input:
  runs/heldout_3models_l1_3e-3_geneheldout/<model>/<layer>/test_concept_f1_scores.csv

Example:
python test_go_reduce_3models.py \
  --labels scgpt c2sscale geneformer \
  --layers layer_4.norm2 layer_17 layer_4 \
  --f1-tables \
    /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/heldout_3models_l1_3e-3_geneheldout/scgpt/layer_4.norm2/test_concept_f1_scores.csv \
    /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/heldout_3models_l1_3e-3_geneheldout/c2sscale/layer_17/test_concept_f1_scores.csv \
    /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/heldout_3models_l1_3e-3_geneheldout/geneformer/layer_4/test_concept_f1_scores.csv \
  --out-root /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/f1_go_parent_nmi_3models \
  --go-obo-path /maiziezhou_lab2/yunfei/Projects/interpTFM/resources/go-basic.obo \
  --threshold 0.15 \
  --min-true-pos 3
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from interp_pipeline.downstream.f1.go_reduce import (
    ensure_go_basic_obo,
    nmi_sweep,
    reduce_go_terms_strategy_c,
    ParentSelectConfig,
)


DEFAULT_MAX_DESC_SWEEP = [50, 100, 200, 500, 1000, 2000, 5000]
DEFAULT_F1_MIN_SWEEP = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def infer_col(df: pd.DataFrame, candidates: Sequence[str], required: bool = True) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"Could not infer column among {candidates}. Have columns: {list(df.columns)}")
    return None


def normalize_f1_table(
    f1_table: str | Path,
    *,
    threshold: Optional[float],
    min_true_pos: Optional[int],
) -> pd.DataFrame:
    df = pd.read_csv(f1_table)

    concept_col = infer_col(df, ["concept", "term_id", "concept_id", "native"])
    feature_col = infer_col(df, ["feature", "latent", "feature_id", "latent_id"])
    f1_col = infer_col(df, ["f1", "best_f1", "F1"])
    thr_col = infer_col(df, ["threshold_pct", "threshold", "thr", "latent_threshold"], required=False)
    true_pos_col = infer_col(df, ["true_pos", "n_true", "true"], required=False)

    out = pd.DataFrame(
        {
            "concept": df[concept_col].astype(str),
            "feature": pd.to_numeric(df[feature_col], errors="coerce"),
            "f1": pd.to_numeric(df[f1_col], errors="coerce"),
        }
    )

    if thr_col is not None:
        out["threshold"] = pd.to_numeric(df[thr_col], errors="coerce")
    else:
        out["threshold"] = np.nan

    if true_pos_col is not None:
        out["true_pos"] = pd.to_numeric(df[true_pos_col], errors="coerce")
    else:
        out["true_pos"] = np.nan

    for c in ["tp", "fp", "fn", "precision", "recall", "pred_pos"]:
        if c in df.columns:
            out[c] = pd.to_numeric(df[c], errors="coerce")

    out = out.dropna(subset=["feature", "f1"]).copy()
    out["feature"] = out["feature"].astype(int)

    out = out[out["concept"].astype(str).str.startswith("GO:")].copy()

    if threshold is not None and out["threshold"].notna().any():
        out = out[np.isclose(out["threshold"].astype(float), float(threshold))].copy()

    if min_true_pos is not None and out["true_pos"].notna().any():
        out = out[out["true_pos"] >= int(min_true_pos)].copy()

    return out


def plot_heatmap(
    pivot: pd.DataFrame,
    *,
    title: str,
    cbar_label: str,
    out_path: Path,
    fmt: str = ".2f",
) -> None:
    plt.figure(figsize=(8, 5.5))
    arr = pivot.values
    im = plt.imshow(arr, aspect="auto")
    plt.xticks(range(pivot.shape[1]), [str(x) for x in pivot.columns], rotation=45, ha="right")
    plt.yticks(range(pivot.shape[0]), [str(x) for x in pivot.index])
    plt.xlabel("max_descendants")
    plt.ylabel("f1_min")
    plt.title(title)

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            if pd.isna(val):
                txt = "nan"
            elif fmt == "d":
                txt = str(int(round(float(val))))
            else:
                txt = format(float(val), fmt)
            plt.text(j, i, txt, ha="center", va="center", fontsize=8)

    plt.colorbar(im, label=cbar_label)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_nmi_lines(
    sweep: pd.DataFrame,
    *,
    title: str,
    out_path: Path,
) -> None:
    plt.figure(figsize=(7, 4.5))
    for max_desc, g in sweep.groupby("max_descendants"):
        g = g.sort_values("f1_min")
        plt.plot(g["f1_min"], g["nmi"], marker="o", label=str(max_desc))
    plt.xlabel("f1_min")
    plt.ylabel("NMI")
    plt.title(title)
    plt.legend(title="max_desc", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def run_one_model(
    *,
    label: str,
    layer: str,
    f1_table: str,
    out_dir: Path,
    go_obo_path: str,
    max_descendants_grid: Sequence[int],
    f1_min_grid: Sequence[float],
    threshold: Optional[float],
    min_true_pos: Optional[int],
    exclude_self: bool,
    allow_self_fallback: bool,
    sanity_max_desc: int,
    sanity_print_n: int,
) -> Dict[str, Any]:
    out_dir = ensure_dir(out_dir)
    plot_dir = ensure_dir(out_dir / "plots")

    print("=" * 100)
    print(f"[GO NMI] {label} | {layer}")
    print(f"  f1_table={f1_table}")
    print(f"  out_dir={out_dir}")
    print(f"  threshold={threshold}")
    print(f"  min_true_pos={min_true_pos}")
    print("=" * 100)

    f1_df = normalize_f1_table(f1_table, threshold=threshold, min_true_pos=min_true_pos)

    if f1_df.empty:
        raise RuntimeError(
            f"No GO rows remain after filtering for model={label}, layer={layer}. "
            f"Check threshold/min_true_pos or input table."
        )

    f1_df.to_csv(out_dir / "go_f1_input_filtered.csv", index=False)

    sweep = nmi_sweep(
        f1_df,
        go_obo_path=go_obo_path,
        max_descendants_grid=[int(x) for x in max_descendants_grid],
        f1_min_grid=[float(x) for x in f1_min_grid],
        concept_col="concept",
        feature_col="feature",
        f1_col="f1",
        only_go=True,
        exclude_self=bool(exclude_self),
        allow_self_fallback=bool(allow_self_fallback),
    )
    sweep["model"] = label
    sweep["layer"] = layer
    sweep["threshold_filter"] = np.nan if threshold is None else float(threshold)
    sweep["min_true_pos_filter"] = np.nan if min_true_pos is None else int(min_true_pos)
    sweep.to_csv(out_dir / "nmi_sweep.csv", index=False)

    sweep_valid = sweep.dropna(subset=["nmi"]).copy()
    if len(sweep_valid):
        best = sweep_valid.sort_values(["nmi", "n_concepts"], ascending=[False, False]).head(1).copy()
        best.to_csv(out_dir / "best_setting.csv", index=False)
        print("[best]")
        print(best.to_string(index=False))
    else:
        best = pd.DataFrame()
        print("[warn] no valid sweep rows after dropna(nmi).")

    if len(sweep):
        nmi_pivot = sweep.pivot_table(index="f1_min", columns="max_descendants", values="nmi", aggfunc="mean")
        plot_heatmap(
            nmi_pivot,
            title=f"{label} {layer}: GO parent NMI",
            cbar_label="NMI",
            out_path=plot_dir / "nmi_heatmap.png",
            fmt=".2f",
        )

        if "n_concepts" in sweep.columns:
            nconcept_pivot = sweep.pivot_table(
                index="f1_min",
                columns="max_descendants",
                values="n_concepts",
                aggfunc="mean",
            )
            plot_heatmap(
                nconcept_pivot,
                title=f"{label} {layer}: GO concepts retained",
                cbar_label="# concepts",
                out_path=plot_dir / "n_concepts_heatmap.png",
                fmt="d",
            )

        plot_nmi_lines(
            sweep,
            title=f"{label} {layer}: NMI vs F1 cutoff",
            out_path=plot_dir / "nmi_vs_f1_min_by_max_descendants.png",
        )

    unique_go = sorted(set(f1_df["concept"].astype(str).unique()))
    cfg = ParentSelectConfig(
        max_descendants=int(sanity_max_desc),
        exclude_self=bool(exclude_self),
        allow_self_fallback=bool(allow_self_fallback),
    )
    mapping = reduce_go_terms_strategy_c(unique_go, go_obo_path=go_obo_path, cfg=cfg)

    mapping_rows = []
    for go_id in unique_go:
        parent = mapping.get(go_id, go_id)
        mapping_rows.append(
            {
                "model": label,
                "layer": layer,
                "go_id": go_id,
                "mapped_parent": parent,
                "changed": bool(go_id != parent),
            }
        )
    mapping_df = pd.DataFrame(mapping_rows)
    mapping_df.to_csv(out_dir / "go_parent_mapping_sanity.csv", index=False)

    n_changed = int(mapping_df["changed"].sum()) if len(mapping_df) else 0
    sanity = {
        "model": label,
        "layer": layer,
        "f1_table": f1_table,
        "threshold_filter": None if threshold is None else float(threshold),
        "min_true_pos_filter": None if min_true_pos is None else int(min_true_pos),
        "n_go_rows_after_filter": int(len(f1_df)),
        "n_unique_go_terms_after_filter": int(len(unique_go)),
        "sanity_max_desc": int(sanity_max_desc),
        "n_changed": n_changed,
        "changed_fraction": float(n_changed / max(1, len(mapping_df))),
        "example_mappings": mapping_rows[: int(sanity_print_n)],
    }
    with open(out_dir / "sanity_summary.json", "w") as f:
        json.dump(sanity, f, indent=2)

    print(
        f"[sanity] max_descendants={sanity_max_desc} "
        f"changed={n_changed}/{len(mapping_df)} "
        f"({100.0 * n_changed / max(1, len(mapping_df)):.2f}%)"
    )
    print("[sanity] examples:")
    for row in mapping_rows[: int(sanity_print_n)]:
        print(" ", row["go_id"], "->", row["mapped_parent"])

    summary = {
        "model": label,
        "layer": layer,
        "n_go_rows_after_filter": int(len(f1_df)),
        "n_unique_go_terms_after_filter": int(len(unique_go)),
        "n_changed": n_changed,
        "changed_fraction": float(n_changed / max(1, len(mapping_df))),
    }

    if len(best):
        for c in best.columns:
            val = best.iloc[0][c]
            if isinstance(val, np.integer):
                val = int(val)
            elif isinstance(val, np.floating):
                val = float(val)
            summary[f"best_{c}"] = val

    return summary


def plot_cross_model_best(best_df: pd.DataFrame, *, out_dir: Path) -> None:
    plot_dir = ensure_dir(out_dir / "plots_cross_model")

    if best_df.empty or "best_nmi" not in best_df.columns:
        return

    plt.figure(figsize=(7, 4.5))
    x = np.arange(len(best_df))
    plt.bar(x, best_df["best_nmi"].astype(float).values)
    plt.xticks(x, best_df["model"].astype(str).tolist())
    plt.ylabel("Best NMI")
    plt.title("Best GO parent NMI by model")
    plt.tight_layout()
    plt.savefig(plot_dir / "best_nmi_by_model.png", dpi=200)
    plt.close()

    if "best_n_concepts" in best_df.columns:
        plt.figure(figsize=(7, 4.5))
        plt.bar(x, best_df["best_n_concepts"].astype(float).values)
        plt.xticks(x, best_df["model"].astype(str).tolist())
        plt.ylabel("# concepts at best setting")
        plt.title("# GO concepts retained at best NMI setting")
        plt.tight_layout()
        plt.savefig(plot_dir / "best_n_concepts_by_model.png", dpi=200)
        plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 5: 3-model GO parent NMI sweep for F1 results.")
    ap.add_argument("--labels", nargs=3, required=True)
    ap.add_argument("--layers", nargs=3, required=True)
    ap.add_argument("--f1-tables", nargs=3, required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--go-obo-path", default="resources/go-basic.obo")

    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--min-true-pos", type=int, default=None)

    ap.add_argument("--max-descendants-grid", nargs="+", type=int, default=DEFAULT_MAX_DESC_SWEEP)
    ap.add_argument("--f1-min-grid", nargs="+", type=float, default=DEFAULT_F1_MIN_SWEEP)

    ap.add_argument("--exclude-self", action="store_true", default=True)
    ap.add_argument("--include-self", dest="exclude_self", action="store_false")
    ap.add_argument("--allow-self-fallback", action="store_true", default=True)
    ap.add_argument("--no-self-fallback", dest="allow_self_fallback", action="store_false")

    ap.add_argument("--sanity-max-desc", type=int, default=500)
    ap.add_argument("--sanity-print-n", type=int, default=15)

    args = ap.parse_args()

    out_root = ensure_dir(args.out_root)
    ensure_go_basic_obo(args.go_obo_path)

    summaries: List[Dict[str, Any]] = []
    sweeps: List[pd.DataFrame] = []
    best_rows: List[pd.DataFrame] = []

    for label, layer, f1_table in zip(args.labels, args.layers, args.f1_tables):
        model_out = out_root / label / layer.replace("/", "_")
        summary = run_one_model(
            label=label,
            layer=layer,
            f1_table=f1_table,
            out_dir=model_out,
            go_obo_path=args.go_obo_path,
            max_descendants_grid=args.max_descendants_grid,
            f1_min_grid=args.f1_min_grid,
            threshold=args.threshold,
            min_true_pos=args.min_true_pos,
            exclude_self=args.exclude_self,
            allow_self_fallback=args.allow_self_fallback,
            sanity_max_desc=args.sanity_max_desc,
            sanity_print_n=args.sanity_print_n,
        )
        summaries.append(summary)

        sweep_path = model_out / "nmi_sweep.csv"
        if sweep_path.exists():
            sweeps.append(pd.read_csv(sweep_path))

        best_path = model_out / "best_setting.csv"
        if best_path.exists():
            b = pd.read_csv(best_path)
            if len(b):
                best_rows.append(b)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out_root / "combined_summary.csv", index=False)

    if sweeps:
        combined_sweep = pd.concat(sweeps, ignore_index=True)
        combined_sweep.to_csv(out_root / "combined_nmi_sweep.csv", index=False)

    if best_rows:
        combined_best = pd.concat(best_rows, ignore_index=True)
        combined_best.to_csv(out_root / "combined_best_settings_raw.csv", index=False)

        best_friendly_rows = []
        for _, row in combined_best.iterrows():
            best_friendly_rows.append(
                {
                    "model": row.get("model"),
                    "layer": row.get("layer"),
                    "best_nmi": row.get("nmi"),
                    "best_n_concepts": row.get("n_concepts"),
                    "best_f1_min": row.get("f1_min"),
                    "best_max_descendants": row.get("max_descendants"),
                    "threshold_filter": row.get("threshold_filter"),
                    "min_true_pos_filter": row.get("min_true_pos_filter"),
                }
            )
        best_friendly = pd.DataFrame(best_friendly_rows)
        best_friendly.to_csv(out_root / "combined_best_settings.csv", index=False)
        plot_cross_model_best(best_friendly, out_dir=out_root)

    print(f"\n[OK] wrote GO NMI outputs to: {out_root}")


if __name__ == "__main__":
    main()



# python test_go_reduce_3models.py \
#   --labels scgpt c2sscale geneformer \
#   --layers layer_4.norm2 layer_17 layer_4 \
#   --f1-tables \
#     /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/heldout_3models_l1_3e-3_geneheldout/scgpt/layer_4.norm2/test_concept_f1_scores.csv \
#     /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/heldout_3models_l1_3e-3_geneheldout/c2sscale/layer_17/test_concept_f1_scores.csv \
#     /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/heldout_3models_l1_3e-3_geneheldout/geneformer/layer_4/test_concept_f1_scores.csv \
#   --out-root /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/f1_go_parent_nmi_3models \
#   --go-obo-path /maiziezhou_lab2/yunfei/Projects/interpTFM/resources/go-basic.obo \
#   --threshold 0.15 \
#   --min-true-pos 3