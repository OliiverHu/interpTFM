#!/usr/bin/env python3
from __future__ import annotations

"""
3-model downstream F1 comparison: SAE latents vs raw/model embedding dimensions.

This script is meant to live in scripts/ or tests/, and uses:
  interp_pipeline.downstream.f1.metrics
  interp_pipeline.downstream.f1.plot

It accepts either:
  1) heldout score CSVs such as test_concept_f1_scores.csv
     columns: concept, feature, threshold_pct, f1, tp, true_pos, ...
  2) already-collapsed per-feature-best CSVs
     columns: layer, threshold, latent, best_term_id, best_f1, ...

For each model, it writes:
  - normalized long tables for SAE and embedding baseline
  - per-feature-best tables
  - concept-support tables
  - SAE-vs-embedding summary table
  - overlay histogram
  - top-k curve
  - concept usage barplots
  - concept-feature heatmaps via downstream.f1.plot
  - threshold × min_true_pos count grids for F1 >= 0.5 / 0.6 / 0.7
"""

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from interp_pipeline.downstream.f1.metrics import (
    normalize_f1_table,
    best_term_per_latent,
    concept_support_table,
)
from interp_pipeline.downstream.f1.plot import (
    LatentsVsActsPlotConfig,
    plot_latents_vs_acts,
    load_term_meta,
)


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_csv_any(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, index_col=0)


def infer_term_id_col(df: pd.DataFrame) -> str:
    for c in ["term_id", "concept", "term", "concept_id", "native", "best_term_id"]:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot infer term/concept column. Columns={list(df.columns)}")


def infer_feature_col(df: pd.DataFrame) -> str:
    for c in ["latent", "feature", "k", "latent_id", "feature_id"]:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot infer latent/feature column. Columns={list(df.columns)}")


def infer_threshold_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["threshold", "threshold_pct", "thr", "latent_threshold"]:
        if c in df.columns:
            return c
    return None


def infer_f1_col(df: pd.DataFrame) -> str:
    for c in ["f1", "F1", "best_f1", "max_F1", "f1_score"]:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot infer F1 column. Columns={list(df.columns)}")


def add_term_names_from_meta(df: pd.DataFrame, term_meta_path: Optional[str]) -> pd.DataFrame:
    out = df.copy()
    if "term_name" in out.columns and out["term_name"].notna().any():
        return out
    if term_meta_path is None or not os.path.exists(term_meta_path):
        out["term_name"] = out["term_id"].astype(str)
        return out
    meta = load_term_meta(term_meta_path)
    if meta.empty:
        out["term_name"] = out["term_id"].astype(str)
        return out
    mapping = meta.set_index("term_id")["term_name"].to_dict()
    out["term_name"] = out["term_id"].astype(str).map(lambda x: mapping.get(str(x), str(x)))
    return out


def normalize_any_f1_csv(
    csv_path: str | Path,
    *,
    layer: str,
    default_threshold: Optional[float],
    term_meta_path: Optional[str],
) -> pd.DataFrame:
    """
    Normalize either heldout score CSV or per-feature-best CSV to metrics.py canonical schema:
      layer, threshold, latent, term_id, term_name, f1, tp, pred_pos, true_pos, source_file
    """
    raw = read_csv_any(csv_path)

    # Already per-feature-best from an older script.
    if {"best_term_id", "best_f1"}.issubset(raw.columns):
        feat_col = infer_feature_col(raw)
        thr_col = infer_threshold_col(raw)

        d = raw.copy()
        if "layer" in d.columns:
            d = d[d["layer"].astype(str) == str(layer)].copy()
        if thr_col is None:
            d["threshold"] = default_threshold
            thr_col = "threshold"

        out = pd.DataFrame({
            "layer": layer,
            "threshold": pd.to_numeric(d[thr_col], errors="coerce"),
            "latent": pd.to_numeric(d[feat_col].astype(str).str.extract(r"(-?\d+)", expand=False), errors="coerce"),
            "term_id": d["best_term_id"].astype(str),
            "term_name": d["best_term_name"].astype(str) if "best_term_name" in d.columns else None,
            "f1": pd.to_numeric(d["best_f1"], errors="coerce"),
            "tp": pd.to_numeric(d["tp"], errors="coerce") if "tp" in d.columns else np.nan,
            "pred_pos": pd.to_numeric(d["pred_pos"], errors="coerce") if "pred_pos" in d.columns else np.nan,
            "true_pos": pd.to_numeric(d["true_pos"], errors="coerce") if "true_pos" in d.columns else np.nan,
            "source_file": str(csv_path),
        })
        out = out.dropna(subset=["latent", "f1", "term_id"]).copy()
        out["latent"] = out["latent"].astype(int)
        out = add_term_names_from_meta(out, term_meta_path)
        return out

    # Generic heldout/grid table. Let the repo helper handle most schema variants.
    # IMPORTANT: normalize_f1_table already knows threshold_pct and concept/feature.
    out = normalize_f1_table(
        raw,
        layer=layer,
        source_file=str(csv_path),
        default_threshold=default_threshold,
    )
    out = add_term_names_from_meta(out, term_meta_path)
    return out


def write_per_feature_best_for_plot(
    f1_long: pd.DataFrame,
    *,
    out_csv: str | Path,
) -> pd.DataFrame:
    """
    Convert canonical long F1 table into the exact per_feature_best schema expected by
    downstream.f1.plot.build_f1_df_latents_best_only.
    """
    best = best_term_per_latent(f1_long)
    out = best.rename(columns={
        "latent": "latent",
        "top_term_id": "best_term_id",
        "top_term_name": "best_term_name",
        "max_f1": "best_f1",
    })[["layer", "threshold", "latent", "best_term_id", "best_term_name", "best_f1"]].copy()

    # Preserve useful support columns when present.
    for c in ["tp", "pred_pos", "true_pos"]:
        if c in best.columns:
            out[c] = best[c]

    out.to_csv(out_csv, index=False)
    return out


def filter_long(
    df: pd.DataFrame,
    *,
    threshold: Optional[float],
    min_true_pos: Optional[int],
) -> pd.DataFrame:
    d = df.copy()
    if threshold is not None and d["threshold"].notna().any():
        d = d[np.isclose(d["threshold"].astype(float), float(threshold))].copy()
    if min_true_pos is not None and "true_pos" in d.columns and d["true_pos"].notna().any():
        d = d[pd.to_numeric(d["true_pos"], errors="coerce") >= int(min_true_pos)].copy()
    return d


def best_feature_per_concept(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=list(df.columns) + ["best_f1"])
    sort_cols = ["term_id", "f1"]
    ascending = [True, False]
    if "tp" in df.columns:
        sort_cols.append("tp")
        ascending.append(False)
    best = df.sort_values(sort_cols, ascending=ascending).groupby("term_id", as_index=False).first()
    best = best.rename(columns={"latent": "best_feature", "f1": "best_f1"})
    return best


def summarize_best_per_feature(best: pd.DataFrame, cutoffs: Sequence[float]) -> Dict[str, Any]:
    if best.empty:
        s = pd.Series(dtype=float)
    else:
        f1_col = "max_f1" if "max_f1" in best.columns else ("best_f1" if "best_f1" in best.columns else "f1")
        s = pd.to_numeric(best[f1_col], errors="coerce").dropna()

    out: Dict[str, Any] = {
        "n_features": int(len(s)),
        "mean_best_f1": float(s.mean()) if len(s) else np.nan,
        "median_best_f1": float(s.median()) if len(s) else np.nan,
        "p90_best_f1": float(s.quantile(0.90)) if len(s) else np.nan,
        "p95_best_f1": float(s.quantile(0.95)) if len(s) else np.nan,
        "max_best_f1": float(s.max()) if len(s) else np.nan,
    }
    for c in cutoffs:
        out[f"n_features_best_f1_ge_{c}"] = int((s >= float(c)).sum()) if len(s) else 0
    return out


def summarize_best_per_concept(best: pd.DataFrame, cutoffs: Sequence[float]) -> Dict[str, Any]:
    s = pd.to_numeric(best["best_f1"], errors="coerce").dropna() if not best.empty else pd.Series(dtype=float)
    out: Dict[str, Any] = {
        "n_concepts": int(best["term_id"].nunique()) if not best.empty else 0,
        "n_unique_best_features": int(best["best_feature"].nunique()) if not best.empty else 0,
        "mean_best_concept_f1": float(s.mean()) if len(s) else np.nan,
        "median_best_concept_f1": float(s.median()) if len(s) else np.nan,
        "p90_best_concept_f1": float(s.quantile(0.90)) if len(s) else np.nan,
        "p95_best_concept_f1": float(s.quantile(0.95)) if len(s) else np.nan,
        "max_best_concept_f1": float(s.max()) if len(s) else np.nan,
    }
    for c in cutoffs:
        out[f"n_concepts_best_f1_ge_{c}"] = int((s >= float(c)).sum()) if len(s) else 0
    return out


def plot_overlay_hist(
    sae_best: pd.DataFrame,
    act_best: pd.DataFrame,
    *,
    out_path: str | Path,
    title: str,
) -> None:
    def vals(df: pd.DataFrame) -> pd.Series:
        if "max_f1" in df.columns:
            return pd.to_numeric(df["max_f1"], errors="coerce").dropna()
        if "best_f1" in df.columns:
            return pd.to_numeric(df["best_f1"], errors="coerce").dropna()
        return pd.to_numeric(df["f1"], errors="coerce").dropna()

    plt.figure(figsize=(7, 4.5))
    plt.hist(vals(sae_best), bins=40, alpha=0.6, label="SAE")
    plt.hist(vals(act_best), bins=40, alpha=0.6, label="Embedding dims")
    plt.xlabel("Best F1 per feature")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_topk(
    sae_best: pd.DataFrame,
    act_best: pd.DataFrame,
    *,
    out_path: str | Path,
    title: str,
    topk: int = 100,
) -> None:
    def vals(df: pd.DataFrame) -> np.ndarray:
        col = "max_f1" if "max_f1" in df.columns else ("best_f1" if "best_f1" in df.columns else "f1")
        return pd.to_numeric(df[col], errors="coerce").dropna().sort_values(ascending=False).head(topk).to_numpy()

    s = vals(sae_best)
    a = vals(act_best)
    plt.figure(figsize=(7, 4.5))
    if len(s):
        plt.plot(np.arange(1, len(s) + 1), s, marker="o", markersize=3, label="SAE")
    if len(a):
        plt.plot(np.arange(1, len(a) + 1), a, marker="o", markersize=3, label="Embedding dims")
    plt.xlabel("Rank")
    plt.ylabel("Best F1")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_concept_usage_bar(
    best: pd.DataFrame,
    *,
    out_path: str | Path,
    title: str,
    topn: int,
) -> None:
    if best.empty:
        return
    concept_col = "top_term_name" if "top_term_name" in best.columns else ("term_name" if "term_name" in best.columns else "term_id")
    counts = best[concept_col].fillna(best.get("top_term_id", best.get("term_id", ""))).astype(str).value_counts().head(topn)
    plt.figure(figsize=(9, max(4, 0.28 * len(counts) + 1.5)))
    plt.barh(counts.index[::-1], counts.values[::-1])
    plt.xlabel("# features")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def count_grid(
    df: pd.DataFrame,
    *,
    thresholds: Sequence[float],
    min_true_pos_values: Sequence[int],
    f1_cutoff: float,
) -> pd.DataFrame:
    arr = np.zeros((len(thresholds), len(min_true_pos_values)), dtype=int)
    for i, thr in enumerate(thresholds):
        for j, mtp in enumerate(min_true_pos_values):
            d = filter_long(df, threshold=float(thr), min_true_pos=int(mtp))
            best = best_feature_per_concept(d)
            arr[i, j] = int((best["best_f1"] >= float(f1_cutoff)).sum()) if not best.empty else 0
    return pd.DataFrame(arr, index=[str(x) for x in thresholds], columns=[str(x) for x in min_true_pos_values])


def plot_grid(table: pd.DataFrame, *, out_path: str | Path, title: str, colorbar_label: str) -> None:
    arr = table.values
    plt.figure(figsize=(7, 5))
    im = plt.imshow(arr, aspect="auto")
    plt.xticks(range(table.shape[1]), table.columns.tolist())
    plt.yticks(range(table.shape[0]), table.index.tolist())
    plt.xlabel("min_true_pos")
    plt.ylabel("threshold")
    plt.title(title)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            plt.text(j, i, str(arr[i, j]), ha="center", va="center")
    plt.colorbar(im, label=colorbar_label)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def safe_tag(x: Any) -> str:
    return str(x).replace(".", "p").replace("/", "_").replace(" ", "_")


def analyze_one_model(
    *,
    label: str,
    layer: str,
    sae_csv: str,
    act_csv: str,
    term_meta_path: Optional[str],
    out_root: Path,
    sae_threshold: float,
    act_threshold: float,
    thresholds: Sequence[float],
    min_true_pos_values: Sequence[int],
    f1_cutoffs: Sequence[float],
    primary_min_true_pos: int,
    topk: int,
    topn_concepts: int,
    heatmap_max_words: int,
    heatmap_max_features: int,
    heatmap_max_concepts: int,
    sae_heatmap_f1_cutoff: float,
    act_heatmap_f1_cutoff: float,
) -> pd.DataFrame:
    model_dir = ensure_dir(out_root / label)
    table_dir = ensure_dir(model_dir / "tables")
    plot_dir = ensure_dir(model_dir / "plots")
    grid_dir = ensure_dir(model_dir / "threshold_min_true_pos_grids")

    print(f"[load] {label} SAE: {sae_csv}")
    sae_long = normalize_any_f1_csv(
        sae_csv,
        layer=layer,
        default_threshold=sae_threshold,
        term_meta_path=term_meta_path,
    )
    print(f"[load] {label} embedding: {act_csv}")
    act_long = normalize_any_f1_csv(
        act_csv,
        layer=layer,
        default_threshold=act_threshold,
        term_meta_path=term_meta_path,
    )

    sae_long.to_csv(table_dir / "sae_long.csv", index=False)
    act_long.to_csv(table_dir / "embedding_long.csv", index=False)

    sae_best_all = write_per_feature_best_for_plot(sae_long, out_csv=table_dir / "sae_per_feature_best.csv")
    act_best_all = write_per_feature_best_for_plot(act_long, out_csv=table_dir / "embedding_per_feature_best.csv")

    # Run your existing downstream plotting wrapper on the generated best-only inputs.
    # This reuses concept-feature heatmap, high-F1 histogram, overlay hist, top-k, concept counts.
    try:
        cfg = LatentsVsActsPlotConfig(
            runs_root=str(model_dir),
            layer=layer,
            sae_threshold=float(sae_threshold),
            act_threshold=float(act_threshold),
            sae_per_feature_best=str(table_dir / "sae_per_feature_best.csv"),
            act_best_path=str(table_dir / "embedding_per_feature_best.csv"),
            term_meta_path=str(term_meta_path or ""),
            outdir=str(model_dir / "repo_plot_outputs"),
            max_words=int(heatmap_max_words),
            max_features=int(heatmap_max_features),
            max_concepts=int(heatmap_max_concepts),
            sae_f1_cutoff_for_heatmap=float(sae_heatmap_f1_cutoff),
            acts_f1_cutoff_for_heatmap=float(act_heatmap_f1_cutoff),
            high_f1_cutoff=0.6,
            topn_concepts=int(topn_concepts),
        )
        plot_latents_vs_acts(cfg)
    except Exception as e:
        print(f"[warn] repo plot_latents_vs_acts failed for {label}: {e}")

    # Primary filtered summaries.
    sae_primary = filter_long(sae_long, threshold=sae_threshold, min_true_pos=primary_min_true_pos)
    act_primary = filter_long(act_long, threshold=act_threshold, min_true_pos=primary_min_true_pos)

    sae_best_primary = best_term_per_latent(sae_primary)
    act_best_primary = best_term_per_latent(act_primary)
    sae_best_concept = best_feature_per_concept(sae_primary)
    act_best_concept = best_feature_per_concept(act_primary)

    sae_best_primary.to_csv(table_dir / "sae_best_term_per_feature_primary.csv", index=False)
    act_best_primary.to_csv(table_dir / "embedding_best_term_per_feature_primary.csv", index=False)
    sae_best_concept.to_csv(table_dir / "sae_best_feature_per_concept_primary.csv", index=False)
    act_best_concept.to_csv(table_dir / "embedding_best_feature_per_concept_primary.csv", index=False)

    # Concept support tables using repo metric helper.
    for cutoff in f1_cutoffs:
        concept_support_table(sae_primary, f1_cutoff=float(cutoff)).to_csv(
            table_dir / f"sae_concept_support_f1_ge_{safe_tag(cutoff)}.csv", index=False
        )
        concept_support_table(act_primary, f1_cutoff=float(cutoff)).to_csv(
            table_dir / f"embedding_concept_support_f1_ge_{safe_tag(cutoff)}.csv", index=False
        )

    rows: List[Dict[str, Any]] = []
    for source, best_feat, best_conc in [
        ("SAE", sae_best_primary, sae_best_concept),
        ("embedding", act_best_primary, act_best_concept),
    ]:
        row: Dict[str, Any] = {
            "model": label,
            "layer": layer,
            "source": source,
            "primary_sae_threshold": float(sae_threshold),
            "primary_act_threshold": float(act_threshold),
            "primary_min_true_pos": int(primary_min_true_pos),
        }
        row.update(summarize_best_per_feature(best_feat, f1_cutoffs))
        row.update(summarize_best_per_concept(best_conc, f1_cutoffs))
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(table_dir / "summary_primary.csv", index=False)

    # Extra standalone plots.
    plot_overlay_hist(
        sae_best_primary,
        act_best_primary,
        out_path=plot_dir / "overlay_hist_best_f1_per_feature.png",
        title=f"{label}: best F1 per feature",
    )
    plot_topk(
        sae_best_primary,
        act_best_primary,
        out_path=plot_dir / "topk_best_f1_curve.png",
        title=f"{label}: top-{topk} best-F1 features",
        topk=topk,
    )
    plot_concept_usage_bar(
        sae_best_primary,
        out_path=plot_dir / "top_concepts_sae.png",
        title=f"{label}: top concepts by SAE feature count",
        topn=topn_concepts,
    )
    plot_concept_usage_bar(
        act_best_primary,
        out_path=plot_dir / "top_concepts_embedding.png",
        title=f"{label}: top concepts by embedding-dim count",
        topn=topn_concepts,
    )

    # Threshold × min_true_pos grids for SAE, embedding, and delta.
    grid_rows: List[Dict[str, Any]] = []
    for cutoff in f1_cutoffs:
        tag = safe_tag(cutoff)
        sae_grid = count_grid(sae_long, thresholds=thresholds, min_true_pos_values=min_true_pos_values, f1_cutoff=cutoff)
        act_grid = count_grid(act_long, thresholds=thresholds, min_true_pos_values=min_true_pos_values, f1_cutoff=cutoff)
        delta_grid = sae_grid.astype(int) - act_grid.astype(int)

        sae_grid.to_csv(grid_dir / f"sae_counts_f1_ge_{tag}.csv")
        act_grid.to_csv(grid_dir / f"embedding_counts_f1_ge_{tag}.csv")
        delta_grid.to_csv(grid_dir / f"delta_counts_f1_ge_{tag}.csv")

        plot_grid(sae_grid, out_path=grid_dir / f"sae_counts_f1_ge_{tag}.png",
                  title=f"{label}: SAE # concepts with best F1 ≥ {cutoff}", colorbar_label="# concepts")
        plot_grid(act_grid, out_path=grid_dir / f"embedding_counts_f1_ge_{tag}.png",
                  title=f"{label}: embedding # concepts with best F1 ≥ {cutoff}", colorbar_label="# concepts")
        plot_grid(delta_grid, out_path=grid_dir / f"delta_counts_f1_ge_{tag}.png",
                  title=f"{label}: SAE - embedding # concepts with best F1 ≥ {cutoff}", colorbar_label="delta # concepts")

        for thr in thresholds:
            for mtp in min_true_pos_values:
                grid_rows.append({
                    "model": label,
                    "layer": layer,
                    "f1_cutoff": float(cutoff),
                    "threshold": float(thr),
                    "min_true_pos": int(mtp),
                    "sae_count": int(sae_grid.loc[str(thr), str(mtp)]),
                    "embedding_count": int(act_grid.loc[str(thr), str(mtp)]),
                    "delta": int(delta_grid.loc[str(thr), str(mtp)]),
                })
    pd.DataFrame(grid_rows).to_csv(table_dir / "grid_summary_long.csv", index=False)

    with open(model_dir / "manifest.json", "w") as f:
        json.dump({
            "label": label,
            "layer": layer,
            "sae_csv": sae_csv,
            "embedding_csv": act_csv,
            "sae_threshold": float(sae_threshold),
            "act_threshold": float(act_threshold),
            "primary_min_true_pos": int(primary_min_true_pos),
            "thresholds": [float(x) for x in thresholds],
            "min_true_pos_values": [int(x) for x in min_true_pos_values],
            "f1_cutoffs": [float(x) for x in f1_cutoffs],
        }, f, indent=2)

    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Run downstream F1 SAE-vs-embedding analysis for 3 models.")
    ap.add_argument("--labels", nargs=3, required=True)
    ap.add_argument("--layers", nargs=3, required=True)
    ap.add_argument("--sae-score-csvs", nargs=3, required=True)
    ap.add_argument("--embedding-score-csvs", nargs=3, required=True)
    ap.add_argument("--out-root", required=True)

    ap.add_argument("--term-meta", default=None, help="gprofiler_terms.tsv, optional")
    ap.add_argument("--sae-threshold", type=float, default=0.15)
    ap.add_argument("--act-threshold", type=float, default=0.15)
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.0, 0.15, 0.3, 0.6])
    ap.add_argument("--min-true-pos-values", nargs="+", type=int, default=[1, 3, 5, 10])
    ap.add_argument("--primary-min-true-pos", type=int, default=3)
    ap.add_argument("--f1-cutoffs", nargs="+", type=float, default=[0.5, 0.6, 0.7])

    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--topn-concepts", type=int, default=20)
    ap.add_argument("--heatmap-max-words", type=int, default=8)
    ap.add_argument("--heatmap-max-features", type=int, default=60)
    ap.add_argument("--heatmap-max-concepts", type=int, default=60)
    ap.add_argument("--sae-heatmap-f1-cutoff", type=float, default=0.5)
    ap.add_argument("--embedding-heatmap-f1-cutoff", type=float, default=0.3)

    args = ap.parse_args()
    out_root = ensure_dir(args.out_root)

    summaries = []
    for label, layer, sae_csv, act_csv in zip(args.labels, args.layers, args.sae_score_csvs, args.embedding_score_csvs):
        print("\n" + "=" * 100)
        print(f"[analyze] model={label} layer={layer}")
        print("=" * 100)
        summaries.append(analyze_one_model(
            label=label,
            layer=layer,
            sae_csv=sae_csv,
            act_csv=act_csv,
            term_meta_path=args.term_meta,
            out_root=out_root,
            sae_threshold=args.sae_threshold,
            act_threshold=args.act_threshold,
            thresholds=args.thresholds,
            min_true_pos_values=args.min_true_pos_values,
            f1_cutoffs=args.f1_cutoffs,
            primary_min_true_pos=args.primary_min_true_pos,
            topk=args.topk,
            topn_concepts=args.topn_concepts,
            heatmap_max_words=args.heatmap_max_words,
            heatmap_max_features=args.heatmap_max_features,
            heatmap_max_concepts=args.heatmap_max_concepts,
            sae_heatmap_f1_cutoff=args.sae_heatmap_f1_cutoff,
            act_heatmap_f1_cutoff=args.embedding_heatmap_f1_cutoff,
        ))

    combined = pd.concat(summaries, ignore_index=True)
    combined.to_csv(out_root / "combined_summary_primary.csv", index=False)

    # Combined barplots for concept counts above F1 cutoffs.
    for cutoff in args.f1_cutoffs:
        col = f"n_concepts_best_f1_ge_{cutoff}"
        if col not in combined.columns:
            continue
        pivot = combined.pivot(index="model", columns="source", values=col)
        pivot.to_csv(out_root / f"combined_concepts_best_f1_ge_{safe_tag(cutoff)}.csv")

        x = np.arange(len(pivot.index))
        width = 0.35
        plt.figure(figsize=(8, 4.5))
        if "SAE" in pivot.columns:
            plt.bar(x - width / 2, pivot["SAE"].fillna(0).values, width, label="SAE")
        if "embedding" in pivot.columns:
            plt.bar(x + width / 2, pivot["embedding"].fillna(0).values, width, label="Embedding")
        plt.xticks(x, pivot.index.tolist())
        plt.ylabel("# concepts")
        plt.title(f"# concepts with best F1 ≥ {cutoff}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_root / f"combined_concepts_best_f1_ge_{safe_tag(cutoff)}.png", dpi=200)
        plt.close()

    print(f"\n[done] outputs written to: {out_root}")


if __name__ == "__main__":
    main()


# python test_downstream_f1_3models.py \
#   --labels scgpt c2sscale geneformer \
#   --layers layer_4.norm2 layer_17 layer_4 \
#   --sae-score-csvs \
#     /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/heldout_3models_l1_3e-3_geneheldout/scgpt/layer_4.norm2/test_concept_f1_scores.csv \
#     /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/heldout_3models_l1_3e-3_geneheldout/c2sscale/layer_17/test_concept_f1_scores.csv \
#     /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/heldout_3models_l1_3e-3_geneheldout/geneformer/layer_4/test_concept_f1_scores.csv \
#   --embedding-score-csvs \
#     /path/to/scgpt/activation_per_feature_best.csv \
#     /path/to/c2sscale/activation_per_feature_best.csv \
#     /path/to/geneformer/activation_per_feature_best.csv \
#   --term-meta /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/heldout_3models_l1_3e-3_geneheldout/gprofiler/gprofiler_terms.tsv \
#   --out-root /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/downstream_f1_sae_vs_embedding_3models \
#   --sae-threshold 0.15 \
#   --act-threshold 0.15 \
#   --primary-min-true-pos 3 \
#   --thresholds 0.0 0.15 0.3 0.6 \
#   --min-true-pos-values 1 3 5 10 \
#   --f1-cutoffs 0.5 0.6 0.7