"""F1-alignment result analysis utilities.

This is a cleaned, framework-friendly port of the plotting + checking code from
`tissue_inference.ipynb` (F1 concept–feature mapping / coverage plots).

It assumes you already computed and saved a long-form table with (at least)
columns:
  - feature: latent id/name (string or int)
  - concept: term name/id (string)
  - f1: float

Typical source in this repo is `top_hits_<layer>.csv` (or equivalent).

The functions here are intentionally lightweight: no dependency on the rest of
`interp_pipeline` besides standard Python scientific stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class F1AnalysisConfig:
    f1_cutoff: float = 0.50
    top_n_concepts: int = 20
    top_m_features: Optional[int] = None  # None => keep all used
    max_words_in_concept: int = 6
    wrap_width: int = 30
    max_xticks: int = 40


def load_f1_table(path: str | Path, *, index_col: Optional[int] = None) -> pd.DataFrame:
    """Load a long-form F1 table.

    Expected columns include: feature, concept, f1.
    Extra columns are preserved.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path, index_col=index_col)
    required = {"feature", "concept", "f1"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"F1 table is missing required columns: {sorted(missing)}")
    return df


def word_count(name: str) -> int:
    s = str(name).replace("_", " ").replace("-", " ")
    return len([w for w in s.split() if w])


def build_f1_matrix(
    f1_df: pd.DataFrame,
    *,
    f1_cutoff: float,
    max_words_in_concept: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build a concept×feature dense matrix (values are F1, zeros otherwise).

    Returns:
      - M: concept×feature pivot table
      - dropped: a 1-col DF of dropped concept names (possibly empty)
    """
    subset = f1_df.loc[f1_df["f1"] >= float(f1_cutoff)].copy()
    M = subset.pivot_table(index="concept", columns="feature", values="f1", fill_value=0.0)

    dropped = pd.DataFrame({"dropped_concept": []})
    if max_words_in_concept is not None:
        keep = [c for c in M.index if word_count(c) <= int(max_words_in_concept)]
        drop = sorted(set(M.index) - set(keep))
        dropped = pd.DataFrame({"dropped_concept": drop})
        M = M.loc[keep]

    return M, dropped


def rank_concepts(M: pd.DataFrame) -> pd.DataFrame:
    """Compute concept ranking stats used for selecting top concepts."""
    if M.empty:
        return pd.DataFrame(columns=["support", "maxF1", "sumF1"])

    stats = pd.DataFrame(
        {
            "support": (M > 0).sum(axis=1),
            "maxF1": M.max(axis=1),
            "sumF1": M.sum(axis=1),
        }
    ).sort_values(["support", "maxF1", "sumF1"], ascending=False)
    return stats


def select_top_matrix(
    M: pd.DataFrame,
    *,
    top_n_concepts: int,
    top_m_features: Optional[int] = None,
) -> pd.DataFrame:
    """Select a smaller matrix for plotting (top concepts + optionally top features)."""
    if M.empty:
        return M

    concept_stats = rank_concepts(M)
    keep_rows = concept_stats.index[: int(top_n_concepts)]
    M_rows = M.loc[keep_rows]

    cols_used = M_rows.columns[(M_rows > 0).any(axis=0)]

    if top_m_features is None:
        keep_cols = cols_used
    else:
        col_stats = (M_rows > 0).sum(axis=0).sort_values(ascending=False)
        keep_cols = col_stats.index[: int(top_m_features)]

    return M_rows.loc[:, keep_cols]


def feature_top_concept_summary(f1_df: pd.DataFrame) -> pd.DataFrame:
    """For each feature, pick the concept row with max f1."""
    # stable tie-breaker: highest f1 then first row order
    s = (
        f1_df.sort_values(["feature", "f1"], ascending=[True, False])
        .groupby("feature", as_index=False)
        .first()[["feature", "f1", "concept"]]
        .rename(columns={"f1": "max_F1", "concept": "top_concept"})
        .sort_values("max_F1", ascending=False)
    )
    return s


def check_f1_table(f1_df: pd.DataFrame) -> dict:
    """Lightweight sanity checks; returns a dict report."""
    report: dict = {}

    report["n_rows"] = int(len(f1_df))
    report["n_features"] = int(f1_df["feature"].nunique())
    report["n_concepts"] = int(f1_df["concept"].nunique())

    # Range checks
    f1 = pd.to_numeric(f1_df["f1"], errors="coerce")
    report["f1_nan"] = int(f1.isna().sum())
    report["f1_min"] = float(np.nanmin(f1.to_numpy())) if len(f1) else float("nan")
    report["f1_max"] = float(np.nanmax(f1.to_numpy())) if len(f1) else float("nan")
    report["f1_out_of_range"] = int(((f1 < -1e-6) | (f1 > 1 + 1e-6)).sum())

    # Duplicate (feature, concept)
    report["duplicate_pairs"] = int(f1_df.duplicated(subset=["feature", "concept"]).sum())

    return report


def plot_heatmap(
    M_plot: pd.DataFrame,
    *,
    outpath: str | Path,
    title: str,
    wrap_width: int = 30,
    max_xticks: int = 40,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> Path:
    """Plot and save a grayscale heatmap (PDF/SVG/PNG supported by extension)."""
    import textwrap

    import matplotlib.pyplot as plt
    import seaborn as sns

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    sns.set_theme(context="paper", style="white", font_scale=1.0)

    n_rows, n_cols = M_plot.shape
    panel_scale = 1.35
    height = float(np.clip(0.36 * n_rows * panel_scale, 3.0, 22.0))
    width = float(np.clip(0.48 * n_cols * panel_scale, 6.0, 30.0))

    fig, ax = plt.subplots(figsize=(width, height))
    sns.heatmap(
        M_plot,
        ax=ax,
        cmap="Greys",
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": "F1 (concept ↔ feature)", "shrink": 0.9, "pad": 0.02},
        square=False,
        linewidths=0,
    )

    ax.set_xlabel("Feature (latent)")
    ax.set_ylabel("Concept")
    ax.set_title(title, pad=10)

    yt = ax.get_yticklabels()
    wrapped_y = [textwrap.fill(t.get_text(), wrap_width) for t in yt]
    ax.set_yticklabels(wrapped_y, rotation=0, va="center")

    every_xtick = max(1, int(np.ceil(n_cols / max_xticks)))
    for i, lab in enumerate(ax.get_xticklabels()):
        lab.set_visible(i % every_xtick == 0)
    ax.tick_params(axis="x", rotation=45, labelrotation=45)

    fig.subplots_adjust(left=0.18, right=0.98, top=0.92, bottom=0.20)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)

    return outpath


def plot_coverage_sweep(
    f1_df: pd.DataFrame,
    *,
    outpath: str | Path,
    thresholds: Optional[Sequence[float]] = None,
) -> Path:
    """Plot how many concepts/features remain as threshold increases."""
    import matplotlib.pyplot as plt

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 50)

    concept_counts = []
    feature_counts = []
    for t in thresholds:
        sub = f1_df.loc[f1_df["f1"] > float(t)]
        concept_counts.append(int(sub["concept"].nunique()))
        feature_counts.append(int(sub["feature"].nunique()))

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(thresholds, concept_counts, marker="o", lw=1.5, label="# Concepts")
    ax.plot(thresholds, feature_counts, marker="s", lw=1.5, label="# Features")
    ax.set_xlabel("F1 threshold")
    ax.set_ylabel("Count")
    ax.set_title("Concept and feature coverage vs. F1 threshold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()

    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)

    return outpath


def run_f1_analysis(
    f1_csv_path: str | Path,
    *,
    outdir: str | Path,
    cfg: F1AnalysisConfig = F1AnalysisConfig(),
) -> dict:
    """End-to-end analysis: checks + coverage sweep + heatmap + summary table.

    Outputs (inside outdir):
      - sanity_report.json
      - dropped_concepts.csv (if any)
      - coverage_sweep.pdf
      - concept_feature_heatmap.pdf
      - feature_top_concept.csv

    Returns a dict with the key output paths.
    """
    import json

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    f1_df = load_f1_table(f1_csv_path)
    report = check_f1_table(f1_df)
    (outdir / "sanity_report.json").write_text(json.dumps(report, indent=2))

    cov_path = plot_coverage_sweep(f1_df, outpath=outdir / "coverage_sweep.pdf")

    M, dropped = build_f1_matrix(
        f1_df,
        f1_cutoff=cfg.f1_cutoff,
        max_words_in_concept=cfg.max_words_in_concept,
    )
    if len(dropped):
        dropped.to_csv(outdir / f"dropped_concepts_gt{cfg.max_words_in_concept}words.csv", index=False)

    M_plot = select_top_matrix(M, top_n_concepts=cfg.top_n_concepts, top_m_features=cfg.top_m_features)

    hm_path = plot_heatmap(
        M_plot,
        outpath=outdir / "concept_feature_heatmap.pdf",
        title=f"Concept–feature mapping (F1 ≥ {cfg.f1_cutoff:.2f})",
        wrap_width=cfg.wrap_width,
        max_xticks=cfg.max_xticks,
    )

    summary = feature_top_concept_summary(f1_df)
    summary_path = outdir / "feature_top_concept.csv"
    summary.to_csv(summary_path, index=False)

    return {
        "outdir": str(outdir),
        "sanity_report": str(outdir / "sanity_report.json"),
        "coverage_plot": str(cov_path),
        "heatmap": str(hm_path),
        "summary": str(summary_path),
    }
