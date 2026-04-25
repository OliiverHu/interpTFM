#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# EDIT THESE PATHS
# =========================
MODEL_CSVS = {
    "scGPT": "/maiziezhou_lab2/yunfei/Projects/interpTFM/runs/heldout_3models_l1_3e-3_geneheldout/scgpt/layer_4.norm2/test_concept_f1_scores.csv",
    "c2sscale": "/maiziezhou_lab2/yunfei/Projects/interpTFM/runs/heldout_3models_l1_3e-3_geneheldout/c2sscale/layer_17/test_concept_f1_scores.csv",
    "geneformer": "/maiziezhou_lab2/yunfei/Projects/interpTFM/runs/heldout_3models_l1_3e-3_geneheldout/geneformer/layer_4/test_concept_f1_scores.csv",
}

OUT_ROOT = Path("/maiziezhou_lab2/yunfei/Projects/interpTFM/runs/heldout_3models_l1_3e-3_geneheldout/summary_heatmaps")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# x-axis
MIN_TRUE_POS_VALUES = [1, 2, 3, 5, 10]

# y-axis
LATENT_THRESHOLDS = [0.0, 0.15, 0.3, 0.6]

# one heatmap per cutoff
F1_CUTOFFS = [0.5, 0.6, 0.7]


# =========================
# CORE LOGIC
# =========================
def best_per_concept_count(
    df: pd.DataFrame,
    latent_threshold: float,
    min_true_pos: int,
    f1_cutoff: float,
) -> int:
    """
    Filter rows by threshold and true_pos, then keep the best feature per concept,
    and count how many concepts have best_f1 >= f1_cutoff.
    """
    d = df.copy()

    # threshold column name
    if "threshold_pct" in d.columns:
        thr_col = "threshold_pct"
    elif "threshold" in d.columns:
        thr_col = "threshold"
    else:
        raise ValueError(f"No threshold column found. Columns: {list(d.columns)}")

    required = ["concept", "feature", "f1", "true_pos"]
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Columns: {list(d.columns)}")

    d = d[d[thr_col].astype(float) == float(latent_threshold)].copy()
    d["true_pos"] = pd.to_numeric(d["true_pos"], errors="coerce")
    d["f1"] = pd.to_numeric(d["f1"], errors="coerce")

    d = d[d["true_pos"] >= int(min_true_pos)].copy()
    d = d.dropna(subset=["concept", "feature", "f1", "true_pos"])

    if d.empty:
        return 0

    # tie-break by tp if available
    sort_cols = ["concept", "f1"]
    ascending = [True, False]
    if "tp" in d.columns:
        d["tp"] = pd.to_numeric(d["tp"], errors="coerce").fillna(0)
        sort_cols.append("tp")
        ascending.append(False)

    best = (
        d.sort_values(sort_cols, ascending=ascending)
         .groupby("concept", as_index=False)
         .first()
    )

    return int((best["f1"] >= float(f1_cutoff)).sum())


def build_heatmap_table(
    df: pd.DataFrame,
    latent_thresholds: list[float],
    min_true_pos_values: list[int],
    f1_cutoff: float,
) -> pd.DataFrame:
    """
    Return a DataFrame:
      rows = latent thresholds
      cols = min_true_pos values
      values = kept concept counts
    """
    mat = np.zeros((len(latent_thresholds), len(min_true_pos_values)), dtype=int)

    for i, thr in enumerate(latent_thresholds):
        for j, min_tp in enumerate(min_true_pos_values):
            mat[i, j] = best_per_concept_count(
                df=df,
                latent_threshold=thr,
                min_true_pos=min_tp,
                f1_cutoff=f1_cutoff,
            )

    out = pd.DataFrame(
        mat,
        index=[str(x) for x in latent_thresholds],
        columns=[str(x) for x in min_true_pos_values],
    )
    out.index.name = "threshold"
    out.columns.name = "min_true_pos"
    return out


def plot_single_heatmap(
    table: pd.DataFrame,
    model_name: str,
    f1_cutoff: float,
    out_path: Path,
) -> None:
    arr = table.values

    plt.figure(figsize=(7, 5))
    im = plt.imshow(arr, aspect="auto")

    plt.xticks(range(table.shape[1]), table.columns.tolist())
    plt.yticks(range(table.shape[0]), table.index.tolist())
    plt.xlabel("min_true_pos")
    plt.ylabel("latent threshold")
    plt.title(f"{model_name}: # concepts with best F1 ≥ {f1_cutoff}")

    # annotate counts in each cell
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            plt.text(j, i, str(arr[i, j]), ha="center", va="center")

    plt.colorbar(im, label="# concepts")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_three_heatmaps(
    tables: dict[float, pd.DataFrame],
    model_name: str,
    out_path: Path,
) -> None:
    """
    One figure with 3 heatmaps side by side for F1 cutoffs 0.5 / 0.6 / 0.7.
    """
    cutoffs = list(tables.keys())
    fig = plt.figure(figsize=(16, 5))

    vmax = max(int(t.values.max()) for t in tables.values()) if tables else 1

    for k, cutoff in enumerate(cutoffs, start=1):
        table = tables[cutoff]
        arr = table.values

        ax = fig.add_subplot(1, len(cutoffs), k)
        im = ax.imshow(arr, aspect="auto", vmin=0, vmax=vmax)

        ax.set_xticks(range(table.shape[1]))
        ax.set_xticklabels(table.columns.tolist())
        ax.set_yticks(range(table.shape[0]))
        ax.set_yticklabels(table.index.tolist())
        ax.set_xlabel("min_true_pos")
        if k == 1:
            ax.set_ylabel("latent threshold")
        ax.set_title(f"best F1 ≥ {cutoff}")

        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                ax.text(j, i, str(arr[i, j]), ha="center", va="center")

    fig.suptitle(f"{model_name}: kept concept counts", fontsize=16)
    fig.colorbar(im, ax=fig.axes, shrink=0.85, label="# concepts")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


# =========================
# RUN
# =========================
def main():
    summary = {}

    for model_name, csv_path in MODEL_CSVS.items():
        csv_path = Path(csv_path)
        if not csv_path.exists():
            print(f"[skip] missing file for {model_name}: {csv_path}")
            continue

        print(f"[load] {model_name}: {csv_path}")
        df = pd.read_csv(csv_path)

        model_out = OUT_ROOT / model_name
        model_out.mkdir(parents=True, exist_ok=True)

        cutoff_tables = {}

        for cutoff in F1_CUTOFFS:
            table = build_heatmap_table(
                df=df,
                latent_thresholds=LATENT_THRESHOLDS,
                min_true_pos_values=MIN_TRUE_POS_VALUES,
                f1_cutoff=cutoff,
            )
            cutoff_tables[cutoff] = table

            # save table
            csv_out = model_out / f"heatmap_table_f1_ge_{str(cutoff).replace('.', 'p')}.csv"
            table.to_csv(csv_out)

            # save single heatmap
            png_out = model_out / f"heatmap_f1_ge_{str(cutoff).replace('.', 'p')}.png"
            plot_single_heatmap(
                table=table,
                model_name=model_name,
                f1_cutoff=cutoff,
                out_path=png_out,
            )

        # also save one combined figure with all 3 cutoffs
        combined_png = model_out / "heatmaps_f1_ge_0p5_0p6_0p7.png"
        plot_three_heatmaps(
            tables=cutoff_tables,
            model_name=model_name,
            out_path=combined_png,
        )

        summary[model_name] = {
            str(cutoff): cutoff_tables[cutoff].to_dict()
            for cutoff in F1_CUTOFFS
        }

    with open(OUT_ROOT / "heatmap_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[done] outputs written to: {OUT_ROOT}")


if __name__ == "__main__":
    main()