#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_arr(path: str | Path) -> np.ndarray:
    arr = np.load(path)
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return arr


def summarize(arr: np.ndarray) -> dict:
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
    }


def plot_one_model(label: str, layer: str, model_dir: Path, out_dir: Path, bins: int) -> list[dict]:
    paths = {
        "model": model_dir / "tis_model.npy",
        "shuffle": model_dir / "tis_model_shuffled.npy",
        "pca": model_dir / "tis_model_pca.npy",
    }

    arrays = {}
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name} TIS array: {path}")
        arrays[name] = load_arr(path)

    rows = []
    for name, arr in arrays.items():
        row = {
            "model": label,
            "layer": layer,
            "distribution": name,
            **summarize(arr),
        }
        rows.append(row)

    plt.figure(figsize=(7, 4.8))
    for name in ["model", "pca", "shuffle"]:
        plt.hist(
            arrays[name],
            bins=bins,
            density=True,
            alpha=0.35,
            label=f"{name} mean={np.mean(arrays[name]):.3f}",
        )
    plt.xlabel("TIS")
    plt.ylabel("Density")
    plt.title(f"{label} {layer}: TIS distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{label}_{layer.replace('/', '_')}_tis_distribution.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4.8))
    data = [arrays["model"], arrays["pca"], arrays["shuffle"]]
    plt.boxplot(data, labels=["model", "PCA", "shuffle"], showfliers=False)
    plt.ylabel("TIS")
    plt.title(f"{label} {layer}: TIS distribution")
    plt.tight_layout()
    plt.savefig(out_dir / f"{label}_{layer.replace('/', '_')}_tis_boxplot.png", dpi=200)
    plt.close()

    return rows


def plot_combined(summary_df: pd.DataFrame, out_dir: Path) -> None:
    labels = summary_df["model"].unique().tolist()
    dists = ["model", "pca", "shuffle"]

    pivot = summary_df.pivot(index="model", columns="distribution", values="mean").loc[labels]

    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(8, 4.8))
    for i, dist in enumerate(dists):
        if dist not in pivot.columns:
            continue
        plt.bar(x + (i - 1) * width, pivot[dist].values, width, label=dist)
    plt.xticks(x, labels)
    plt.ylabel("Mean TIS")
    plt.title("TIS distribution mean: model vs PCA vs shuffle")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "combined_tis_mean_bar.png", dpi=200)
    plt.close()

    delta_rows = []
    for model in labels:
        row = {"model": model}
        row["model_minus_shuffle"] = float(pivot.loc[model, "model"] - pivot.loc[model, "shuffle"])
        row["model_minus_pca"] = float(pivot.loc[model, "model"] - pivot.loc[model, "pca"])
        delta_rows.append(row)
    delta = pd.DataFrame(delta_rows)

    plt.figure(figsize=(8, 4.8))
    x = np.arange(len(labels))
    plt.bar(x - width / 2, delta["model_minus_shuffle"], width, label="model - shuffle")
    plt.bar(x + width / 2, delta["model_minus_pca"], width, label="model - PCA")
    plt.xticks(x, labels)
    plt.ylabel("Mean TIS difference")
    plt.title("TIS improvement over baselines")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "combined_tis_delta_bar.png", dpi=200)
    plt.close()

    delta.to_csv(out_dir / "combined_tis_delta_summary.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot TIS distribution comparison for 3 models.")
    ap.add_argument("--labels", nargs=3, required=True)
    ap.add_argument("--layers", nargs=3, required=True)
    ap.add_argument("--tis-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--bins", type=int, default=50)
    args = ap.parse_args()

    tis_root = Path(args.tis_root)
    out_dir = ensure_dir(args.out_dir)

    rows = []
    for label, layer in zip(args.labels, args.layers):
        model_dir = tis_root / label / layer.replace("/", "_")
        rows.extend(plot_one_model(label, layer, model_dir, out_dir, bins=args.bins))

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_dir / "tis_distribution_summary.csv", index=False)

    plot_combined(summary_df, out_dir)

    print("[OK] wrote:", out_dir)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()


# python test_tis_3models_plot.py \
#   --labels scgpt c2sscale geneformer \
#   --layers layer_4.norm2 layer_17 layer_4 \
#   --tis-root /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/tis_3models \
#   --out-dir /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/tis_3models/plots_distribution \
#   --bins 80