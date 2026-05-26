from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd


def layer_number(layer: str) -> int:
    nums = re.findall(r"\d+", str(layer))
    return int(nums[0]) if nums else 10**9


def find_summary_files(root: Path) -> List[Path]:
    return sorted(root.rglob("tis_summary_3models.csv"))


def aggregate_tis_layer_screen(tis_root: str | Path) -> pd.DataFrame:
    tis_root = Path(tis_root)
    files = find_summary_files(tis_root)
    if not files:
        raise FileNotFoundError(f"No tis_summary_3models.csv files found under {tis_root}")

    frames = []
    for f in files:
        df = pd.read_csv(f)
        df["source_file"] = str(f)
        df["run_dir"] = str(f.parent)
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)
    if "shuffle_mean" in all_df.columns and "tis_mean" in all_df.columns:
        all_df["tis_gap_mean"] = all_df["tis_mean"] - all_df["shuffle_mean"]
    else:
        all_df["tis_gap_mean"] = pd.NA

    token_col = "token_value" if "token_value" in all_df.columns else None
    subset = ["model", "layer", "pooling"] + ([token_col] if token_col else [])
    dedup = all_df.drop_duplicates(subset=subset, keep="first").copy()
    dedup["layer_num"] = dedup["layer"].map(layer_number)
    return dedup.sort_values(["model", "layer_num", "layer"])


def write_tis_layer_plots(
    tis_root: str | Path,
    out_dir: Optional[str | Path] = None,
    metric: str = "tis_mean",
) -> dict[str, Path]:
    tis_root = Path(tis_root)
    out_dir = Path(out_dir) if out_dir else tis_root / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    dedup = aggregate_tis_layer_screen(tis_root)
    if metric not in dedup.columns:
        raise ValueError(f"Metric {metric!r} not found. Columns: {list(dedup.columns)}")

    outputs: dict[str, Path] = {}
    combined_path = out_dir / "tis_layer_screen_combined.csv"
    best_path = out_dir / "tis_layer_screen_best_by_model.csv"
    dedup.to_csv(combined_path, index=False)
    best = dedup.sort_values(["model", metric], ascending=[True, False]).groupby("model", as_index=False).head(5)
    best.to_csv(best_path, index=False)
    outputs["combined_csv"] = combined_path
    outputs["best_csv"] = best_path

    for model, g in dedup.groupby("model", sort=True):
        g = g.sort_values(["layer_num", "layer"])
        fig = plt.figure(figsize=(max(6, 0.45 * len(g)), 4))
        plt.plot(g["layer"], g[metric], marker="o")
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Layer")
        plt.ylabel(metric)
        plt.title(f"{model}: {metric} by layer")
        plt.tight_layout()
        out = out_dir / f"{model}_{metric}_by_layer.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        outputs[f"{model}_plot"] = out

    fig = plt.figure(figsize=(10, 5))
    for model, g in dedup.groupby("model", sort=True):
        g = g.sort_values(["layer_num", "layer"])
        x = [f"{model}:{layer}" for layer in g["layer"]]
        plt.plot(x, g[metric], marker="o", label=model)
    plt.xticks(rotation=60, ha="right")
    plt.xlabel("Model:layer")
    plt.ylabel(metric)
    plt.title(f"TIS layer screen: {metric}")
    plt.legend()
    plt.tight_layout()
    out = out_dir / f"all_models_{metric}_by_layer.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    outputs["all_models_plot"] = out
    return outputs
