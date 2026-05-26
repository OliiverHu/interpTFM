from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Sequence

import pandas as pd


def collect_rows(tis_root: str | Path) -> pd.DataFrame:
    tis_root = Path(tis_root)
    rows: List[Dict[str, Any]] = []
    for p in sorted(tis_root.glob("*/*/summary_row.json")):
        try:
            row = json.loads(p.read_text())
            row["summary_path"] = str(p)
            row["out_dir"] = str(p.parent)
            rows.append(row)
        except Exception as e:
            print(f"[warn] failed reading {p}: {e}")
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "tis_mean" in df.columns and "shuffle_mean" in df.columns:
        df["tis_gap_mean"] = df["tis_mean"].astype(float) - df["shuffle_mean"].astype(float)
    if "tis_median" in df.columns and "shuffle_median" in df.columns:
        df["tis_gap_median"] = df["tis_median"].astype(float) - df["shuffle_median"].astype(float)
    return df.drop_duplicates(subset=["model", "layer"], keep="last")


def layer_order_key(layer: str):
    import re
    nums = re.findall(r"\d+", str(layer))
    return int(nums[0]) if nums else 10**9


def plot_layers(df: pd.DataFrame, out_dir: Path, metric: str) -> None:
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    for model, sub in df.groupby("model"):
        sub = sub.copy()
        sub["_order"] = sub["layer"].map(layer_order_key)
        sub = sub.sort_values(["_order", "layer"])
        plt.figure(figsize=(8, 4.5))
        plt.plot(sub["layer"].astype(str), sub[metric].astype(float), marker="o")
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Layer")
        plt.ylabel(metric)
        plt.title(f"{model}: {metric}")
        plt.tight_layout()
        plt.savefig(out_dir / f"{model}_{metric}.png", dpi=200)
        plt.close()


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Aggregate and plot one-job-per-layer TIS outputs.")
    p.add_argument("--tis-root", required=True)
    p.add_argument("--metric", default="tis_gap_mean")
    args = p.parse_args(argv)
    root = Path(args.tis_root)
    df = collect_rows(root)
    if df.empty:
        raise SystemExit(f"No summary_row.json files found under {root}")
    metric = args.metric
    if metric not in df.columns:
        raise SystemExit(f"Metric {metric!r} not found. Available columns include: {list(df.columns)}")
    df.to_csv(root / "tis_layer_screen_summary.csv", index=False)
    ranked = df.sort_values(["model", metric], ascending=[True, False])
    ranked.to_csv(root / "tis_layer_screen_ranked.csv", index=False)
    plot_layers(df, root / "plots", metric)
    print(f"wrote: {root / 'tis_layer_screen_summary.csv'}")
    print(f"wrote: {root / 'tis_layer_screen_ranked.csv'}")
    print(f"wrote plots under: {root / 'plots'}")


if __name__ == "__main__":
    main()
