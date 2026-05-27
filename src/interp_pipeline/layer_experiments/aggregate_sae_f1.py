from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd


def safe_read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def flatten(prefix: str, obj: Dict, out: Dict) -> None:
    for k, v in obj.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[f"{prefix}{k}"] = v


def find_tis_summary(tis_root: Optional[str]) -> pd.DataFrame:
    if not tis_root:
        return pd.DataFrame()
    root = Path(tis_root)
    # Prefer the corrected per-layer summary if it exists.
    for candidate in [root / "tis_layer_screen_summary.csv", root / "tis_layers_summary.csv", root / "summary.csv"]:
        if candidate.exists():
            df = pd.read_csv(candidate)
            if "model" in df.columns and "layer" in df.columns:
                return df
    rows = []
    for p in root.glob("**/summary_row.json"):
        obj = safe_read_json(p)
        if isinstance(obj, dict):
            rows.append(obj)
    for p in root.glob("**/tis_summary_3models.csv"):
        try:
            df = pd.read_csv(p)
            rows.extend(df.to_dict("records"))
        except Exception:
            pass
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "tis_gap_mean" not in df.columns and {"tis_mean", "shuffle_mean"}.issubset(df.columns):
        df["tis_gap_mean"] = df["tis_mean"] - df["shuffle_mean"]
    return df.drop_duplicates(subset=["model", "layer"], keep="last") if {"model", "layer"}.issubset(df.columns) else df


def collect(root: str, tis_root: Optional[str] = None, run_tag: Optional[str] = None) -> pd.DataFrame:
    base = Path(root)
    rows: List[Dict] = []
    for job_file in base.glob("*/*/.job.json"):
        job = safe_read_json(job_file)
        if not isinstance(job, dict):
            continue
        row = {"model": job.get("model"), "layer": job.get("layer"), "job_dir": str(job_file.parent)}
        # SAE summary
        sae_summaries = list((job_file.parent / "sae").glob("*/sae_*_summary.json"))
        if run_tag:
            sae_summaries = [p for p in sae_summaries if f"/sae/{run_tag}/" in str(p)]
        if sae_summaries:
            obj = safe_read_json(sorted(sae_summaries)[-1])
            if isinstance(obj, dict):
                flatten("sae_", obj, row)
                row["sae_summary_path"] = str(sorted(sae_summaries)[-1])
        # QC summary
        qc_summaries = list((job_file.parent / "activation_qc").glob("*/dead_neuron_summary_*.json"))
        if run_tag:
            qc_summaries = [p for p in qc_summaries if f"/activation_qc/{run_tag}/" in str(p)]
        if qc_summaries:
            obj = safe_read_json(sorted(qc_summaries)[-1])
            if isinstance(obj, list) and obj:
                obj = obj[0]
            if isinstance(obj, dict):
                flatten("qc_", obj, row)
                row["qc_summary_path"] = str(sorted(qc_summaries)[-1])
        # F1 summaries. Prefer the new cell-heldout files when present.
        f1_root = job_file.parent / "f1_heldout"
        if run_tag:
            f1_root = f1_root / run_tag
        if f1_root.exists():
            csvs = sorted(f1_root.glob("**/*.csv"))
            row["f1_dir"] = str(f1_root)
            row["f1_n_csv"] = len(csvs)
            cell_summary = f1_root / "counts_summary_cell.json"
            if cell_summary.exists():
                obj = safe_read_json(cell_summary)
                if isinstance(obj, dict):
                    flatten("cell_f1_", obj, row)
                    row["f1_mode"] = "cell_heldout"
            for split in ["valid", "test"]:
                p_cell = f1_root / f"{split}_cell_concept_f1_scores.csv"
                if p_cell.exists():
                    try:
                        dfc = pd.read_csv(p_cell)
                        if "f1" in dfc.columns:
                            vals = pd.to_numeric(dfc["f1"], errors="coerce")
                            row[f"cell_f1_{split}_max"] = float(vals.max())
                            row[f"cell_f1_{split}_mean"] = float(vals.mean())
                            row[f"cell_f1_{split}_n_ge_0p3"] = int((vals >= 0.3).sum())
                            row[f"cell_f1_{split}_n_ge_0p5"] = int((vals >= 0.5).sum())
                            row[f"cell_f1_{split}_n_concepts"] = int(vals.notna().sum())
                    except Exception:
                        pass
            # Fallback/generic extraction, useful for old gene-heldout files.
            best_f1 = None
            best_path = None
            for p in csvs:
                try:
                    df = pd.read_csv(p)
                except Exception:
                    continue
                f1_cols = [c for c in df.columns if c.lower() in {"f1", "test_f1", "best_f1", "f1_score"} or c.lower().endswith("f1")]
                for c in f1_cols:
                    vals = pd.to_numeric(df[c], errors="coerce")
                    if vals.notna().any():
                        val = float(vals.max())
                        if best_f1 is None or val > best_f1:
                            best_f1 = val
                            best_path = str(p)
            if best_f1 is not None:
                row["f1_best_detected"] = best_f1
                row["f1_best_detected_path"] = best_path
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    tis = find_tis_summary(tis_root)
    if not tis.empty and {"model", "layer"}.issubset(tis.columns):
        keep = [c for c in ["model", "layer", "tis_mean", "shuffle_mean", "pca_mean", "tis_gap_mean", "n_has_row", "activation_dim"] if c in tis.columns]
        df = df.merge(tis[keep].drop_duplicates(["model", "layer"]), on=["model", "layer"], how="left")
    return df


def plot_metric(df: pd.DataFrame, metric: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if metric not in df.columns:
        print(f"[plot] metric not found: {metric}")
        return
    for model, sub in df.groupby("model"):
        ss = sub.copy()
        ss[metric] = pd.to_numeric(ss[metric], errors="coerce")
        ss = ss.dropna(subset=[metric])
        if ss.empty:
            continue
        ss = ss.sort_values("layer")
        plt.figure(figsize=(8, 4))
        plt.plot(ss["layer"].astype(str), ss[metric], marker="o")
        plt.xticks(rotation=45, ha="right")
        plt.title(f"{model}: {metric}")
        plt.xlabel("layer")
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(out_dir / f"{model}__{metric}.png", dpi=160)
        plt.close()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Aggregate all-layer SAE/QC/F1 outputs and optionally merge TIS metrics.")
    ap.add_argument("--root", default="runs/all_layer_sae_f1_cosmx")
    ap.add_argument("--tis-root", default=None)
    ap.add_argument("--run-tag", default=None)
    ap.add_argument("--metric", default="qc_near_dead_frac")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    root = Path(args.root)
    df = collect(args.root, tis_root=args.tis_root, run_tag=args.run_tag)
    if df.empty:
        raise SystemExit(f"No job outputs found under {root}")
    out_csv = root / "all_layer_sae_f1_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    if args.metric in df.columns:
        rank = df.copy()
        rank[args.metric] = pd.to_numeric(rank[args.metric], errors="coerce")
        rank = rank.sort_values(["model", args.metric], ascending=[True, True])
        rank.to_csv(root / f"all_layer_sae_f1_ranked_by_{args.metric}.csv", index=False)
    plot_metric(df, args.metric, root / "plots")

if __name__ == "__main__":
    main()
