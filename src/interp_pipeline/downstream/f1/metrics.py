from __future__ import annotations

import os
import glob
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd


@dataclass(frozen=True)
class F1AnalysisConfig:
    f1_cutoff: float = 0.20
    high_f1_latent_cutoff: float = 0.35
    top_n_concepts: int = 20


def _first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _infer_layer_from_path(path: str) -> str:
    parts = os.path.normpath(path).split(os.sep)
    if "heldout_report" in parts:
        i = parts.index("heldout_report")
        if i + 1 < len(parts):
            return parts[i + 1]
    return "unknown_layer"


def _parse_latent_series(s: pd.Series) -> pd.Series:
    """
    Accept ints OR strings like 'latent_123'/'k=123'/etc.
    Returns nullable Int64.
    """
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").astype("Int64")
    extracted = s.astype(str).str.extract(r"(-?\d+)", expand=False)
    return pd.to_numeric(extracted, errors="coerce").astype("Int64")


def normalize_f1_table(
    df: pd.DataFrame,
    *,
    layer: str,
    source_file: str,
    default_threshold: Optional[float] = None,
) -> pd.DataFrame:
    """
    Normalize a generic f1 table into a canonical long table.

    Supports BOTH:
    - heldout grid: concept/feature/threshold_pct/precision/recall/f1/tp/fp/fn
    - top_hits: term_id(or concept)/latent(or feature)/threshold/f1/tp/pred_pos/true_pos
    """
    # latent / feature id
    latent_col = _first_existing(df, ["latent", "feature", "k", "latent_id", "feature_id"])
    if latent_col is None:
        raise ValueError(f"Cannot find latent/feature column in {source_file}. Columns: {list(df.columns)}")

    # term/concept id
    term_id_col = _first_existing(df, ["term_id", "concept", "term", "concept_id", "native"])
    if term_id_col is None:
        raise ValueError(f"Cannot find term/concept column in {source_file}. Columns: {list(df.columns)}")

    # threshold (optional) — IMPORTANT: threshold_pct is your actual column
    thr_col = _first_existing(df, ["threshold", "thr", "latent_threshold", "threshold_pct"])
    if thr_col is None:
        df = df.copy()
        df["threshold"] = default_threshold
        thr_col = "threshold"

    # f1
    f1_col = _first_existing(df, ["f1", "F1", "max_F1", "f1_score"])
    if f1_col is None:
        raise ValueError(f"Cannot find f1 column in {source_file}. Columns: {list(df.columns)}")

    # optional term name
    term_name_col = _first_existing(df, ["term_name", "name", "concept_name"])

    # counts (optional)
    tp_col = _first_existing(df, ["tp", "TP"])
    pred_col = _first_existing(df, ["pred_pos", "predicted_pos", "n_pred", "pred"])
    true_col = _first_existing(df, ["true_pos", "n_true", "true"])

    out = pd.DataFrame(
        {
            "layer": layer,
            "threshold": pd.to_numeric(df[thr_col], errors="coerce"),
            "latent": _parse_latent_series(df[latent_col]),
            "term_id": df[term_id_col].astype(str),
            "term_name": df[term_name_col].astype(str) if term_name_col else None,
            "f1": pd.to_numeric(df[f1_col], errors="coerce"),
            "tp": pd.to_numeric(df[tp_col], errors="coerce") if tp_col else None,
            "pred_pos": pd.to_numeric(df[pred_col], errors="coerce") if pred_col else None,
            "true_pos": pd.to_numeric(df[true_col], errors="coerce") if true_col else None,
            "source_file": source_file,
        }
    )

    # critical: do NOT allow NaN latent or f1
    out = out.dropna(subset=["latent", "f1", "term_id"]).copy()
    out["latent"] = out["latent"].astype(int)
    return out


def load_f1_long_from_root(
    root: str,
    *,
    patterns: List[str],
    skip_empty: bool = True,
) -> pd.DataFrame:
    """
    Recursively load CSVs from root matching patterns, normalize, concat.
    """
    files: List[str] = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(root, pat), recursive=True))
    files = sorted(set(files))

    if not files:
        raise FileNotFoundError(f"No files matched under {root} with patterns={patterns}")

    frames: List[pd.DataFrame] = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
        except Exception:
            df = pd.read_csv(fp, index_col=0)

        if skip_empty and len(df) == 0:
            continue

        layer = _infer_layer_from_path(fp)
        frames.append(normalize_f1_table(df, layer=layer, source_file=fp))

    if not frames:
        return pd.DataFrame(columns=["layer","threshold","latent","term_id","term_name","f1","tp","pred_pos","true_pos","source_file"])

    return pd.concat(frames, ignore_index=True)


def best_term_per_latent(f1_long: pd.DataFrame) -> pd.DataFrame:
    """
    For each (layer, threshold, latent), keep row with max f1.
    IMPORTANT: dropna=False so threshold NaNs don't nuke everything.
    """
    if f1_long.empty:
        return f1_long.copy()

    df = f1_long.sort_values(
        ["layer", "threshold", "latent", "f1"],
        ascending=[True, True, True, False],
    )
    best = df.groupby(["layer", "threshold", "latent"], as_index=False, dropna=False).first()
    best = best.rename(columns={"term_id": "top_term_id", "term_name": "top_term_name", "f1": "max_f1"})
    return best


def concept_support_table(f1_long: pd.DataFrame, *, f1_cutoff: float) -> pd.DataFrame:
    """
    Per (layer, threshold, term_id): how many latents link to this term with f1>=cutoff.
    """
    if f1_long.empty:
        return pd.DataFrame(columns=["layer","threshold","term_id","support","max_f1","sum_f1"])

    df = f1_long.dropna(subset=["f1"]).copy()
    df = df[df["f1"] >= float(f1_cutoff)]
    if df.empty:
        return pd.DataFrame(columns=["layer","threshold","term_id","support","max_f1","sum_f1"])

    g = df.groupby(["layer", "threshold", "term_id"], as_index=False, dropna=False)
    out = g.agg(
        support=("latent", "nunique"),
        max_f1=("f1", "max"),
        sum_f1=("f1", "sum"),
    ).sort_values(["layer","threshold","support","max_f1","sum_f1"], ascending=[True, True, False, False, False])
    return out