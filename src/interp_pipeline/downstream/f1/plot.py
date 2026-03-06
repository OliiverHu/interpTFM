# src/interp_pipeline/downstream/f1/plot.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class LatentsVsActsPlotConfig:
    runs_root: str
    layer: str
    sae_threshold: float
    act_threshold: float

    sae_per_feature_best: str
    act_best_path: str
    term_meta_path: str

    outdir: str

    # heatmap selection
    max_words: int = 6
    max_features: int = 60
    max_concepts: int = 60

    # f1 cutoffs (can differ)
    sae_f1_cutoff_for_heatmap: float = 0.5
    acts_f1_cutoff_for_heatmap: float = 0.3

    # histogram
    high_f1_cutoff: float = 0.6

    # top concepts
    topn_concepts: int = 20


# -------------------------
# Term mapping
# -------------------------
def load_term_meta(term_meta_path: str) -> pd.DataFrame:
    if not os.path.exists(term_meta_path):
        return pd.DataFrame(columns=["term_id", "term_name"])
    meta = pd.read_csv(term_meta_path, sep="\t")
    if "term_name" not in meta.columns and "name" in meta.columns:
        meta = meta.rename(columns={"name": "term_name"})
    if "term_name" not in meta.columns:
        meta["term_name"] = ""
    if "term_id" not in meta.columns and "concept_id" in meta.columns:
        meta = meta.rename(columns={"concept_id": "term_id"})
    return meta[["term_id", "term_name"]].drop_duplicates("term_id")


def map_term_id_to_name(series_term_id: pd.Series, term_meta_path: str) -> pd.Series:
    meta = load_term_meta(term_meta_path)
    if meta.empty:
        return series_term_id.astype(str)
    m = meta.set_index("term_id")["term_name"].to_dict()
    return series_term_id.astype(str).map(lambda x: m.get(str(x), str(x)))


# -------------------------
# Input adapters (best-only)
# -------------------------
def build_f1_df_latents_best_only(
    sae_per_feature_best_csv: str,
    *,
    layer: str,
    sae_threshold: float,
    term_meta_path: str,
) -> pd.DataFrame:
    """
    per_feature_best.csv -> schema: feature, concept, f1 (best-only)
    """
    if not os.path.exists(sae_per_feature_best_csv):
        raise FileNotFoundError(f"Missing SAE best csv: {sae_per_feature_best_csv}")

    df = pd.read_csv(sae_per_feature_best_csv)
    needed = {"layer", "threshold", "latent", "best_term_id", "best_f1"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"{sae_per_feature_best_csv} missing columns {missing}. have={list(df.columns)}")

    df = df[(df["layer"] == layer) & (df["threshold"] == float(sae_threshold))].copy()
    if df.empty:
        raise RuntimeError(f"No SAE rows for layer={layer}, thr={sae_threshold} in {sae_per_feature_best_csv}")

    if "best_term_name" in df.columns:
        df["concept"] = df["best_term_name"].fillna(df["best_term_id"]).astype(str)
    else:
        df["concept"] = map_term_id_to_name(df["best_term_id"].astype(str), term_meta_path)

    f1_df = df.rename(columns={"latent": "feature", "best_f1": "f1"})[["feature", "concept", "f1"]].copy()
    f1_df["feature"] = f1_df["feature"].astype(int)
    f1_df["f1"] = f1_df["f1"].astype(float)
    return f1_df


def build_acts_best_only_f1_df(
    act_best_path: str,
    *,
    act_threshold: float,
    term_meta_path: str,
) -> pd.DataFrame:
    """
    activation_per_feature_best.csv -> schema: feature, concept, f1 (best-only)
    """
    if not os.path.exists(act_best_path):
        raise FileNotFoundError(f"Missing ACT baseline: {act_best_path}")

    act = pd.read_csv(act_best_path)

    # filter activation threshold if present
    if "threshold" in act.columns:
        act = act[act["threshold"] == float(act_threshold)].copy()

    act["concept"] = map_term_id_to_name(act["best_term_id"].astype(str), term_meta_path)
    f1_df = act.rename(columns={"best_f1": "f1"})[["feature", "concept", "f1"]].copy()
    f1_df["feature"] = f1_df["feature"].astype(int)
    f1_df["f1"] = f1_df["f1"].astype(float)
    return f1_df


# -------------------------
# Plotting helpers
# -------------------------
def plot_concept_feature_heatmap(
    f1_df: pd.DataFrame,
    *,
    outdir: str,
    title: str,
    f1_cutoff: float,
    tag: str,
    max_words: int,
    max_features: int,
    max_concepts: int,
) -> None:
    import textwrap
    import seaborn as sns
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)

    def word_count(s):
        return len(str(s).split())

    # wider wrap helps long GO names remain readable
    def wrap(s, width=65):
        return "\n".join(textwrap.wrap(str(s), width=width))

    df = f1_df.copy()
    df = df[df["f1"] >= float(f1_cutoff)]
    df = df[df["concept"].apply(word_count) <= int(max_words)]

    if len(df) == 0:
        print(f"[heatmap:{tag}] No rows after filter: f1 >= {f1_cutoff} & words <= {max_words}")
        return

    top_features = df.groupby("feature")["f1"].max().sort_values(ascending=False).head(int(max_features)).index
    top_concepts = df.groupby("concept")["f1"].max().sort_values(ascending=False).head(int(max_concepts)).index
    df = df[df["feature"].isin(top_features) & df["concept"].isin(top_concepts)]

    pivot = df.pivot_table(index="concept", columns="feature", values="f1", fill_value=0.0)
    pivot.index = [wrap(x, width=65) for x in pivot.index]

    # dynamic sizing: more concepts -> taller fig
    n_rows = pivot.shape[0]
    fig_h = max(7.0, min(18.0, 0.22 * n_rows + 4.5))
    fig_w = 11.5

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
        }
    )

    plt.figure(figsize=(fig_w, fig_h))
    ax = sns.heatmap(
        pivot,
        cmap="Greys",
        vmin=0.0,
        vmax=float(pivot.values.max()) if pivot.values.size else 1.0,
        cbar_kws={"label": "F1"},
        linewidths=0.2,
        linecolor="white",
    )
    ax.set_title(title)
    ax.set_xlabel("feature")
    ax.set_ylabel("concept")

    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", labelsize=8)

    plt.tight_layout()
    plt.subplots_adjust(left=0.40)  # extra room for long labels

    plt.savefig(os.path.join(outdir, f"concept_feature_heatmap_{tag}.png"))
    plt.savefig(os.path.join(outdir, f"concept_feature_heatmap_{tag}.pdf"))
    plt.savefig(os.path.join(outdir, f"concept_feature_heatmap_{tag}.svg"))
    plt.close()


def plot_high_f1_histogram(
    f1_df: pd.DataFrame,
    *,
    outdir: str,
    title: str,
    filename: str,
    high_f1_cutoff: float,
) -> pd.DataFrame:
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)

    f1_summary = f1_df.sort_values("f1", ascending=False).groupby("feature").first()
    summary = (
        f1_summary[["f1", "concept"]]
        .rename(columns={"f1": "max_F1", "concept": "top_concept"})
        .sort_values("max_F1", ascending=False)
    )
    high = summary.query("max_F1 > @high_f1_cutoff")

    print(f"[hist] Features with max_F1 > {high_f1_cutoff}: {len(high)}")

    plt.figure(figsize=(6, 4))
    plt.hist(high["max_F1"], bins=20)
    plt.xlabel("Max F1 per feature")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, filename), dpi=200)
    plt.close()

    return summary


def run_extra_comparison_analysis(
    sae_df: pd.DataFrame,
    acts_df: pd.DataFrame,
    *,
    outdir: str,
    plot_dir: str,
    layer: str,
    sae_threshold: float,
    act_threshold: float,
    topn_concepts: int,
) -> None:
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    sae = sae_df.rename(columns={"f1": "best_f1"})[["feature", "concept", "best_f1"]].copy()
    sae["source"] = "SAE"
    acts = acts_df.rename(columns={"f1": "best_f1"})[["feature", "concept", "best_f1"]].copy()
    acts["source"] = "ACTS"

    summary = pd.concat([sae, acts], ignore_index=True).groupby("source", as_index=False).agg(
        n_features=("feature", "nunique"),
        mean_best_f1=("best_f1", "mean"),
        median_best_f1=("best_f1", "median"),
        p90_best_f1=("best_f1", lambda s: s.quantile(0.90)),
        max_best_f1=("best_f1", "max"),
        n_gt_06=("best_f1", lambda s: int((s >= 0.6).sum())),
    )
    summary.to_csv(os.path.join(outdir, "summary.csv"), index=False)

    sae_counts = sae["concept"].value_counts().rename("count_sae").to_frame()
    act_counts = acts["concept"].value_counts().rename("count_acts").to_frame()
    counts = (
        sae_counts.join(act_counts, how="outer")
        .fillna(0)
        .astype(int)
        .reset_index()
        .rename(columns={"index": "concept"})
    )
    counts["count_total"] = counts["count_sae"] + counts["count_acts"]
    counts.to_csv(os.path.join(outdir, "concept_counts.csv"), index=False)

    import matplotlib.pyplot as plt

    # overlay hist
    plt.figure(figsize=(6, 4))
    plt.hist(sae["best_f1"].dropna(), bins=40, alpha=0.6, label="SAE latents")
    plt.hist(acts["best_f1"].dropna(), bins=40, alpha=0.6, label="Raw activation dims")
    plt.xlabel("Best F1 per feature")
    plt.ylabel("Count")
    plt.title(f"Best-F1 distribution — {layer} (SAE {sae_threshold} vs ACTS {act_threshold})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "hist_overlay_best_f1.png"), dpi=200)
    plt.close()

    # top-k overlay
    topk = 50
    sae_top = sae.sort_values("best_f1", ascending=False).head(topk)["best_f1"].values
    act_top = acts.sort_values("best_f1", ascending=False).head(topk)["best_f1"].values
    k = min(len(sae_top), len(act_top))
    plt.figure(figsize=(7, 4))
    plt.plot(range(k), sae_top[:k], marker="o", label="SAE top-k")
    plt.plot(range(k), act_top[:k], marker="o", label="ACTS top-k")
    plt.xlabel("Rank")
    plt.ylabel("Best F1")
    plt.title(f"Top-{k} best-F1 features — {layer}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "topk_overlay.png"), dpi=200)
    plt.close()

    # split top concepts
    topN = int(topn_concepts)

    cc_sae = counts.sort_values(["count_sae", "count_total"], ascending=False).head(topN).copy()
    plt.figure(figsize=(8, 5))
    plt.barh(cc_sae["concept"][::-1], cc_sae["count_sae"].values[::-1])
    plt.xlabel("Count (SAE)")
    plt.title(f"Top {topN} concepts by usage — SAE — {layer}")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "top_concepts_sae.png"), dpi=200)
    plt.close()

    cc_act = counts.sort_values(["count_acts", "count_total"], ascending=False).head(topN).copy()
    plt.figure(figsize=(8, 5))
    plt.barh(cc_act["concept"][::-1], cc_act["count_acts"].values[::-1])
    plt.xlabel("Count (ACTS)")
    plt.title(f"Top {topN} concepts by usage — ACTS — {layer}")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "top_concepts_acts.png"), dpi=200)
    plt.close()

    print("[extra] wrote comparison tables + plots to:", outdir)
    print(summary.to_string(index=False))


def plot_latents_vs_acts(cfg: LatentsVsActsPlotConfig) -> None:
    os.makedirs(cfg.outdir, exist_ok=True)
    plot_dir = os.path.join(cfg.outdir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    sae_dir = os.path.join(cfg.outdir, "SAE")
    acts_dir = os.path.join(cfg.outdir, "ACTS")
    os.makedirs(sae_dir, exist_ok=True)
    os.makedirs(acts_dir, exist_ok=True)

    # Build inputs
    f1_df_lat = build_f1_df_latents_best_only(
        cfg.sae_per_feature_best,
        layer=cfg.layer,
        sae_threshold=cfg.sae_threshold,
        term_meta_path=cfg.term_meta_path,
    )
    f1_df_act = build_acts_best_only_f1_df(
        cfg.act_best_path,
        act_threshold=cfg.act_threshold,
        term_meta_path=cfg.term_meta_path,
    )

    # Heatmaps (same settings; different cutoffs)
    plot_concept_feature_heatmap(
        f1_df_lat,
        outdir=sae_dir,
        title=f"Concept–Feature mapping (SAE, best-only) — {cfg.layer} thr={cfg.sae_threshold} (F1≥{cfg.sae_f1_cutoff_for_heatmap})",
        f1_cutoff=cfg.sae_f1_cutoff_for_heatmap,
        tag="sae",
        max_words=cfg.max_words,
        max_features=cfg.max_features,
        max_concepts=cfg.max_concepts,
    )
    plot_concept_feature_heatmap(
        f1_df_act,
        outdir=acts_dir,
        title=f"Concept–Feature mapping (ACTS, best-only) — {cfg.layer} (F1≥{cfg.acts_f1_cutoff_for_heatmap})",
        f1_cutoff=cfg.acts_f1_cutoff_for_heatmap,
        tag="acts",
        max_words=cfg.max_words,
        max_features=cfg.max_features,
        max_concepts=cfg.max_concepts,
    )

    # Histograms
    plot_high_f1_histogram(
        f1_df_lat,
        outdir=sae_dir,
        title="Distribution of interpretability (max F1 > 0.6) — SAE (best-only)",
        filename="hist_high_f1.png",
        high_f1_cutoff=cfg.high_f1_cutoff,
    )
    plot_high_f1_histogram(
        f1_df_act,
        outdir=acts_dir,
        title="Distribution of interpretability (max F1 > 0.6) — ACTS (best-only)",
        filename="hist_high_f1.png",
        high_f1_cutoff=cfg.high_f1_cutoff,
    )

    # Extra comparisons + split concept plots
    run_extra_comparison_analysis(
        sae_df=f1_df_lat,
        acts_df=f1_df_act,
        outdir=cfg.outdir,
        plot_dir=plot_dir,
        layer=cfg.layer,
        sae_threshold=cfg.sae_threshold,
        act_threshold=cfg.act_threshold,
        topn_concepts=cfg.topn_concepts,
    )

    print(f"\n[OK] wrote everything under: {cfg.outdir}")