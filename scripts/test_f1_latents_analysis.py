import os
import pandas as pd

from interp_pipeline.downstream.f1.metrics import (
    F1AnalysisConfig,
    load_f1_long_from_root,
)

RUNS_ROOT = "runs/full_scgpt_cosmx"
HELDOUT_ROOT = os.path.join(RUNS_ROOT, "heldout_report")
OUT_DIR = os.path.join(RUNS_ROOT, "f1_analysis")

# Use the heldout reporter output you inspected
PATTERNS = ["**/valid_concept_f1_scores.csv"]

TERM_META_PATH = os.path.join(RUNS_ROOT, "gprofiler", "gprofiler_terms.tsv")

CFG = F1AnalysisConfig(
    f1_cutoff=0.20,
    high_f1_latent_cutoff=0.35,
    top_n_concepts=20,
)

MAKE_PLOTS = True


def _load_term_meta(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[warn] term meta not found: {path}")
        return pd.DataFrame(columns=["term_id", "term_name"])
    meta = pd.read_csv(path, sep="\t")
    if "term_id" not in meta.columns:
        raise ValueError(f"term meta missing term_id: {list(meta.columns)}")
    if "term_name" not in meta.columns and "name" in meta.columns:
        meta = meta.rename(columns={"name": "term_name"})
    if "term_name" not in meta.columns:
        meta["term_name"] = ""
    return meta[["term_id", "term_name"]].drop_duplicates("term_id")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) Load f1 rows
    f1_long = load_f1_long_from_root(HELDOUT_ROOT, patterns=PATTERNS, skip_empty=True)

    # 2) Attach names
    term_meta = _load_term_meta(TERM_META_PATH)
    if len(term_meta):
        f1_long = f1_long.merge(term_meta, on="term_id", how="left")
        if "term_name_x" in f1_long.columns and "term_name_y" in f1_long.columns:
            f1_long["term_name"] = f1_long["term_name_x"]
            m = f1_long["term_name"].isna() | (f1_long["term_name"] == "None") | (f1_long["term_name"] == "")
            f1_long.loc[m, "term_name"] = f1_long.loc[m, "term_name_y"]
            f1_long = f1_long.drop(columns=["term_name_x", "term_name_y"])
    else:
        if "term_name" not in f1_long.columns:
            f1_long["term_name"] = None

    f1_long.to_parquet(os.path.join(OUT_DIR, "f1_long.parquet"), index=False)

    # 3) Best concept per feature(latent) per threshold per layer
    per_feature_best = (
        f1_long.sort_values(["layer", "threshold", "latent", "f1"], ascending=[True, True, True, False])
        .groupby(["layer", "threshold", "latent"], as_index=False, dropna=False)
        .first()
        .rename(columns={"term_id": "best_term_id", "term_name": "best_term_name", "f1": "best_f1"})
    )
    per_feature_best.to_csv(os.path.join(OUT_DIR, "per_feature_best.csv"), index=False)

    # 4) Best per feature overall (collapse thresholds)
    per_feature_best_overall = (
        per_feature_best.sort_values(["layer", "latent", "best_f1"], ascending=[True, True, False])
        .groupby(["layer", "latent"], as_index=False)
        .first()
        .rename(columns={"threshold": "best_threshold"})
    )
    per_feature_best_overall.to_csv(os.path.join(OUT_DIR, "per_feature_best_overall.csv"), index=False)

    high = per_feature_best_overall[per_feature_best_overall["best_f1"] >= CFG.high_f1_latent_cutoff].copy()
    high.to_csv(os.path.join(OUT_DIR, "high_f1_features.csv"), index=False)

    # 5) Summary table (layer x threshold)
    layer_thr_summary = (
        per_feature_best.groupby(["layer", "threshold"], as_index=False, dropna=False)
        .agg(
            n_features=("latent", "nunique"),
            mean_best_f1=("best_f1", "mean"),
            median_best_f1=("best_f1", "median"),
            p90_best_f1=("best_f1", lambda s: s.quantile(0.90)),
            max_best_f1=("best_f1", "max"),
            n_high=("best_f1", lambda s: int((s >= CFG.high_f1_latent_cutoff).sum())),
        )
        .sort_values(["layer", "threshold"])
    )
    layer_thr_summary.to_csv(os.path.join(OUT_DIR, "layer_threshold_summary.csv"), index=False)

    # 6) Concept support (how many features support a term above cutoff)
    concept_support = (
        f1_long[f1_long["f1"] >= float(CFG.f1_cutoff)]
        .groupby(["layer", "threshold", "term_id"], as_index=False, dropna=False)
        .agg(
            support=("latent", "nunique"),
            max_f1=("f1", "max"),
            sum_f1=("f1", "sum"),
        )
        .merge(term_meta, on="term_id", how="left")
        .sort_values(["layer", "threshold", "support", "max_f1"], ascending=[True, True, False, False])
    )
    concept_support.to_csv(os.path.join(OUT_DIR, "concept_support.csv"), index=False)

    # 7) Plots (notebook-style)
    if MAKE_PLOTS:
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"[warn] matplotlib unavailable, skipping plots: {e}")
        else:
            plot_dir = os.path.join(OUT_DIR, "plots")
            os.makedirs(plot_dir, exist_ok=True)

            # A) histogram of best_f1 per layer (overall best threshold)
            for layer in sorted(per_feature_best_overall["layer"].unique()):
                dfL = per_feature_best_overall[per_feature_best_overall["layer"] == layer]
                plt.figure(figsize=(6, 4))
                plt.hist(dfL["best_f1"].dropna(), bins=40)
                plt.xlabel("Best F1 per feature (across thresholds)")
                plt.ylabel("Count")
                plt.title(f"{layer}: distribution of best F1")
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, f"hist_best_f1_{layer}.png"), dpi=200)
                plt.close()

            # B) trend plot: median / max best_f1 vs threshold, one line per layer (too busy)
            # Better: aggregate across layers
            agg = (
                layer_thr_summary.groupby("threshold", as_index=False)
                .agg(
                    median_of_medians=("median_best_f1", "median"),
                    max_of_max=("max_best_f1", "max"),
                    mean_of_means=("mean_best_f1", "mean"),
                )
                .sort_values("threshold")
            )
            plt.figure(figsize=(6, 4))
            plt.plot(agg["threshold"], agg["median_of_medians"], marker="o", label="median(best_f1) across layers")
            plt.plot(agg["threshold"], agg["max_of_max"], marker="o", label="max(best_f1) across layers")
            plt.xlabel("Threshold")
            plt.ylabel("F1")
            plt.title("F1 vs threshold (aggregated across layers)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "trend_f1_vs_threshold.png"), dpi=200)
            plt.close()

            # C) top concepts by pooled support (across all layers/thresholds)
            pooled = (
                concept_support.groupby(["term_id", "term_name"], as_index=False)
                .agg(support=("support", "sum"), max_f1=("max_f1", "max"))
                .sort_values(["support", "max_f1"], ascending=False)
                .head(CFG.top_n_concepts)
            )
            plt.figure(figsize=(8, 5))
            y = pooled["term_name"].fillna(pooled["term_id"])
            plt.barh(y[::-1], pooled["support"][::-1])
            plt.xlabel(f"Total #features with F1 ≥ {CFG.f1_cutoff} (summed over layers/thresholds)")
            plt.title(f"Top {CFG.top_n_concepts} concepts by support")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "top_concepts_by_support.png"), dpi=200)
            plt.close()

    print(f"[OK] wrote outputs to: {OUT_DIR}")
    print(f"     f1_long rows: {len(f1_long):,}")
    print(f"     per_feature_best rows: {len(per_feature_best):,}")
    print(f"     per_feature_best_overall rows: {len(per_feature_best_overall):,}")
    print(f"     high rows: {len(high):,}")


if __name__ == "__main__":
    main()