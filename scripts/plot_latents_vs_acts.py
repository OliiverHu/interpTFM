# scripts/plot_latents_vs_acts.py
from __future__ import annotations

import os

from interp_pipeline.downstream.f1.plot import (
    LatentsVsActsPlotConfig,
    plot_latents_vs_acts,
)

RUNS_ROOT = "runs/full_scgpt_cosmx"
LAYER = "layer_4.norm2"

SAE_THRESHOLD = 0.6
ACT_THRESHOLD = 0.3

cfg = LatentsVsActsPlotConfig(
    runs_root=RUNS_ROOT,
    layer=LAYER,
    sae_threshold=SAE_THRESHOLD,
    act_threshold=ACT_THRESHOLD,
    sae_per_feature_best=os.path.join(RUNS_ROOT, "f1_analysis", "per_feature_best.csv"),
    act_best_path=os.path.join(RUNS_ROOT, "activation_f1_baseline", LAYER, "activation_per_feature_best.csv"),
    term_meta_path=os.path.join(RUNS_ROOT, "gprofiler", "gprofiler_terms.tsv"),
    outdir=os.path.join(
        RUNS_ROOT,
        "f1_analysis",
        "compare_sae_vs_acts",
        LAYER,
        f"sae_{SAE_THRESHOLD}__acts_{ACT_THRESHOLD}",
    ),
    # keep your settings
    max_words=6,
    max_features=60,
    max_concepts=60,
    sae_f1_cutoff_for_heatmap=0.5,
    acts_f1_cutoff_for_heatmap=0.3,
    high_f1_cutoff=0.6,
    topn_concepts=20,
)

if __name__ == "__main__":
    plot_latents_vs_acts(cfg)