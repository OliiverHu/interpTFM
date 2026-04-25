#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from interp_pipeline.downstream.f1.plot import (
    LatentsVsActsPlotConfig,
    plot_latents_vs_acts,
)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description="3-model SAE-vs-activation F1 comparison plots.")
    ap.add_argument("--labels", nargs=3, required=True)
    ap.add_argument("--runs-roots", nargs=3, required=True)
    ap.add_argument("--layers", nargs=3, required=True)

    ap.add_argument("--sae-threshold", type=float, default=0.15)
    ap.add_argument("--act-threshold", type=float, default=0.15)

    ap.add_argument("--sae-per-feature-best", nargs=3, default=None)
    ap.add_argument("--act-best-paths", nargs=3, default=None)
    ap.add_argument("--term-meta-paths", nargs=3, default=None)
    ap.add_argument("--outdirs", nargs=3, default=None)

    ap.add_argument("--max-words", type=int, default=6)
    ap.add_argument("--max-features", type=int, default=60)
    ap.add_argument("--max-concepts", type=int, default=60)
    ap.add_argument("--sae-f1-cutoff-for-heatmap", type=float, default=0.5)
    ap.add_argument("--acts-f1-cutoff-for-heatmap", type=float, default=0.3)
    ap.add_argument("--high-f1-cutoff", type=float, default=0.6)
    ap.add_argument("--topn-concepts", type=int, default=20)
    args = ap.parse_args()

    for i, (label, runs_root, layer) in enumerate(zip(args.labels, args.runs_roots, args.layers)):
        runs_root = str(runs_root)

        sae_path = (
            args.sae_per_feature_best[i]
            if args.sae_per_feature_best is not None
            else os.path.join(runs_root, "f1_analysis", "per_feature_best.csv")
        )
        act_path = (
            args.act_best_paths[i]
            if args.act_best_paths is not None
            else os.path.join(runs_root, "activation_f1_baseline", layer, "activation_per_feature_best.csv")
        )
        term_meta_path = (
            args.term_meta_paths[i]
            if args.term_meta_paths is not None
            else os.path.join(runs_root, "gprofiler", "gprofiler_terms.tsv")
        )
        outdir = (
            args.outdirs[i]
            if args.outdirs is not None
            else os.path.join(
                runs_root,
                "f1_analysis",
                "compare_sae_vs_acts",
                layer,
                f"sae_{args.sae_threshold}__acts_{args.act_threshold}",
            )
        )

        ensure_dir(outdir)

        print("=" * 100)
        print(f"[compare SAE vs acts] {label}")
        print(f"  sae_per_feature_best={sae_path}")
        print(f"  act_best_path={act_path}")
        print(f"  term_meta_path={term_meta_path}")
        print(f"  outdir={outdir}")
        print("=" * 100)

        cfg = LatentsVsActsPlotConfig(
            runs_root=runs_root,
            layer=layer,
            sae_threshold=float(args.sae_threshold),
            act_threshold=float(args.act_threshold),
            sae_per_feature_best=sae_path,
            act_best_path=act_path,
            term_meta_path=term_meta_path,
            outdir=outdir,
            max_words=int(args.max_words),
            max_features=int(args.max_features),
            max_concepts=int(args.max_concepts),
            sae_f1_cutoff_for_heatmap=float(args.sae_f1_cutoff_for_heatmap),
            acts_f1_cutoff_for_heatmap=float(args.acts_f1_cutoff_for_heatmap),
            high_f1_cutoff=float(args.high_f1_cutoff),
            topn_concepts=int(args.topn_concepts),
        )
        plot_latents_vs_acts(cfg)

    print("\nDONE")


if __name__ == "__main__":
    main()


# python plot_f1_latents_vs_acts_3models.py \
#   --labels scgpt c2sscale geneformer \
#   --runs-roots \
#     /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/full_scgpt_cosmx \
#     /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/full_c2sscale_cosmx \
#     /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/full_geneformer_cosmx \
#   --layers layer_4.norm2 layer_17 layer_4 \
#   --sae-threshold 0.15 \
#   --act-threshold 0.15 \
#   --term-meta-paths \
#     /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/heldout_3models_l1_3e-3_geneheldout/gprofiler/gprofiler_terms.tsv \
#     /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/heldout_3models_l1_3e-3_geneheldout/gprofiler/gprofiler_terms.tsv \
#     /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/heldout_3models_l1_3e-3_geneheldout/gprofiler/gprofiler_terms.tsv