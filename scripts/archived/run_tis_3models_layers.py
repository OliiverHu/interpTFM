#!/usr/bin/env python
from __future__ import annotations

import argparse

from interp_pipeline.tis.layer_screen import (
    TISLayerScreenParams,
    apply_layer_overrides,
    load_model_config,
    parse_layer_csv,
    run_layer_screen,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run flexible per-model/per-layer TIS screen using the 3-model backend.")
    p.add_argument("--config", default=None, help="Optional JSON config overriding default model roots/layers/pooling/tokens.")
    p.add_argument("--backend", default="scripts/test_tis_3models.py")
    p.add_argument("--out-root", default="runs/tis_layer_screen_3models_cosmx")
    p.add_argument("--adata-path", default="/maiziezhou_lab2/yunfei/Projects/FM_temp/interGFM/ge_shards/cosmx_human_lung_sec8.h5ad")
    p.add_argument("--models", nargs="+", default=["scgpt", "c2sscale", "geneformer"])
    p.add_argument("--scgpt-layers", default=None, help="Override scGPT layers, space/comma separated.")
    p.add_argument("--c2sscale-layers", default=None, help="Override C2S-scale layers, space/comma separated.")
    p.add_argument("--geneformer-layers", default=None, help="Override Geneformer layers, space/comma separated.")
    p.add_argument("--max-shards", default="NONE")
    p.add_argument("--K", default="32")
    p.add_argument("--n-trials", default="800")
    p.add_argument("--seed", default="42")
    p.add_argument("--q-high", default="0.75")
    p.add_argument("--q-low", default="0.25")
    p.add_argument("--no-pca", action="store_true")
    p.add_argument("--save-cell-activations", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--status-only", action="store_true", help="Report complete/incomplete/pending batches without running anything.")
    p.add_argument("--no-retry-incomplete", action="store_true", help="Do not rerun batches with an incomplete summary file.")
    p.add_argument("--clean-incomplete", action="store_true", help="Move incomplete batch directories aside before rerunning them.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    models = load_model_config(args.config)
    models = apply_layer_overrides(
        models,
        scgpt_layers=parse_layer_csv(args.scgpt_layers),
        c2sscale_layers=parse_layer_csv(args.c2sscale_layers),
        geneformer_layers=parse_layer_csv(args.geneformer_layers),
    )
    params = TISLayerScreenParams(
        out_root=args.out_root,
        adata_path=args.adata_path,
        backend=args.backend,
        max_shards=str(args.max_shards),
        K=str(args.K),
        n_trials=str(args.n_trials),
        seed=str(args.seed),
        q_high=str(args.q_high),
        q_low=str(args.q_low),
        no_pca=args.no_pca,
        save_cell_activations=args.save_cell_activations,
        force=args.force,
        dry_run=args.dry_run,
        status_only=args.status_only,
        retry_incomplete=not args.no_retry_incomplete,
        clean_incomplete=args.clean_incomplete,
    )
    run_layer_screen(models=models, selected_models=args.models, params=params)


if __name__ == "__main__":
    main()
