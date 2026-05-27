#!/usr/bin/env python
"""Run one model/layer cell-heldout F1 directly.

This wrapper bypasses the all-layer runner and calls
interp_pipeline.layer_experiments.cell_heldout_f1.run_cell_heldout_f1
with explicit model/layer/store/SAE paths.

For scGPT cell rows, use --activation-pooling token --token-value 60695.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np


def _parse_thresholds(xs: Sequence[str]) -> Tuple[float, ...]:
    return tuple(float(x) for x in xs)


def _maybe_make_cell_subset_h5ad(adata_path: str, out_dir: str, max_cells: Optional[int], seed: int) -> str:
    """Create a deterministic cell-subset h5ad for scalable cell-F1 runs."""
    if max_cells is None or int(max_cells) <= 0:
        return adata_path

    import scanpy as sc

    src = Path(adata_path)
    out_tmp = Path(out_dir) / "_tmp"
    out_tmp.mkdir(parents=True, exist_ok=True)
    dst = out_tmp / f"adata_cell_subset_n{int(max_cells)}_seed{int(seed)}.h5ad"
    manifest = out_tmp / f"adata_cell_subset_n{int(max_cells)}_seed{int(seed)}.json"

    if dst.is_file():
        print(f"[subset] reusing existing subset h5ad: {dst}", flush=True)
        return str(dst)

    print(f"[subset] reading AnnData: {src}", flush=True)
    adata = sc.read_h5ad(src)
    n = int(adata.n_obs)
    k = min(int(max_cells), n)
    rng = np.random.default_rng(int(seed))
    idx = np.sort(rng.choice(n, size=k, replace=False))
    sub = adata[idx].copy()
    print(f"[subset] writing {k}/{n} cells to: {dst}", flush=True)
    sub.write_h5ad(dst)
    manifest.write_text(json.dumps({
        "source_adata": str(src),
        "subset_adata": str(dst),
        "seed": int(seed),
        "max_cells_requested": int(max_cells),
        "n_source_cells": n,
        "n_subset_cells": int(sub.n_obs),
    }, indent=2, sort_keys=True))
    return str(dst)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run cell-heldout F1 for exactly one model/layer using an existing SAE checkpoint."
    )
    ap.add_argument("--model", required=True, help="Model label, e.g. scgpt")
    ap.add_argument("--layer", required=True, help="Layer name, e.g. layer_4.norm2")
    ap.add_argument("--store-root", required=True, help="Activation store root")
    ap.add_argument("--sae-ckpt", required=True, help="SAE checkpoint path")
    ap.add_argument("--gt-csv", required=True, help="gprofiler_binary_gene_by_term.csv")
    ap.add_argument("--adata-path", required=True, help="AnnData h5ad path")
    ap.add_argument("--out-dir", required=True, help="Output directory for cell-heldout F1")

    ap.add_argument("--token-value", default=None, help="Token id used as the cell row. For scGPT use 60695.")
    ap.add_argument("--activation-pooling", choices=["token", "mean"], default="token")
    ap.add_argument("--c2s-cell-pooling", choices=["mean", "max"], default="mean")

    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--valid-frac", type=float, default=0.2)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--concept-score-quantile", type=float, default=0.75)
    ap.add_argument("--latent-thresholds", nargs="+", default=["0.0", "0.15", "0.3", "0.6"])

    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--device", default="cuda")

    ap.add_argument("--min-concept-genes", type=int, default=3)
    ap.add_argument("--min-pos-valid", type=int, default=20)
    ap.add_argument("--min-pos-test", type=int, default=20)
    ap.add_argument("--min-neg-valid", type=int, default=20)
    ap.add_argument("--min-neg-test", type=int, default=20)
    ap.add_argument("--max-concepts", type=int, default=None)
    ap.add_argument("--concept-chunk-size", type=int, default=25, help="Number of concepts per chunk during valid threshold search.")
    ap.add_argument("--max-cells", type=int, default=None, help="Deterministically subset AnnData to this many cells before cell-F1. No limit if omitted.")
    ap.add_argument("--no-dedupe-identical-concepts", action="store_true")
    args = ap.parse_args()

    token_value: Optional[str] = args.token_value
    if args.activation_pooling == "token" and token_value is None and args.model.lower() == "scgpt":
        token_value = "60695"
        print("[INFO] --token-value not supplied; using scGPT default token_value=60695", flush=True)

    if args.activation_pooling == "token" and token_value is None:
        raise SystemExit("--activation-pooling token requires --token-value")

    for label, path in [
        ("store_root", args.store_root),
        ("sae_ckpt", args.sae_ckpt),
        ("gt_csv", args.gt_csv),
        ("adata_path", args.adata_path),
    ]:
        p = Path(path)
        if label == "store_root":
            if not p.exists():
                raise FileNotFoundError(f"{label} does not exist: {p}")
        else:
            if not p.is_file():
                raise FileNotFoundError(f"{label} does not exist or is not a file: {p}")

    adata_path_for_run = _maybe_make_cell_subset_h5ad(
        adata_path=args.adata_path,
        out_dir=args.out_dir,
        max_cells=args.max_cells,
        seed=args.split_seed,
    )

    from interp_pipeline.layer_experiments.cell_heldout_f1 import (
        CellHeldoutF1Config,
        run_cell_heldout_f1,
    )

    cfg = CellHeldoutF1Config(
        model=args.model,
        layer=args.layer,
        store_root=args.store_root,
        sae_ckpt_path=args.sae_ckpt,
        gt_csv=args.gt_csv,
        adata_path=adata_path_for_run,
        out_dir=args.out_dir,
        token_value=str(token_value) if token_value is not None else None,
        activation_pooling=args.activation_pooling,
        c2s_cell_pooling=args.c2s_cell_pooling,
        split_seed=args.split_seed,
        train_frac=args.train_frac,
        valid_frac=args.valid_frac,
        test_frac=args.test_frac,
        concept_score_quantile=args.concept_score_quantile,
        latent_thresholds=_parse_thresholds(args.latent_thresholds),
        batch_size=args.batch_size,
        max_shards=args.max_shards,
        device=args.device,
        min_concept_genes=args.min_concept_genes,
        min_pos_valid=args.min_pos_valid,
        min_pos_test=args.min_pos_test,
        min_neg_valid=args.min_neg_valid,
        min_neg_test=args.min_neg_test,
        max_concepts=args.max_concepts,
        concept_chunk_size=args.concept_chunk_size,
        dedupe_identical_concepts=not args.no_dedupe_identical_concepts,
    )

    print("=" * 100, flush=True)
    print(f"[one-layer cell F1] model={cfg.model} layer={cfg.layer}", flush=True)
    print(f"store_root={cfg.store_root}", flush=True)
    print(f"sae_ckpt={cfg.sae_ckpt_path}", flush=True)
    print(f"gt_csv={cfg.gt_csv}", flush=True)
    print(f"adata_path={cfg.adata_path}", flush=True)
    print(f"out_dir={cfg.out_dir}", flush=True)
    print(f"activation_pooling={cfg.activation_pooling} token_value={cfg.token_value}", flush=True)
    print(f"latent_thresholds={cfg.latent_thresholds}", flush=True)
    print(f"concept_chunk_size={cfg.concept_chunk_size}", flush=True)
    print("=" * 100, flush=True)

    summary = run_cell_heldout_f1(cfg)
    print("[OK] one-layer cell-heldout F1 complete", flush=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
