#!/usr/bin/env python3
# python debug_c2s_eval_mismatch.py   --mode both   --extraction-root /maiziezhou_lab2/yunfei/Projects/interpTFM/c2s_full_extraction   --store-root runs/full_c2sscale_cosmx   --layer layer_17   --adata /maiziezhou_lab2/yunfei/Projects/FM_temp/interGFM/ge_shards/cosmx_human_lung_sec8.h5ad   --gt-csv runs/full_c2sscale_cosmx/gprofiler/gprofiler_binary_gene_by_term.csv   --max-batches 10   --max-shards 10   --sample-rows 20
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import torch


def read_pairs_txt(path: Path) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            if "\t" in s:
                parts = s.split("\t")
            elif "," in s:
                parts = s.split(",")
            else:
                parts = s.split()
            if len(parts) < 2:
                raise ValueError(f"Malformed line {i} in {path}: {line!r}")
            pairs.append((parts[0].strip(), parts[1].strip()))
    return pairs


def load_tensor(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, dict):
        for k in ["acts", "activations", "tensor", "x"]:
            if k in obj and isinstance(obj[k], torch.Tensor):
                return obj[k]
    raise TypeError(f"Unsupported tensor payload in {path}: {type(obj)}")


def load_adata_mappings(adata_path: Path):
    adata = sc.read_h5ad(adata_path)
    ensembl_ids = adata.var_names.astype(str).tolist()
    if "feature_name" in adata.var.columns:
        gene_symbols = adata.var["feature_name"].astype(str).tolist()
    elif "index" in adata.var.columns:
        gene_symbols = adata.var["index"].astype(str).tolist()
    else:
        raise ValueError("Need adata.var['feature_name'] or adata.var['index'].")

    sym2ens: Dict[str, str] = {}
    sym2ens_upper: Dict[str, str] = {}
    panel_symbols = set(gene_symbols)
    ens_set = set(ensembl_ids)
    for sym, ens in zip(gene_symbols, ensembl_ids):
        sym = sym.strip()
        ens = ens.strip()
        if sym and ens and sym.lower() != "nan" and ens.lower() != "nan":
            sym2ens[sym] = ens
            sym2ens_upper[sym.upper()] = ens
    obs_names = set(adata.obs_names.astype(str).tolist())
    return sym2ens, sym2ens_upper, panel_symbols, ens_set, obs_names


def resolve_gene_to_ens(gene: str, sym2ens: Dict[str, str], sym2ens_upper: Dict[str, str], ens_set: set[str]):
    g = str(gene).strip()
    if not g:
        return None, "empty"
    if g in ens_set:
        return g, "already_ensg"
    if g in sym2ens:
        return sym2ens[g], "symbol_exact"
    if g.upper() in sym2ens_upper:
        return sym2ens_upper[g.upper()], "symbol_upper"
    return None, "unmapped"


def load_gt(gt_csv: Path):
    df = pd.read_csv(gt_csv, index_col=0)
    return set(df.index.astype(str).tolist()), list(df.columns.astype(str))


def iter_c2s_batches(extraction_root: Path, layer: str, max_batches: int | None = None):
    act_paths = sorted(
        Path(p) for p in glob.glob(str(extraction_root / "activations" / layer / "shard_*" / "batch_*_gene_acts.pt"))
    )
    if max_batches is not None:
        act_paths = act_paths[:max_batches]
    for act_path in act_paths:
        pair_path = act_path.with_name(act_path.name.replace("_gene_acts.pt", "_cell_gene_pairs.txt"))
        if not pair_path.exists():
            raise FileNotFoundError(f"Missing pair file for {act_path}")
        yield act_path, pair_path


def iter_generic_shards(store_root: Path, layer: str, max_shards: int | None = None):
    shard_dirs = sorted(Path(p) for p in glob.glob(str(store_root / "activations" / layer / "shard_*")))
    if max_shards is not None:
        shard_dirs = shard_dirs[:max_shards]
    for shard_dir in shard_dirs:
        act_path = shard_dir / "activations.pt"
        idx_path = shard_dir / "index.pt"
        if act_path.exists() and idx_path.exists():
            yield shard_dir, act_path, idx_path


def summarize_counts(title: str, counter: Counter, topk: int = 15):
    print(f"\n[{title}]")
    for k, v in counter.most_common(topk):
        print(f"  {k}: {v}")


def audit_original_c2s(extraction_root: Path, layer: str, adata_path: Path, gt_csv: Path, max_batches: int | None):
    sym2ens, sym2ens_upper, panel_symbols, ens_set, obs_names = load_adata_mappings(adata_path)
    gt_gene_ids, gt_term_ids = load_gt(gt_csv)

    total_rows = 0
    unique_cells = set()
    unique_genes = set()
    pair_counter = Counter()
    map_reason = Counter()
    overlap_reason = Counter()
    missing_panel = Counter()
    missing_cells = Counter()
    act_stats = []
    bad_batches = []

    panel_upper = {g.upper() for g in panel_symbols}

    for act_path, pair_path in iter_c2s_batches(extraction_root, layer, max_batches=max_batches):
        acts = load_tensor(act_path)
        pairs = read_pairs_txt(pair_path)

        if acts.ndim != 2:
            bad_batches.append({"batch": str(act_path), "error": f"acts ndim={acts.ndim}"})
            continue
        if acts.shape[0] != len(pairs):
            bad_batches.append({"batch": str(act_path), "error": f"row mismatch acts={acts.shape[0]} pairs={len(pairs)}"})
            continue

        row_norms = torch.norm(acts.float(), dim=1)
        act_stats.append({
            "batch": str(act_path),
            "rows": int(acts.shape[0]),
            "dim": int(acts.shape[1]),
            "mean_abs": float(acts.float().abs().mean().item()),
            "std": float(acts.float().std().item()),
            "min": float(acts.float().min().item()),
            "max": float(acts.float().max().item()),
            "zero_rows": int((row_norms == 0).sum().item()),
        })

        total_rows += acts.shape[0]
        for cell_id, gene in pairs:
            unique_cells.add(cell_id)
            unique_genes.add(gene)
            pair_counter[(cell_id, gene)] += 1

            if cell_id not in obs_names:
                missing_cells[cell_id] += 1
            if gene not in panel_symbols and gene.upper() not in panel_upper:
                missing_panel[gene] += 1

            ens, why = resolve_gene_to_ens(gene, sym2ens, sym2ens_upper, ens_set)
            map_reason[why] += 1
            if ens is None:
                overlap_reason["dropped_unmapped"] += 1
            elif ens in gt_gene_ids:
                overlap_reason["kept_overlap_gt"] += 1
            else:
                overlap_reason["mapped_but_not_in_gt"] += 1

    result = {
        "mode": "original_c2s",
        "layer": layer,
        "total_rows": total_rows,
        "unique_cells": len(unique_cells),
        "unique_genes": len(unique_genes),
        "duplicate_pairs": sum(1 for _, c in pair_counter.items() if c > 1),
        "gt_terms": len(gt_term_ids),
        "mapping_counts": dict(map_reason),
        "gt_overlap_counts": dict(overlap_reason),
        "missing_panel_top20": missing_panel.most_common(20),
        "missing_adata_cells_top20": missing_cells.most_common(20),
        "bad_batches": bad_batches[:20],
        "act_stats_first5": act_stats[:5],
    }

    print("\n=== ORIGINAL C2S AUDIT ===")
    print(json.dumps(result, indent=2))
    if total_rows:
        kept = overlap_reason["kept_overlap_gt"]
        print(f"\nkeep rate into GT = {kept}/{total_rows} = {kept / total_rows:.4f}")

    summarize_counts("mapping counts", map_reason)
    summarize_counts("GT overlap counts", overlap_reason)
    if missing_panel:
        summarize_counts("missing panel genes", missing_panel, topk=20)
    if missing_cells:
        summarize_counts("missing adata cells", missing_cells, topk=20)


def audit_generic_store(store_root: Path, layer: str, adata_path: Path, gt_csv: Path, max_shards: int | None):
    sym2ens, sym2ens_upper, panel_symbols, ens_set, obs_names = load_adata_mappings(adata_path)
    gt_gene_ids, gt_term_ids = load_gt(gt_csv)

    total_rows = 0
    map_reason = Counter()
    overlap_reason = Counter()
    idx_errors = []
    first_examples = []

    for shard_dir, act_path, idx_path in iter_generic_shards(store_root, layer, max_shards=max_shards):
        acts = torch.load(act_path, map_location="cpu")
        idx = torch.load(idx_path, map_location="cpu")
        token_ids = idx.get("token_ids")
        example_ids = idx.get("example_ids")

        if token_ids is None or example_ids is None:
            idx_errors.append({"shard": str(shard_dir), "error": "missing token_ids/example_ids"})
            continue
        if acts.shape[0] != len(token_ids) or acts.shape[0] != len(example_ids):
            idx_errors.append({
                "shard": str(shard_dir),
                "error": f"row mismatch acts={acts.shape[0]} token_ids={len(token_ids)} example_ids={len(example_ids)}",
            })
            continue

        total_rows += acts.shape[0]
        if not first_examples:
            first_examples = [{"row": i, "cell": str(example_ids[i]), "gene": str(token_ids[i])} for i in range(min(10, len(token_ids)))]

        for ex, tok in zip(example_ids, token_ids):
            ex = str(ex)
            tok = str(tok)

            if ex not in obs_names:
                overlap_reason["missing_cell_in_adata"] += 1

            ens, why = resolve_gene_to_ens(tok, sym2ens, sym2ens_upper, ens_set)
            map_reason[why] += 1
            if ens is None:
                overlap_reason["dropped_unmapped"] += 1
            elif ens in gt_gene_ids:
                overlap_reason["kept_overlap_gt"] += 1
            else:
                overlap_reason["mapped_but_not_in_gt"] += 1

    result = {
        "mode": "generic_store",
        "layer": layer,
        "total_rows": total_rows,
        "gt_terms": len(gt_term_ids),
        "mapping_counts": dict(map_reason),
        "gt_overlap_counts": dict(overlap_reason),
        "index_errors": idx_errors[:20],
        "first_examples": first_examples,
    }

    print("\n=== GENERIC STORE AUDIT ===")
    print(json.dumps(result, indent=2))
    if total_rows:
        kept = overlap_reason["kept_overlap_gt"]
        print(f"\nkeep rate into GT = {kept}/{total_rows} = {kept / total_rows:.4f}")

    summarize_counts("mapping counts", map_reason)
    summarize_counts("GT overlap counts", overlap_reason)


def compare_original_vs_generic(extraction_root: Path, store_root: Path, layer: str, sample_rows: int):
    print("\n=== ORIGINAL vs GENERIC COMPARISON ===")
    orig_batches = list(iter_c2s_batches(extraction_root, layer, max_batches=1))
    gen_shards = list(iter_generic_shards(store_root, layer, max_shards=1))
    if not orig_batches:
        print("No original c2s batch found.")
        return
    if not gen_shards:
        print("No generic shard found.")
        return

    act_path, pair_path = orig_batches[0]
    shard_dir, gen_act_path, gen_idx_path = gen_shards[0]

    orig_acts = load_tensor(act_path)
    orig_pairs = read_pairs_txt(pair_path)
    gen_acts = torch.load(gen_act_path, map_location="cpu")
    gen_idx = torch.load(gen_idx_path, map_location="cpu")
    token_ids = gen_idx["token_ids"]
    example_ids = gen_idx["example_ids"]

    n = min(sample_rows, len(orig_pairs), len(token_ids), orig_acts.shape[0], gen_acts.shape[0])
    mism_meta = 0
    mism_tensor = 0

    for i in range(n):
        cell, gene = orig_pairs[i]
        gcell = str(example_ids[i])
        ggene = str(token_ids[i])
        if cell != gcell or gene != ggene:
            mism_meta += 1
            print(f"META MISMATCH row={i}: original=({cell}, {gene}) generic=({gcell}, {ggene})")
        if not torch.equal(orig_acts[i], gen_acts[i]):
            mism_tensor += 1
            diff = (orig_acts[i].float() - gen_acts[i].float()).abs().max().item()
            print(f"TENSOR MISMATCH row={i}: max_abs_diff={diff}")

    print(json.dumps({
        "sample_rows_checked": n,
        "meta_mismatches": mism_meta,
        "tensor_mismatches": mism_tensor,
        "original_batch": str(act_path),
        "generic_shard": str(shard_dir),
    }, indent=2))


def main():
    ap = argparse.ArgumentParser(description="Audit c2s evaluation/input mismatch.")
    ap.add_argument("--mode", choices=["original", "generic", "both"], default="both")
    ap.add_argument("--extraction-root", type=Path, required=True)
    ap.add_argument("--store-root", type=Path, default=None)
    ap.add_argument("--layer", type=str, required=True)
    ap.add_argument("--adata", type=Path, required=True)
    ap.add_argument("--gt-csv", type=Path, required=True)
    ap.add_argument("--max-batches", type=int, default=10)
    ap.add_argument("--max-shards", type=int, default=10)
    ap.add_argument("--sample-rows", type=int, default=20)
    args = ap.parse_args()

    if args.mode in ("original", "both"):
        audit_original_c2s(args.extraction_root, args.layer, args.adata, args.gt_csv, args.max_batches)

    if args.mode in ("generic", "both"):
        if args.store_root is None:
            raise ValueError("--store-root is required for mode=generic or both")
        audit_generic_store(args.store_root, args.layer, args.adata, args.gt_csv, args.max_shards)
        compare_original_vs_generic(args.extraction_root, args.store_root, args.layer, args.sample_rows)


if __name__ == "__main__":
    main()
