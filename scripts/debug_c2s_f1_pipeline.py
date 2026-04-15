#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from collections import Counter

import numpy as np
import scanpy as sc
import scipy.sparse as sp
import torch


def read_pairs_txt(path: Path):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            s = line.strip()
            if not s:
                continue
            # tolerate tab/comma/space separators
            if "\t" in s:
                parts = s.split("\t")
            elif "," in s:
                parts = s.split(",")
            else:
                parts = s.split()

            if len(parts) < 2:
                raise ValueError(f"Malformed line {i+1} in {path}: {line!r}")

            cell_id = parts[0].strip()
            gene = parts[1].strip()
            pairs.append((cell_id, gene))
    return pairs


def load_pt_tensor(path: Path):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, dict):
        # try common keys
        for k in ["acts", "activations", "tensor", "x"]:
            if k in obj and isinstance(obj[k], torch.Tensor):
                return obj[k]
    raise TypeError(f"Unsupported tensor payload in {path}: {type(obj)}")


def normalize_var_names(adata):
    # prefer feature_name if present, else existing var_names
    if "feature_name" in adata.var.columns:
        genes = adata.var["feature_name"].astype(str).tolist()
    else:
        genes = adata.var_names.astype(str).tolist()
    return genes


def build_sym2ens_from_adata(adata):
    var = adata.var.copy()
    var_names = normalize_var_names(adata)

    sym2ens = {}
    ens_col = None
    for c in ["ensembl_id", "ensembl", "gene_id", "feature_id"]:
        if c in var.columns:
            ens_col = c
            break

    if ens_col is not None:
        ens_vals = var[ens_col].astype(str).tolist()
        for sym, ens in zip(var_names, ens_vals):
            sym = str(sym).strip()
            ens = str(ens).strip()
            if sym and ens and ens.lower() != "nan":
                sym2ens[sym] = ens
    return sym2ens


def summarize_counter(name, counter, topk=15):
    print(f"\n[{name}] top {topk}")
    for k, v in counter.most_common(topk):
        print(f"  {k}: {v}")


def inspect_batch(
    batch_gene_acts: Path,
    batch_pairs: Path,
    adata_path: Path,
    converted_shard: Path | None,
    sample_n: int,
):
    print("=== INPUTS ===")
    print(f"gene_acts: {batch_gene_acts}")
    print(f"pairs_txt:  {batch_pairs}")
    print(f"adata:      {adata_path}")
    print(f"converted:  {converted_shard if converted_shard else 'None'}")

    acts = load_pt_tensor(batch_gene_acts)
    pairs = read_pairs_txt(batch_pairs)
    adata = sc.read_h5ad(adata_path)

    print("\n=== BASIC SHAPES ===")
    print(f"gene_acts.shape = {tuple(acts.shape)}")
    print(f"num_pairs       = {len(pairs)}")
    if acts.ndim != 2:
        print("WARNING: gene_acts is not 2D")
    if acts.shape[0] != len(pairs):
        print("ERROR: row count mismatch between gene_acts and cell_gene_pairs")
    else:
        print("OK: row count matches")

    print("\n=== FIRST 10 PAIRS ===")
    for i, (cell, gene) in enumerate(pairs[:10]):
        print(f"  {i}: cell={cell!r}, gene={gene!r}")

    # duplicates / malformed
    pair_counter = Counter(pairs)
    dup_pairs = sum(1 for _, c in pair_counter.items() if c > 1)
    empty_cells = sum(1 for c, _ in pairs if not c)
    empty_genes = sum(1 for _, g in pairs if not g)

    print("\n=== PAIR QUALITY ===")
    print(f"unique_pairs      = {len(pair_counter)}")
    print(f"duplicate_pairs   = {dup_pairs}")
    print(f"empty_cell_ids    = {empty_cells}")
    print(f"empty_gene_names  = {empty_genes}")

    summarize_counter("duplicate pair counts (>1 only)", Counter({k: v for k, v in pair_counter.items() if v > 1}), topk=10)

    # gene stats
    genes_in_pairs = [g for _, g in pairs]
    gene_counter = Counter(genes_in_pairs)
    unique_genes = set(genes_in_pairs)
    unique_cells = set(c for c, _ in pairs)

    print("\n=== COVERAGE ===")
    print(f"unique genes in batch = {len(unique_genes)}")
    print(f"unique cells in batch = {len(unique_cells)}")
    summarize_counter("gene frequency", gene_counter, topk=15)

    # adata overlap
    adata_genes = set(normalize_var_names(adata))
    sym2ens = build_sym2ens_from_adata(adata)
    overlap_genes = unique_genes & adata_genes
    mapped_sym2ens = {g for g in unique_genes if g in sym2ens}
    ens_like = {g for g in unique_genes if g.startswith("ENSG")}

    print("\n=== GENE OVERLAP WITH ADATA ===")
    print(f"adata n_vars                 = {adata.n_vars}")
    print(f"genes in batch               = {len(unique_genes)}")
    print(f"overlap with adata var genes = {len(overlap_genes)}")
    print(f"mapped by sym2ens            = {len(mapped_sym2ens)}")
    print(f'ENSG-like labels             = {len(ens_like)}')

    missing = sorted(list(unique_genes - adata_genes))[:30]
    print(f"first 30 genes missing from adata panel: {missing}")

    # cell overlap
    adata_cells = set(adata.obs_names.astype(str).tolist())
    overlap_cells = unique_cells & adata_cells
    missing_cells = sorted(list(unique_cells - adata_cells))[:30]

    print("\n=== CELL OVERLAP WITH ADATA ===")
    print(f"cells in batch               = {len(unique_cells)}")
    print(f"overlap with adata.obs_names = {len(overlap_cells)}")
    print(f"first 30 cells missing from adata: {missing_cells}")

    # activation stats
    acts_f = acts.float()
    row_norms = torch.norm(acts_f, dim=1)
    print("\n=== ACTIVATION STATS ===")
    print(f"dim                      = {acts.shape[1]}")
    print(f"mean(abs)                = {acts_f.abs().mean().item():.6f}")
    print(f"std                      = {acts_f.std().item():.6f}")
    print(f"min                      = {acts_f.min().item():.6f}")
    print(f"max                      = {acts_f.max().item():.6f}")
    print(f"row norm mean            = {row_norms.mean().item():.6f}")
    print(f"row norm min/max         = {row_norms.min().item():.6f} / {row_norms.max().item():.6f}")
    print(f"all-zero rows            = {(row_norms == 0).sum().item()}")

    # gene-wise aggregate sanity
    per_gene_counts = Counter(g for _, g in pairs)
    high_rep = [(g, c) for g, c in per_gene_counts.items() if c > 20]
    print(f"genes appearing >20 times = {len(high_rep)}")

    # converted shard audit
    if converted_shard is not None:
        print("\n=== CONVERTED SHARD AUDIT ===")
        act_path = converted_shard / "activations.pt"
        idx_path = converted_shard / "index.pt"
        if not act_path.exists():
            print(f"ERROR: missing {act_path}")
            return
        if not idx_path.exists():
            print(f"ERROR: missing {idx_path}")
            return

        conv_acts = torch.load(act_path, map_location="cpu")
        conv_idx = torch.load(idx_path, map_location="cpu")

        print(f"converted activations.shape = {tuple(conv_acts.shape)}")
        print(f"index keys                  = {list(conv_idx.keys())}")

        token_ids = conv_idx.get("token_ids", None)
        example_ids = conv_idx.get("example_ids", None)

        if token_ids is None:
            print("ERROR: index missing token_ids")
        if example_ids is None:
            print("ERROR: index missing example_ids")

        if token_ids is not None:
            print(f"len(token_ids)   = {len(token_ids)}")
            print(f"first 10 token_ids   = {list(token_ids[:10])}")

        if example_ids is not None:
            print(f"len(example_ids) = {len(example_ids)}")
            print(f"first 10 example_ids = {list(example_ids[:10])}")

        if conv_acts.shape[0] != acts.shape[0]:
            print("ERROR: converted row count != original row count")
        else:
            print("OK: converted row count matches original")

        if token_ids is not None and len(token_ids) != len(pairs):
            print("ERROR: len(token_ids) != num_pairs")
        if example_ids is not None and len(example_ids) != len(pairs):
            print("ERROR: len(example_ids) != num_pairs")

        # exact row checks
        n = min(sample_n, len(pairs))
        rng = random.Random(0)
        indices = sorted(rng.sample(range(len(pairs)), n)) if len(pairs) >= n else list(range(len(pairs)))

        mismatch_meta = 0
        mismatch_tensor = 0
        for i in indices:
            cell, gene = pairs[i]

            tok = token_ids[i]
            ex = example_ids[i]

            tok = tok.item() if isinstance(tok, np.generic) else tok
            ex = ex.item() if isinstance(ex, np.generic) else ex
            tok = str(tok)
            ex = str(ex)

            if tok != gene or ex != cell:
                mismatch_meta += 1
                print(f"META MISMATCH @ row {i}: original=({cell}, {gene}) converted=({ex}, {tok})")

            same = torch.equal(acts[i], conv_acts[i])
            if not same:
                mismatch_tensor += 1
                max_abs = (acts[i].float() - conv_acts[i].float()).abs().max().item()
                print(f"TENSOR MISMATCH @ row {i}: max_abs_diff={max_abs:.8f}")

        if mismatch_meta == 0:
            print(f"OK: sampled metadata rows matched for {n} rows")
        else:
            print(f"ERROR: metadata mismatches in {mismatch_meta}/{n} sampled rows")

        if mismatch_tensor == 0:
            print(f"OK: sampled tensor rows matched for {n} rows")
        else:
            print(f"ERROR: tensor mismatches in {mismatch_tensor}/{n} sampled rows")

    # compact JSON summary
    summary = {
        "gene_acts_shape": list(acts.shape),
        "num_pairs": len(pairs),
        "row_count_match": acts.shape[0] == len(pairs),
        "unique_pairs": len(pair_counter),
        "duplicate_pairs": dup_pairs,
        "unique_genes": len(unique_genes),
        "unique_cells": len(unique_cells),
        "adata_n_vars": int(adata.n_vars),
        "gene_overlap_with_adata": len(overlap_genes),
        "gene_mapped_sym2ens": len(mapped_sym2ens),
        "cell_overlap_with_adata": len(overlap_cells),
        "all_zero_rows": int((row_norms == 0).sum().item()),
        "act_mean_abs": float(acts_f.abs().mean().item()),
        "act_std": float(acts_f.std().item()),
    }

    print("\n=== JSON SUMMARY ===")
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene-acts", type=Path, required=True, help="Path to batch_*_gene_acts.pt")
    ap.add_argument("--pairs", type=Path, required=True, help="Path to batch_*_cell_gene_pairs.txt")
    ap.add_argument("--adata", type=Path, required=True, help="Path to CosMx h5ad")
    ap.add_argument(
        "--converted-shard",
        type=Path,
        default=None,
        help="Optional path to converted generic shard dir containing activations.pt and index.pt",
    )
    ap.add_argument("--sample-n", type=int, default=20, help="How many random rows to compare in converted-shard audit")
    args = ap.parse_args()

    inspect_batch(
        batch_gene_acts=args.gene_acts,
        batch_pairs=args.pairs,
        adata_path=args.adata,
        converted_shard=args.converted_shard,
        sample_n=args.sample_n,
    )


if __name__ == "__main__":
    main()