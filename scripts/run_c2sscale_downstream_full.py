from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import torch

from interp_pipeline.get_annotation.enrichment_terms import gprofiler_go_term_qvals
from interp_pipeline.get_annotation.gconvert_client import GConvertClient, GConvertSpec
from interp_pipeline.get_annotation.gprofiler_client import GProfilerClient, GProfilerSpec
from interp_pipeline.get_annotation.membership_gt import MembershipGTSpec, build_go_membership_gt
from interp_pipeline.get_annotation.panel import panel_from_cosmx_adata
from interp_pipeline.get_annotation.quickgo_client import QuickGOClient, QuickGOSpec

ADATA_PATH = "/maiziezhou_lab2/yunfei/Projects/FM_temp/InterPLM/interplm/ge_shards/cosmx_human_lung_sec8.h5ad"
RUNS_ROOT = "runs/full_c2sscale_cosmx"
EXTRACTION_ROOT = "/maiziezhou_lab2/yunfei/Projects/interpTFM/c2s_full_extraction"
DEFAULT_LAYER = "layer_17"
DEFAULT_SAE_THR = 0.6
DEFAULT_CELLTYPE_COL = "author_cell_type"
DEFAULT_GO_OBO = "resources/go-basic.obo"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _now() -> float:
    return time.time()


def _fmt_dt(t0: float) -> str:
    return f"{(time.time() - t0):.1f}s"


def load_module_from_path(module_name: str, path: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from path: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def script_path(scripts_root: str, name: str) -> str:
    p = os.path.join(scripts_root, name)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing script: {p}")
    return p


def discover_interpretable_h5ad(runs_root: str, layer: str, sae_thr: float) -> Optional[str]:
    base = os.path.join(runs_root, "interpretable_adata", layer, f"sae_thr_{sae_thr}")
    if not os.path.isdir(base):
        return None
    cands = sorted(Path(base).glob("*.h5ad"))
    return str(cands[0]) if cands else None


# -----------------------------------------------------------------------------
# Stage 1: GO membership enrichment
# -----------------------------------------------------------------------------
def run_go_membership_enrichment(adata_path: str, out_dir: str) -> Dict[str, str]:
    ensure_dir(out_dir)

    print(f"[go-membership 1/5] loading AnnData: {adata_path}")
    t0 = _now()
    adata = sc.read_h5ad(adata_path)
    panel = panel_from_cosmx_adata(adata, symbol_col="index")
    genes_ens = panel.ensembl_ids
    genes_sym = panel.symbols
    print(f" n_genes={len(genes_ens)} loaded in {_fmt_dt(t0)}")

    print("[go-membership 2/5] Ensembl -> UniProt via g:Profiler g:Convert")
    t0 = _now()
    gp = GProfilerClient(cache_dir=os.path.join(out_dir, "gprof_cache"))
    gconv = GConvertClient(gp=gp, cache_dir=os.path.join(out_dir, "gconvert_cache"))
    gconv_spec = GConvertSpec(organism="hsapiens", target="UNIPROTSWISSPROT")
    ens_to_up = gconv.ensg_to_uniprot(genes_ens, spec=gconv_spec, force=False)
    uniprots = sorted({u for accs in ens_to_up.values() for u in accs})
    coverage = sum(1 for g in genes_ens if len(ens_to_up.get(g, [])) > 0) / max(1, len(genes_ens))
    print(f" mapped UniProt accessions={len(uniprots)} coverage={coverage:.3f} in {_fmt_dt(t0)}")

    print("[go-membership 3/5] fetching QuickGO annotations")
    t0 = _now()
    qg = QuickGOClient(cache_dir=os.path.join(out_dir, "quickgo_cache"))
    qg_spec = QuickGOSpec(taxon_id="9606")
    anns = qg.fetch_annotations_for_uniprot(uniprots, qg_spec, force=False)
    print(f" fetched annotations={len(anns)} in {_fmt_dt(t0)}")

    print(" building GO membership GT (gene x GO term)")
    gt_member = build_go_membership_gt(
        genes_ens=genes_ens,
        ensg_to_uniprot=ens_to_up,
        quickgo_annotations=anns,
        spec=MembershipGTSpec(high_conf_only=False, keep_aspects=None),
    )
    member_path = os.path.join(out_dir, "GT_member_GO.csv")
    gt_member.to_csv(member_path)
    print(f" GT_member saved: {member_path}")
    print(f" shape={gt_member.shape} density={float(gt_member.values.mean()):.6f}")

    print("[go-membership 4/5] running g:Profiler enrichment on CosMx panel")
    t0 = _now()
    gp_spec = GProfilerSpec(
        organism="hsapiens",
        sources=["GO:BP", "GO:MF", "GO:CC", "REAC", "KEGG"],
        user_threshold=0.05,
        significance_threshold_method="fdr",
        return_dataframe=True,
    )
    enr = gp.profile(genes_sym, spec=gp_spec, query_name="cosmx_panel")
    enr_path = os.path.join(out_dir, "enrichment_cosmx_panel_GO_REAC.csv")
    if hasattr(enr, "to_csv"):
        enr.to_csv(enr_path, index=False)
    else:
        pd.DataFrame(enr).to_csv(enr_path, index=False)
    print(f" saved enrichment table: {enr_path} in {_fmt_dt(t0)}")

    term_q = gprofiler_go_term_qvals(gp, genes_sym, gp_spec, query_name="cosmx_panel")
    alpha = 0.05
    keep_terms = [t for t in gt_member.columns if (t in term_q and term_q[t] < alpha)]
    gt_masked = gt_member[keep_terms].copy()
    masked_path = os.path.join(out_dir, "GT_member_GO_masked_by_enrichment.csv")
    gt_masked.to_csv(masked_path)

    print(f" masked GT saved: {masked_path}")
    print(f" masked shape={gt_masked.shape}")
    print("[go-membership 5/5] summary")
    print(" GO terms in membership:", gt_member.shape[1])
    print(" GO terms enriched+present:", gt_masked.shape[1])
    print(" fraction kept:", gt_masked.shape[1] / max(1, gt_member.shape[1]))

    return {
        "member_path": member_path,
        "enrichment_path": enr_path,
        "masked_path": masked_path,
    }


# -----------------------------------------------------------------------------
# Stage 2: c2s cell-activation TIS (adapted from test_tis.py helpers)
# -----------------------------------------------------------------------------
def load_c2s_cell_activations_aligned(cell_acts_root: str, obs_names: Sequence[str], max_shards: Optional[int] = None):
    shard_dirs = sorted([p for p in Path(cell_acts_root).glob("shard_*") if p.is_dir()])
    if max_shards is not None:
        shard_dirs = shard_dirs[: int(max_shards)]
    if not shard_dirs:
        raise FileNotFoundError(f"No shard dirs found under {cell_acts_root}")

    obs_names = [str(x) for x in obs_names]
    pos = {x: i for i, x in enumerate(obs_names)}
    hidden_dim: Optional[int] = None
    aligned: Optional[np.ndarray] = None
    has_row = np.zeros((len(obs_names),), dtype=bool)

    for shard_dir in shard_dirs:
        act_files = sorted(shard_dir.glob("batch_*_cell_acts.pt"))
        for act_path in act_files:
            stem = act_path.name.replace("_cell_acts.pt", "")
            id_path = shard_dir / f"{stem}_cell_ids.txt"
            if not id_path.exists():
                continue
            acts = torch.load(act_path, map_location="cpu")
            if not isinstance(acts, torch.Tensor) or acts.ndim != 2:
                continue
            cell_ids = [ln.strip() for ln in id_path.read_text().splitlines() if ln.strip()]
            if len(cell_ids) != acts.shape[0]:
                raise RuntimeError(f"Row mismatch: {act_path} rows={acts.shape[0]} ids={len(cell_ids)}")
            acts_np = acts.float().cpu().numpy()
            if hidden_dim is None:
                hidden_dim = int(acts_np.shape[1])
                aligned = np.zeros((len(obs_names), hidden_dim), dtype=np.float32)
            for row, cid in zip(acts_np, cell_ids):
                j = pos.get(str(cid))
                if j is None:
                    continue
                aligned[j] = row.astype(np.float32, copy=False)
                has_row[j] = True

    if hidden_dim is None or aligned is None:
        raise RuntimeError(f"No usable cell activations found under {cell_acts_root}")
    return aligned, has_row


def run_tis_c2s_cell(
    scripts_root: str,
    adata_path: str,
    extraction_root: str,
    runs_root: str,
    layer: str,
    max_shards: int = 50,
) -> str:
    mod = load_module_from_path("_test_tis_helpers", script_path(scripts_root, "test_tis.py"))

    print("[tis] loading AnnData")
    adata = sc.read_h5ad(adata_path)
    cell_root = os.path.join(extraction_root, "cell_activations", layer)
    print(f"[tis] loading c2s cell activations from {cell_root}")
    A_aligned, has_row = load_c2s_cell_activations_aligned(cell_root, adata.obs_names.tolist(), max_shards=max_shards)
    A_use = A_aligned[has_row]
    J = mod.build_judge_matrix(adata, mode="log1p_cp10k")
    J_use = J[has_row]

    outdir = os.path.join(runs_root, "tis", layer)
    ensure_dir(outdir)
    cfg = getattr(mod, "CFG")

    print("[tis] computing TIS in c2s cell-activation space")
    top_idx, bot_idx, med = mod.build_pools_quantile(A_use, cfg.q_low, cfg.q_high)
    tis_cell = mod.compute_tis_mis(A_use, J_use, top_idx, bot_idx, med, cfg)
    np.save(os.path.join(outdir, "tis_c2sscale_cell.npy"), tis_cell.astype(np.float32))
    mod.save_json(os.path.join(outdir, "tis_c2sscale_cell_summary.json"), mod.summarize(tis_cell))

    print("[tis] PCA baseline")
    Ap = mod.pca_activations(A_use, seed=cfg.seed)
    top_idx_p, bot_idx_p, med_p = mod.build_pools_quantile(Ap, cfg.q_low, cfg.q_high)
    tis_pca = mod.compute_tis_mis(Ap, J_use, top_idx_p, bot_idx_p, med_p, cfg)
    np.save(os.path.join(outdir, "tis_c2sscale_cell_pca.npy"), tis_pca.astype(np.float32))
    mod.save_json(os.path.join(outdir, "tis_c2sscale_cell_pca_summary.json"), mod.summarize(tis_pca))

    print(f"[tis] wrote: {outdir}")
    return outdir


# -----------------------------------------------------------------------------
# Stage wrappers for existing scripts
# -----------------------------------------------------------------------------
def run_go_reduce(scripts_root: str, runs_root: str, layer: str, go_obo_path: str) -> str:
    mod = load_module_from_path("_test_go_reduce", script_path(scripts_root, "test_go_reduce.py"))
    mod.RUNS_ROOT = runs_root
    mod.LAYER = layer
    mod.F1_TABLE = os.path.join(runs_root, "heldout_report", layer, "valid_concept_f1_scores.csv")
    mod.GO_OBO_PATH = go_obo_path
    mod.OUTDIR = os.path.join(runs_root, "f1_analysis", "go_parent_nmi", layer)
    ensure_dir(mod.OUTDIR)
    mod.main()
    return mod.OUTDIR


def run_niche_sweep(scripts_root: str, interp_h5ad: str, runs_root: str, layer: str) -> str:
    mod = load_module_from_path("_test_niche_sweep", script_path(scripts_root, "test_niche_sweep.py"))
    mod.INTERP_H5AD = interp_h5ad
    mod.OUTDIR = os.path.join(runs_root, "niche_discovery", layer)
    ensure_dir(mod.OUTDIR)
    mod.main()
    return mod.OUTDIR


def run_niche_validation(scripts_root: str, interp_h5ad: str, runs_root: str, layer: str, celltype_col: str) -> str:
    mod = load_module_from_path("_test_niche_validation", script_path(scripts_root, "test_niche_validation.py"))
    mod.INTERP_H5AD = interp_h5ad
    mod.OUTDIR = os.path.join(runs_root, "niche_discovery", layer, "validation_k3_r120_m")
    mod.CELLTYPE_COL = celltype_col
    ensure_dir(mod.OUTDIR)
    mod.main()
    return mod.OUTDIR


def run_ccc(scripts_root: str, interp_h5ad: str, niche_labels_csv: str, runs_root: str, layer: str, celltype_key: str) -> str:
    mod = load_module_from_path("_test_ccc", script_path(scripts_root, "test_ccc.py"))
    mod.INTERP_H5AD = interp_h5ad
    mod.NICHE_LABELS_CSV = niche_labels_csv
    mod.CELLTYPE_KEY = celltype_key
    mod.OUTDIR = os.path.join(runs_root, "ccc_interdomain", layer)
    ensure_dir(mod.OUTDIR)
    mod.main()
    return mod.OUTDIR


# -----------------------------------------------------------------------------
# CLI / main
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Integrated downstream workflow for c2s-scale.")
    p.add_argument("--scripts-root", default="scripts")
    p.add_argument("--adata-path", default=ADATA_PATH)
    p.add_argument("--runs-root", default=RUNS_ROOT)
    p.add_argument("--extraction-root", default=EXTRACTION_ROOT)
    p.add_argument("--layer", default=DEFAULT_LAYER)
    p.add_argument("--go-obo-path", default=DEFAULT_GO_OBO)
    p.add_argument("--celltype-col", default=DEFAULT_CELLTYPE_COL)
    p.add_argument("--sae-thr", type=float, default=DEFAULT_SAE_THR)
    p.add_argument("--interp-h5ad", default=None)
    p.add_argument("--run-go-membership", action="store_true")
    p.add_argument("--run-go-reduce", action="store_true")
    p.add_argument("--run-tis", action="store_true")
    p.add_argument("--run-niche-sweep", action="store_true")
    p.add_argument("--run-niche-validation", action="store_true")
    p.add_argument("--run-ccc", action="store_true")
    p.add_argument("--all", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.all:
        args.run_go_membership = True
        args.run_go_reduce = True
        args.run_tis = True
        args.run_niche_sweep = True
        args.run_niche_validation = True
        args.run_ccc = True

    interp_h5ad = args.interp_h5ad or discover_interpretable_h5ad(args.runs_root, args.layer, args.sae_thr)
    if interp_h5ad is not None:
        print(f"[interp] using interpretable h5ad: {interp_h5ad}")
    else:
        print("[interp] no interpretable h5ad detected automatically")

    if args.run_go_membership:
        out_dir = os.path.join(args.runs_root, "go_membership")
        run_go_membership_enrichment(args.adata_path, out_dir)

    if args.run_go_reduce:
        run_go_reduce(args.scripts_root, args.runs_root, args.layer, args.go_obo_path)

    if args.run_tis:
        run_tis_c2s_cell(
            scripts_root=args.scripts_root,
            adata_path=args.adata_path,
            extraction_root=args.extraction_root,
            runs_root=args.runs_root,
            layer=args.layer,
        )

    if args.run_niche_sweep:
        if interp_h5ad is None:
            raise FileNotFoundError("Need --interp-h5ad (or an auto-discoverable interpretable h5ad) for niche sweep.")
        run_niche_sweep(args.scripts_root, interp_h5ad, args.runs_root, args.layer)

    validation_out = None
    if args.run_niche_validation:
        if interp_h5ad is None:
            raise FileNotFoundError("Need --interp-h5ad (or an auto-discoverable interpretable h5ad) for niche validation.")
        validation_out = run_niche_validation(
            args.scripts_root,
            interp_h5ad,
            args.runs_root,
            args.layer,
            args.celltype_col,
        )

    if args.run_ccc:
        if interp_h5ad is None:
            raise FileNotFoundError("Need --interp-h5ad (or an auto-discoverable interpretable h5ad) for CCC.")
        niche_labels_csv = os.path.join(
            args.runs_root,
            "niche_discovery",
            args.layer,
            "validation_k3_r120_m",
            "global_labels.csv",
        )
        if validation_out is not None:
            niche_labels_csv = os.path.join(validation_out, "global_labels.csv")
        if not os.path.exists(niche_labels_csv):
            raise FileNotFoundError(f"Missing niche label csv for CCC: {niche_labels_csv}")
        run_ccc(
            args.scripts_root,
            interp_h5ad,
            niche_labels_csv,
            args.runs_root,
            args.layer,
            args.celltype_col,
        )

    print("\nDONE: c2s-scale downstream workflow complete.")


if __name__ == "__main__":
    main()
