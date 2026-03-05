from __future__ import annotations

import os
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import torch
import scanpy as sc

from interp_pipeline.adapters.datasets.cosmx import CosMxDatasetAdapter
from interp_pipeline.adapters.models.scgpt import ScGPTAdapter
from interp_pipeline.adapters.model_base import ModelSpec

from interp_pipeline.extraction.extractor import extract_activations
from interp_pipeline.io.activation_store import ActivationStore, ActivationStoreSpec

from interp_pipeline.sae.sae_base import SAESpec
from interp_pipeline.sae.trainers import fit_sae_for_layer

from interp_pipeline.get_annotation.panel import panel_from_cosmx_adata
from interp_pipeline.get_annotation.gprofiler_client import GProfilerClient, GProfilerSpec

from interp_pipeline.get_annotation.f1_alignment import heldout_report_for_layer


# -----------------------
# Config
# -----------------------
ADATA_PATH = "/maiziezhou_lab2/yunfei/Projects/FM_temp/InterPLM/interplm/ge_shards/cosmx_human_lung_sec8.h5ad"
SCGPT_CKPT = "/maiziezhou_lab/zihang/SpatialFoundationModel/eval/scGPT/whole-human-pretrain"

OUT_ROOT = "runs/full_scgpt_cosmx"
DEVICE = "cuda"

# Activation extraction
EXTRACT_CFG: Dict[str, Any] = {
    "batch_size": 8,          # raise as GPU allows (e.g. 32/64)
    "max_length": 512,
    "model_name": "scgpt",
    "max_shards": None,       # full data
    # "target_tokens_per_shard": 12000,  # optional if your extractor supports it
}

# g:Profiler GT
GPROF_OUT = os.path.join(OUT_ROOT, "gprofiler")
GPROF_ALPHA = 0.05
GPROF_SOURCES = ["GO:BP", "GO:MF", "GO:CC", "REAC", "KEGG"]

# SAE training
SAE_SPEC = SAESpec(
    n_latents=4096,
    l1=1e-3,
    lr=1e-4,
    steps=20_000,
    seed=0,
)
SAE_TRAIN_BATCH = 1024

# Heldout evaluation (must match your “reported results” protocol)
LATENT_THRESHOLDS = [0.0, 0.15, 0.3, 0.6]
VALID_FRAC = 0.10
TEST_FRAC = 0.10
SPLIT_SEED = 0

TOPM_VALID_PER_CONCEPT_PER_THR = 200
EVAL_BATCH_SIZE = 8192
TP_DTYPE = np.int32


# -----------------------
# Helpers
# -----------------------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def pick_qval_col(df: pd.DataFrame) -> str:
    for c in ["p_value", "p_value_fdr", "p_value_adj", "p_value_corrected"]:
        if c in df.columns:
            return c
    if "p_value" in df.columns:
        return "p_value"
    raise ValueError(f"Cannot find p/q-value column in: {list(df.columns)[:40]}")


def as_list(x):
    import numpy as _np
    if x is None:
        return []
    if isinstance(x, float) and _np.isnan(x):
        return []
    if isinstance(x, list):
        out = []
        for v in x:
            if isinstance(v, str):
                out.append(v)
            elif isinstance(v, dict):
                if "name" in v:
                    out.append(str(v["name"]))
                elif "id" in v:
                    out.append(str(v["id"]))
        return out
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        if s.startswith("[") and s.endswith("]"):
            s2 = s.strip("[]").strip()
            if not s2:
                return []
            parts = [p.strip().strip("'").strip('"') for p in s2.split(",")]
            return [p for p in parts if p]
        if "," in s:
            return [p.strip() for p in s.split(",") if p.strip()]
        return [s]
    return [str(x)]


def build_gprofiler_gt(
    adata_path: str,
    out_dir: str,
    alpha: float,
    sources: List[str],
) -> str:
    """
    Builds and writes gene×term matrix to out_dir, returns path to CSV.
    """
    ensure_dir(out_dir)
    enr_path = os.path.join(out_dir, "gprofiler_enrichment.csv")
    bin_path = os.path.join(out_dir, "gprofiler_binary_gene_by_term.csv")
    meta_path = os.path.join(out_dir, "gprofiler_terms.tsv")

    if os.path.exists(bin_path):
        return bin_path  # reuse

    adata = sc.read_h5ad(adata_path)
    panel = panel_from_cosmx_adata(adata, symbol_col="index")
    genes_ens = panel.ensembl_ids
    genes_sym = panel.symbols
    sym_to_ens = {s: e for e, s in panel.ens_to_sym.items()}

    gp = GProfilerClient(cache_dir=os.path.join(out_dir, "gprof_cache"))
    spec = GProfilerSpec(
        organism="hsapiens",
        sources=sources,
        user_threshold=float(alpha),
        significance_threshold_method="fdr",
        return_dataframe=True,
    )

    res = gp.profile(genes_sym, spec=spec, query_name="cosmx_panel", force=False)
    if not isinstance(res, pd.DataFrame):
        res = pd.DataFrame(res)
    res.to_csv(enr_path, index=False)

    qcol = pick_qval_col(res)
    res_f = res[res[qcol] <= alpha].copy()

    term_id_col = "native" if "native" in res_f.columns else ("term_id" if "term_id" in res_f.columns else None)
    term_name_col = "name" if "name" in res_f.columns else ("term_name" if "term_name" in res_f.columns else None)
    source_col = "source" if "source" in res_f.columns else None
    intersection_col = "intersections" if "intersections" in res_f.columns else ("intersection" if "intersection" in res_f.columns else None)

    if term_id_col is None or intersection_col is None:
        raise ValueError(f"g:Profiler output missing required columns. Have: {list(res_f.columns)[:50]}")

    gene_index = {g: i for i, g in enumerate(genes_ens)}

    mat = np.zeros((len(genes_ens), len(res_f)), dtype=np.int8)
    terms = []
    meta_rows = []
    kept = 0

    for row in res_f.itertuples(index=False):
        rowd = row._asdict() if hasattr(row, "_asdict") else dict(row)

        term_id = str(rowd.get(term_id_col))
        term_name = str(rowd.get(term_name_col)) if term_name_col else ""
        src = str(rowd.get(source_col)) if source_col else ""

        overlap_syms = as_list(rowd.get(intersection_col))
        if not overlap_syms:
            continue

        hit = 0
        for s in overlap_syms:
            ens = sym_to_ens.get(str(s))
            if ens is None:
                continue
            i = gene_index.get(ens)
            if i is None:
                continue
            mat[i, kept] = 1
            hit += 1

        if hit == 0:
            continue

        terms.append(term_id)
        meta_rows.append(
            {
                "term_id": term_id,
                "term_name": term_name,
                "source": src,
                "q_value_col": qcol,
                "q_value": float(rowd.get(qcol)),
                "overlap_in_panel": hit,
                "overlap_list_len": len(overlap_syms),
            }
        )
        kept += 1

    mat = mat[:, :kept]
    pd.DataFrame(mat, index=genes_ens, columns=terms).to_csv(bin_path, index=True)
    pd.DataFrame(meta_rows).to_csv(meta_path, sep="\t", index=False)

    return bin_path


# -----------------------
# Main
# -----------------------
def main():
    ensure_dir(OUT_ROOT)

    # 0) GT once
    print("[0] Build/Load g:Profiler GT")
    gt_csv = build_gprofiler_gt(
        adata_path=ADATA_PATH,
        out_dir=GPROF_OUT,
        alpha=GPROF_ALPHA,
        sources=GPROF_SOURCES,
    )
    print("  GT:", gt_csv)

    # 1) Dataset + model
    print("[1] Load dataset + scGPT")
    ds = CosMxDatasetAdapter().load({"path": ADATA_PATH, "obs_key_map": {}})
    adapter = ScGPTAdapter()
    handle = adapter.load(ModelSpec(name="scgpt", checkpoint=SCGPT_CKPT, device=DEVICE, options={}))

    layers = adapter.list_layers(handle)
    if len(layers) < 12:
        raise RuntimeError(f"Expected >= 12 layers, got {len(layers)}")
    layers = layers[:12]
    print("  layers:", layers)

    # 2) Store
    store = ActivationStore(ActivationStoreSpec(root=OUT_ROOT))

    for layer in layers:
        print(f"\n=== LAYER: {layer} ===")

        # 2a) Extract activations (full data)
        print("[2a] Extract activations")
        extract_activations(
            dataset=ds,
            model_handle=handle,
            model_adapter=adapter,
            layers=[layer],
            store=store,
            extraction_cfg=EXTRACT_CFG,
        )

        # 2b) Train SAE
        print("[2b] Train SAE")
        sae_out_dir = os.path.join(OUT_ROOT, "sae", layer)
        ensure_dir(sae_out_dir)

        res = fit_sae_for_layer(
            store=store,
            layer=layer,
            spec=SAE_SPEC,
            output_dir=sae_out_dir,
            device=DEVICE,
            batch_size=SAE_TRAIN_BATCH,
        )

        sae_ckpt_path = res.model_path
        print("  SAE:", sae_ckpt_path)

        # 2c) Heldout reporting (valid->test), with valid top-M filter
        print("[2c] Heldout reporting (valid->test)")
        heldout_out = os.path.join(OUT_ROOT, "heldout_report", layer)
        ensure_dir(heldout_out)

        heldout_report_for_layer(
            layer=layer,
            store_root=OUT_ROOT,
            gt_csv=gt_csv,
            sae_ckpt_path=sae_ckpt_path,
            out_dir=heldout_out,
            latent_thresholds=[0.0, 0.15, 0.3, 0.6],
            adata_path=ADATA_PATH,
            scgpt_ckpt=SCGPT_CKPT,
            dev_mode=True,
            dev_max_shards=50,
            dev_max_rows_per_split_per_shard=4096,
            dev_only_valid=True,
        )

    print("\nDONE: scGPT 12-layer full pipeline complete.")


if __name__ == "__main__":
    main()