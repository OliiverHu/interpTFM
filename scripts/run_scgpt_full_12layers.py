# from __future__ import annotations

# import os
# from typing import List, Dict, Any

# import numpy as np
# import pandas as pd
# import torch
# import scanpy as sc

# from interp_pipeline.adapters.datasets.cosmx import CosMxDatasetAdapter
# from interp_pipeline.adapters.models.scgpt import ScGPTAdapter
# from interp_pipeline.adapters.model_base import ModelSpec

# from interp_pipeline.extraction.extractor import extract_activations
# from interp_pipeline.io.activation_store import ActivationStore, ActivationStoreSpec

# from interp_pipeline.sae.sae_base import SAESpec
# from interp_pipeline.sae.trainers import fit_sae_for_layer

# from interp_pipeline.get_annotation.panel import panel_from_cosmx_adata
# from interp_pipeline.get_annotation.gprofiler_client import GProfilerClient, GProfilerSpec
# from interp_pipeline.get_annotation.f1_alignment import heldout_report_for_layer


# # -----------------------
# # Config
# # -----------------------
# ADATA_PATH = "/maiziezhou_lab2/yunfei/Projects/FM_temp/InterPLM/interplm/ge_shards/cosmx_human_lung_sec8.h5ad"
# SCGPT_CKPT = "/maiziezhou_lab/zihang/SpatialFoundationModel/eval/scGPT/whole-human-pretrain"

# OUT_ROOT = "runs/full_scgpt_cosmx"
# DEVICE = "cuda"

# # Activation extraction
# EXTRACT_CFG: Dict[str, Any] = {
#     "batch_size": 8,
#     "max_length": 512,
#     "model_name": "scgpt",
#     "max_shards": None,  # full data
# }

# # g:Profiler GT
# GPROF_OUT = os.path.join(OUT_ROOT, "gprofiler")
# GPROF_ALPHA = 0.05
# GPROF_SOURCES = ["GO:BP", "GO:MF", "GO:CC", "REAC", "KEGG"]

# # SAE training
# SAE_SPEC = SAESpec(
#     n_latents=4096,
#     l1=1e-3,
#     lr=1e-4,
#     steps=20_000,
#     seed=0,
# )
# SAE_TRAIN_BATCH = 1024

# # Heldout evaluation thresholds (absolute Z > thr, like your debug script)
# LATENT_THRESHOLDS = [0.0, 0.15, 0.3, 0.6]

# # -----------------------
# # Dev-mode control
# # -----------------------
# DEV_ONLY_LAYER_INDEX = 4          # ONLY dev on the 5th layer in layers[:12]
# DEV_SHARD_FRACTION = 0.200        # e.g. 100% of shards
# DEV_MAX_ROWS_PER_SPLIT_PER_SHARD = 4096
# DEV_ONLY_VALID = True             # keep valid only in dev mode


# # -----------------------
# # Helpers
# # -----------------------
# def ensure_dir(p: str) -> None:
#     os.makedirs(p, exist_ok=True)


# def pick_qval_col(df: pd.DataFrame) -> str:
#     for c in ["p_value", "p_value_fdr", "p_value_adj", "p_value_corrected"]:
#         if c in df.columns:
#             return c
#     if "p_value" in df.columns:
#         return "p_value"
#     raise ValueError(f"Cannot find p/q-value column in: {list(df.columns)[:40]}")


# def as_list(x):
#     import numpy as _np
#     if x is None:
#         return []
#     if isinstance(x, float) and _np.isnan(x):
#         return []
#     if isinstance(x, list):
#         out = []
#         for v in x:
#             if isinstance(v, str):
#                 out.append(v)
#             elif isinstance(v, dict):
#                 if "name" in v:
#                     out.append(str(v["name"]))
#                 elif "id" in v:
#                     out.append(str(v["id"]))
#         return out
#     if isinstance(x, str):
#         s = x.strip()
#         if not s:
#             return []
#         if s.startswith("[") and s.endswith("]"):
#             s2 = s.strip("[]").strip()
#             if not s2:
#                 return []
#             parts = [p.strip().strip("'").strip('"') for p in s2.split(",")]
#             return [p for p in parts if p]
#         if "," in s:
#             return [p.strip() for p in s.split(",") if p.strip()]
#         return [s]
#     return [str(x)]


# def build_gprofiler_gt(
#     adata_path: str,
#     out_dir: str,
#     alpha: float,
#     sources: List[str],
# ) -> str:
#     """
#     Builds and writes gene×term matrix to out_dir, returns path to CSV.
#     """
#     ensure_dir(out_dir)
#     enr_path = os.path.join(out_dir, "gprofiler_enrichment.csv")
#     bin_path = os.path.join(out_dir, "gprofiler_binary_gene_by_term.csv")
#     meta_path = os.path.join(out_dir, "gprofiler_terms.tsv")

#     if os.path.exists(bin_path):
#         return bin_path  # reuse

#     adata = sc.read_h5ad(adata_path)
#     panel = panel_from_cosmx_adata(adata, symbol_col="index")
#     genes_ens = panel.ensembl_ids
#     genes_sym = panel.symbols
#     sym_to_ens = {s: e for e, s in panel.ens_to_sym.items()}

#     gp = GProfilerClient(cache_dir=os.path.join(out_dir, "gprof_cache"))
#     spec = GProfilerSpec(
#         organism="hsapiens",
#         sources=sources,
#         user_threshold=float(alpha),
#         significance_threshold_method="fdr",
#         return_dataframe=True,
#     )

#     res = gp.profile(genes_sym, spec=spec, query_name="cosmx_panel", force=False)
#     if not isinstance(res, pd.DataFrame):
#         res = pd.DataFrame(res)
#     res.to_csv(enr_path, index=False)

#     qcol = pick_qval_col(res)
#     res_f = res[res[qcol] <= alpha].copy()

#     term_id_col = "native" if "native" in res_f.columns else ("term_id" if "term_id" in res_f.columns else None)
#     term_name_col = "name" if "name" in res_f.columns else ("term_name" if "term_name" in res_f.columns else None)
#     source_col = "source" if "source" in res_f.columns else None
#     intersection_col = "intersections" if "intersections" in res_f.columns else ("intersection" if "intersection" in res_f.columns else None)

#     if term_id_col is None or intersection_col is None:
#         raise ValueError(f"g:Profiler output missing required columns. Have: {list(res_f.columns)[:50]}")

#     gene_index = {g: i for i, g in enumerate(genes_ens)}

#     mat = np.zeros((len(genes_ens), len(res_f)), dtype=np.int8)
#     terms = []
#     meta_rows = []
#     kept = 0

#     for row in res_f.itertuples(index=False):
#         rowd = row._asdict() if hasattr(row, "_asdict") else dict(row)

#         term_id = str(rowd.get(term_id_col))
#         term_name = str(rowd.get(term_name_col)) if term_name_col else ""
#         src = str(rowd.get(source_col)) if source_col else ""

#         overlap_syms = as_list(rowd.get(intersection_col))
#         if not overlap_syms:
#             continue

#         hit = 0
#         for s in overlap_syms:
#             ens = sym_to_ens.get(str(s))
#             if ens is None:
#                 continue
#             i = gene_index.get(ens)
#             if i is None:
#                 continue
#             mat[i, kept] = 1
#             hit += 1

#         if hit == 0:
#             continue

#         terms.append(term_id)
#         meta_rows.append(
#             {
#                 "term_id": term_id,
#                 "term_name": term_name,
#                 "source": src,
#                 "q_value_col": qcol,
#                 "q_value": float(rowd.get(qcol)),
#                 "overlap_in_panel": hit,
#                 "overlap_list_len": len(overlap_syms),
#             }
#         )
#         kept += 1

#     mat = mat[:, :kept]
#     pd.DataFrame(mat, index=genes_ens, columns=terms).to_csv(bin_path, index=True)
#     pd.DataFrame(meta_rows).to_csv(meta_path, sep="\t", index=False)

#     return bin_path


# def _compute_dev_max_shards(store_root: str, layer: str, frac: float) -> int:
#     """
#     Uses the extracted activation shards on disk to pick a shard count by proportion.
#     """
#     acts_root = os.path.join(store_root, "activations", layer)
#     shards = [p for p in os.listdir(acts_root) if p.startswith("shard_")]
#     n = len(shards)
#     if n <= 0:
#         return 0
#     k = int(np.ceil(n * float(frac)))
#     return max(1, min(n, k))


# # -----------------------
# # Main
# # -----------------------
# def main():
#     ensure_dir(OUT_ROOT)

#     # 0) GT once
#     print("[0] Build/Load g:Profiler GT")
#     gt_csv = build_gprofiler_gt(
#         adata_path=ADATA_PATH,
#         out_dir=GPROF_OUT,
#         alpha=GPROF_ALPHA,
#         sources=GPROF_SOURCES,
#     )
#     print("  GT:", gt_csv)

#     # 1) Dataset + model
#     print("[1] Load dataset + scGPT")
#     ds = CosMxDatasetAdapter().load({"path": ADATA_PATH, "obs_key_map": {}})
#     adapter = ScGPTAdapter()
#     handle = adapter.load(ModelSpec(name="scgpt", checkpoint=SCGPT_CKPT, device=DEVICE, options={}))

#     layers = adapter.list_layers(handle)
#     if len(layers) < 12:
#         raise RuntimeError(f"Expected >= 12 layers, got {len(layers)}")
#     layers = layers[:12]
#     print("  layers[:12] =", layers)
#     print(f"  DEV_ONLY_LAYER_INDEX={DEV_ONLY_LAYER_INDEX} -> dev layer will be: {layers[DEV_ONLY_LAYER_INDEX]}")

#     # 2) Store
#     store = ActivationStore(ActivationStoreSpec(root=OUT_ROOT))

#     for li, layer in enumerate(layers):
#         print(f"\n=== LAYER {li}/11: {layer} ===")

#         is_dev_layer = (li == DEV_ONLY_LAYER_INDEX)
#         if not is_dev_layer:
#             print("dev mode, skipping layer")
#             continue

#         # 2a) Extract activations (full data)
#         print("[2a] Extract activations")
#         extract_activations(
#             dataset=ds,
#             model_handle=handle,
#             model_adapter=adapter,
#             layers=[layer],
#             store=store,
#             extraction_cfg=EXTRACT_CFG,
#         )

#         # 2b) Train SAE
#         print("[2b] Train SAE")
#         sae_out_dir = os.path.join(OUT_ROOT, "sae", layer)
#         ensure_dir(sae_out_dir)

#         res = fit_sae_for_layer(
#             store=store,
#             layer=layer,
#             spec=SAE_SPEC,
#             output_dir=sae_out_dir,
#             device=DEVICE,
#             batch_size=SAE_TRAIN_BATCH,
#         )

#         sae_ckpt_path = res.model_path
#         print("  SAE:", sae_ckpt_path)

#         # 2c) Heldout reporting
#         heldout_out = os.path.join(OUT_ROOT, "heldout_report", layer)
#         ensure_dir(heldout_out)

        
#         dev_mode = bool(is_dev_layer)

#         dev_max_shards = None
#         if dev_mode:
#             dev_max_shards = _compute_dev_max_shards(OUT_ROOT, layer, DEV_SHARD_FRACTION)
#             print(f"[2c] Heldout reporting (DEV MODE) layer={layer}")
#             print(f"     dev shard fraction={DEV_SHARD_FRACTION} -> dev_max_shards={dev_max_shards}")
#         else:
#             print(f"[2c] Heldout reporting (FULL) layer={layer}")

#         heldout_report_for_layer(
#             layer=layer,
#             store_root=OUT_ROOT,
#             gt_csv=gt_csv,
#             sae_ckpt_path=sae_ckpt_path,
#             out_dir=heldout_out,
#             latent_thresholds=LATENT_THRESHOLDS,
#             adata_path=ADATA_PATH,
#             scgpt_ckpt=SCGPT_CKPT,
#             dev_mode=dev_mode,
#             dev_max_shards=dev_max_shards,
#             dev_max_rows_per_split_per_shard=DEV_MAX_ROWS_PER_SPLIT_PER_SHARD if dev_mode else None,
#             dev_only_valid=DEV_ONLY_VALID if dev_mode else False,
#         )

#     print("\nDONE: scGPT 12-layer pipeline complete.")


# if __name__ == "__main__":
#     main()

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Literal

import numpy as np
import pandas as pd
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


# ============================================================
# Global paths / constants
# ============================================================
ADATA_PATH = "/maiziezhou_lab2/yunfei/Projects/FM_temp/InterPLM/interplm/ge_shards/cosmx_human_lung_sec8.h5ad"
SCGPT_CKPT = "/maiziezhou_lab/zihang/SpatialFoundationModel/eval/scGPT/whole-human-pretrain"
OUT_ROOT = "runs/scgpt12layers_cosmx_human_lung_sec8"
DEVICE = "cuda"

# Top-level run toggle:
#   "dev"  -> fast debugging on a small subset
#   "full" -> real full pipeline
RUN_MODE: Literal["dev", "full"] = "full"

# Optional: number of transformer layers to consider from adapter.list_layers()
N_LAYERS_TO_USE = 12

# Optional: whether to reuse existing GT if already written
REUSE_GPROFILER_GT = True


# ============================================================
# Extraction / SAE / Annotation base config
# ============================================================
EXTRACT_CFG: Dict[str, Any] = {
    "batch_size": 8,
    "max_length": 512,
    "model_name": "scgpt",
    "max_shards": None,  # usually leave None unless extractor itself supports limiting
}

GPROF_ALPHA = 0.05
GPROF_SOURCES = ["GO:BP", "GO:MF", "GO:CC", "REAC", "KEGG"]
GPROF_OUT = os.path.join(OUT_ROOT, "gprofiler")

SAE_SPEC = SAESpec(
    n_latents=4096,
    l1=1e-3,
    lr=1e-4,
    steps=20_000,
    seed=0,
)
SAE_TRAIN_BATCH = 1024

LATENT_THRESHOLDS = [0.0, 0.15, 0.3, 0.6]


# ============================================================
# Mode configs
# ============================================================
@dataclass(frozen=True)
class DevConfig:
    enabled: bool
    only_layer_index: int = 4
    shard_fraction: float = 0.20
    max_rows_per_split_per_shard: int = 4096
    only_valid: bool = True


@dataclass(frozen=True)
class FullConfig:
    enabled: bool


@dataclass(frozen=True)
class PipelineConfig:
    run_mode: Literal["dev", "full"]
    dev: DevConfig
    full: FullConfig


CFG = PipelineConfig(
    run_mode=RUN_MODE,
    dev=DevConfig(
        enabled=(RUN_MODE == "dev"),
        only_layer_index=4,              # dev on the 5th layer among layers[:N_LAYERS_TO_USE]
        shard_fraction=0.20,             # use 20% of extracted shards in heldout eval
        max_rows_per_split_per_shard=4096,
        only_valid=True,
    ),
    full=FullConfig(
        enabled=(RUN_MODE == "full"),
    ),
)


# ============================================================
# Helpers
# ============================================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def pick_qval_col(df: pd.DataFrame) -> str:
    for c in ["p_value", "p_value_fdr", "p_value_adj", "p_value_corrected"]:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot find p/q-value column in dataframe columns: {list(df.columns)[:40]}")


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
    reuse_if_exists: bool = True,
) -> str:
    """
    Builds and writes gene×term matrix to out_dir, returns path to CSV.
    """
    ensure_dir(out_dir)

    enr_path = os.path.join(out_dir, "gprofiler_enrichment.csv")
    bin_path = os.path.join(out_dir, "gprofiler_binary_gene_by_term.csv")
    meta_path = os.path.join(out_dir, "gprofiler_terms.tsv")

    if reuse_if_exists and os.path.exists(bin_path):
        return bin_path

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
        source = str(rowd.get(source_col)) if source_col else ""

        overlap_syms = as_list(rowd.get(intersection_col))
        if not overlap_syms:
            continue

        hit = 0
        for sym in overlap_syms:
            ens = sym_to_ens.get(str(sym))
            if ens is None:
                continue
            idx = gene_index.get(ens)
            if idx is None:
                continue
            mat[idx, kept] = 1
            hit += 1

        if hit == 0:
            continue

        terms.append(term_id)
        meta_rows.append(
            {
                "term_id": term_id,
                "term_name": term_name,
                "source": source,
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


def compute_dev_max_shards(store_root: str, layer: str, frac: float) -> int:
    """
    Uses the extracted activation shards on disk to pick a shard count by proportion.
    """
    acts_root = os.path.join(store_root, "activations", layer)
    if not os.path.isdir(acts_root):
        raise FileNotFoundError(f"Activation shard directory not found: {acts_root}")

    shards = sorted(p for p in os.listdir(acts_root) if p.startswith("shard_"))
    n = len(shards)
    if n <= 0:
        raise RuntimeError(f"No activation shards found in: {acts_root}")

    k = int(np.ceil(n * float(frac)))
    return max(1, min(n, k))


def should_process_layer(layer_index: int, cfg: PipelineConfig) -> bool:
    if cfg.run_mode == "full":
        return True
    if cfg.run_mode == "dev":
        return layer_index == cfg.dev.only_layer_index
    raise ValueError(f"Unsupported run mode: {cfg.run_mode}")


def get_heldout_kwargs(cfg: PipelineConfig, store_root: str, layer: str) -> Dict[str, Any]:
    """
    Centralizes the dev/full differences for heldout reporting.
    """
    if cfg.run_mode == "full":
        return {
            "dev_mode": False,
            "dev_max_shards": None,
            "dev_max_rows_per_split_per_shard": None,
            "dev_only_valid": False,
        }

    if cfg.run_mode == "dev":
        dev_max_shards = compute_dev_max_shards(store_root, layer, cfg.dev.shard_fraction)
        return {
            "dev_mode": True,
            "dev_max_shards": dev_max_shards,
            "dev_max_rows_per_split_per_shard": cfg.dev.max_rows_per_split_per_shard,
            "dev_only_valid": cfg.dev.only_valid,
        }

    raise ValueError(f"Unsupported run mode: {cfg.run_mode}")


def print_run_summary(cfg: PipelineConfig) -> None:
    print("=" * 80)
    print(f"RUN_MODE = {cfg.run_mode}")
    print(f"OUT_ROOT  = {OUT_ROOT}")
    print(f"ADATA     = {ADATA_PATH}")
    print(f"CKPT      = {SCGPT_CKPT}")
    print(f"DEVICE    = {DEVICE}")
    print("-" * 80)

    if cfg.run_mode == "dev":
        print("Dev settings:")
        print(f"  only_layer_index              = {cfg.dev.only_layer_index}")
        print(f"  shard_fraction                = {cfg.dev.shard_fraction}")
        print(f"  max_rows_per_split_per_shard  = {cfg.dev.max_rows_per_split_per_shard}")
        print(f"  only_valid                    = {cfg.dev.only_valid}")
    else:
        print("Full settings:")
        print("  all selected layers will be processed")
        print("  heldout reporting uses full data")

    print("=" * 80)


# ============================================================
# Main
# ============================================================
def main() -> None:
    ensure_dir(OUT_ROOT)
    print_run_summary(CFG)

    # 0) Build or load GT
    print("[0] Build/Load g:Profiler GT")
    gt_csv = build_gprofiler_gt(
        adata_path=ADATA_PATH,
        out_dir=GPROF_OUT,
        alpha=GPROF_ALPHA,
        sources=GPROF_SOURCES,
        reuse_if_exists=REUSE_GPROFILER_GT,
    )
    print(f"  GT CSV: {gt_csv}")

    # 1) Load dataset + model
    print("[1] Load dataset + scGPT")
    ds = CosMxDatasetAdapter().load({"path": ADATA_PATH, "obs_key_map": {}})

    adapter = ScGPTAdapter()
    handle = adapter.load(
        ModelSpec(
            name="scgpt",
            checkpoint=SCGPT_CKPT,
            device=DEVICE,
            options={},
        )
    )

    layers = adapter.list_layers(handle)
    if len(layers) < N_LAYERS_TO_USE:
        raise RuntimeError(f"Expected >= {N_LAYERS_TO_USE} layers, got {len(layers)}")

    layers = layers[:N_LAYERS_TO_USE]
    print(f"  Using first {N_LAYERS_TO_USE} layers:")
    print(f"  {layers}")

    if CFG.run_mode == "dev":
        print(f"  Dev target layer: index={CFG.dev.only_layer_index}, name={layers[CFG.dev.only_layer_index]}")

    # 2) Activation store
    store = ActivationStore(ActivationStoreSpec(root=OUT_ROOT))

    # 3) Per-layer pipeline
    for li, layer in enumerate(layers):
        print(f"\n=== LAYER {li}/{len(layers) - 1}: {layer} ===")

        if not should_process_layer(li, CFG):
            print("  Skipped by current run mode.")
            continue

        # 3a) Extract activations
        print("[2a] Extract activations")
        extract_activations(
            dataset=ds,
            model_handle=handle,
            model_adapter=adapter,
            layers=[layer],
            store=store,
            extraction_cfg=EXTRACT_CFG,
        )

        # 3b) Train SAE
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
        print(f"  SAE checkpoint: {sae_ckpt_path}")

        # 3c) Heldout report
        heldout_out = os.path.join(OUT_ROOT, "heldout_report", layer)
        ensure_dir(heldout_out)

        heldout_kwargs = get_heldout_kwargs(CFG, OUT_ROOT, layer)
        print("[2c] Heldout reporting")
        print(f"  kwargs: {heldout_kwargs}")

        heldout_report_for_layer(
            layer=layer,
            store_root=OUT_ROOT,
            gt_csv=gt_csv,
            sae_ckpt_path=sae_ckpt_path,
            out_dir=heldout_out,
            latent_thresholds=LATENT_THRESHOLDS,
            adata_path=ADATA_PATH,
            scgpt_ckpt=SCGPT_CKPT,
            **heldout_kwargs,
        )

    print("\nDONE: scGPT pipeline complete.")


if __name__ == "__main__":
    main()