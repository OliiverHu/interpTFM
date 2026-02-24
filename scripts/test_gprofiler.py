import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import scanpy as sc

from interp_pipeline.get_annotation.panel import panel_from_cosmx_adata
from interp_pipeline.get_annotation.gprofiler_client import GProfilerClient, GProfilerSpec


ADATA_PATH = "/maiziezhou_lab2/yunfei/Projects/FM_temp/InterPLM/interplm/ge_shards/cosmx_human_lung_sec8.h5ad"
OUT_DIR = "debug_acts/gprofiler_only"
os.makedirs(OUT_DIR, exist_ok=True)

# Keep significant terms only? (recommended for manageable term count)
ALPHA = 0.05  # set to 1.0 if you want everything g:Profiler returns
MAX_TERMS = None  # e.g. 3000 if you want to cap


def _now() -> float:
    return time.time()


def _fmt_dt(t0: float) -> str:
    return f"{(time.time() - t0):.1f}s"


def _as_list(x: Any) -> List[str]:
    """
    Normalize g:Profiler 'intersection' field into List[str].
    It can be:
      - list[str]
      - list[dict] (rare)
      - string like 'A,B,C'
      - NaN / None
    """
    if x is None:
        return []
    if isinstance(x, float) and np.isnan(x):
        return []
    if isinstance(x, list):
        out = []
        for v in x:
            if isinstance(v, str):
                out.append(v)
            elif isinstance(v, dict):
                # sometimes objects with 'name'/'id'
                if "name" in v:
                    out.append(str(v["name"]))
                elif "id" in v:
                    out.append(str(v["id"]))
        return out
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        # common formats: "A,B,C" or "['A','B']"
        if s.startswith("[") and s.endswith("]"):
            # very defensive eval-like parsing without eval:
            s2 = s.strip("[]").strip()
            if not s2:
                return []
            parts = [p.strip().strip("'").strip('"') for p in s2.split(",")]
            return [p for p in parts if p]
        if "," in s:
            return [p.strip() for p in s.split(",") if p.strip()]
        return [s]
    return [str(x)]


def _pick_qval_col(df: pd.DataFrame) -> str:
    # gprofiler-official usually has 'p_value' and it is already corrected
    # depending on options. We'll prefer anything that looks adjusted.
    for c in ["p_value", "p_value_fdr", "p_value_adj", "p_value_corrected"]:
        if c in df.columns:
            return c
    # fallback
    if "p_value" in df.columns:
        return "p_value"
    raise ValueError(f"Cannot find a p/q-value column in columns: {list(df.columns)[:40]}")


print(f"[1/4] Load AnnData: {ADATA_PATH}")
t0 = _now()
adata = sc.read_h5ad(ADATA_PATH)
panel = panel_from_cosmx_adata(adata, symbol_col="index")

genes_ens = panel.ensembl_ids                 # rows must match this order
genes_sym = panel.symbols                     # used as g:Profiler query
sym_to_ens = {s: e for e, s in panel.ens_to_sym.items()}  # best-effort mapping

print(f"  genes: {len(genes_ens)} loaded in {_fmt_dt(t0)}")
print("  var.index (Ensembl) example:", genes_ens[:3])
print("  var['index'] (symbol) example:", genes_sym[:3])

print("[2/4] Query g:Profiler (GO + REAC + KEGG)")
t0 = _now()
gp = GProfilerClient(cache_dir=os.path.join(OUT_DIR, "gprof_cache"))

spec = GProfilerSpec(
    organism="hsapiens",
    sources=["GO:BP", "GO:MF", "GO:CC", "REAC", "KEGG"],
    user_threshold=float(ALPHA),
    significance_threshold_method="fdr",
    return_dataframe=True,
)

res = gp.profile(genes_sym, spec=spec, query_name="cosmx_panel", force=False)
if not isinstance(res, pd.DataFrame):
    res = pd.DataFrame(res)

enr_path = os.path.join(OUT_DIR, "gprofiler_enrichment.csv")
res.to_csv(enr_path, index=False)
print(f"  saved enrichment table: {enr_path} ({len(res)} rows) in {_fmt_dt(t0)}")

print("[3/4] Build gene×term binary matrix from g:Profiler intersections")
t0 = _now()
qcol = _pick_qval_col(res)

# Filter by q-value threshold (optional)
if ALPHA is not None:
    res_f = res[res[qcol] <= ALPHA].copy()
else:
    res_f = res.copy()

# Some tables contain non-significant rows depending on client behavior; this keeps things sane.
# Cap terms (optional)
if MAX_TERMS is not None and len(res_f) > MAX_TERMS:
    res_f = res_f.sort_values(qcol, ascending=True).head(int(MAX_TERMS)).copy()

# Identify columns we need
term_id_col = "native" if "native" in res_f.columns else ("term_id" if "term_id" in res_f.columns else None)
term_name_col = "name" if "name" in res_f.columns else ("term_name" if "term_name" in res_f.columns else None)
source_col = "source" if "source" in res_f.columns else None
# gprofiler-official uses "intersections" (plural) for overlap gene list
intersection_col = None
for cand in ["intersections", "intersection"]:
    if cand in res_f.columns:
        intersection_col = cand
        break

if term_id_col is None or intersection_col is None:
    raise ValueError(
        f"g:Profiler output missing required columns. Have: {list(res_f.columns)[:50]}. "
        "Need at least: term id (native/term_id) and intersections/intersection."
    )

terms: List[str] = []
term_meta_rows: List[Dict[str, Any]] = []
gene_index = {g: i for i, g in enumerate(genes_ens)}
mat = np.zeros((len(genes_ens), len(res_f)), dtype=np.int8)

kept = 0
for j, row in enumerate(res_f.itertuples(index=False)):
    rowd = row._asdict() if hasattr(row, "_asdict") else dict(row)

    term_id = str(rowd.get(term_id_col))
    term_name = str(rowd.get(term_name_col)) if term_name_col else ""
    src = str(rowd.get(source_col)) if source_col else ""

    overlap_syms = _as_list(rowd.get(intersection_col))
    if not overlap_syms:
        continue

    # Map overlap symbols -> Ensembl row indices (best effort)
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

    # Keep term only if it hit at least 1 gene in our 960 universe
    if hit == 0:
        continue

    terms.append(term_id)
    term_meta_rows.append(
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

# trim to kept terms count
mat = mat[:, :kept]

bin_df = pd.DataFrame(mat, index=genes_ens, columns=terms)

bin_path = os.path.join(OUT_DIR, "gprofiler_binary_gene_by_term.csv")
bin_df.to_csv(bin_path, index=True)

meta_path = os.path.join(OUT_DIR, "gprofiler_terms.tsv")
pd.DataFrame(term_meta_rows).to_csv(meta_path, sep="\t", index=False)

density = float(bin_df.values.mean()) if bin_df.size else 0.0
print(f"  built matrix: {bin_df.shape} density={density:.6f} in {_fmt_dt(t0)}")
print(f"  saved matrix: {bin_path}")
print(f"  saved terms meta: {meta_path}")

print("[4/4] Summary")
print("  total enrichment rows:", len(res))
print("  filtered rows used:", len(res_f))
print("  kept terms with >=1 overlap gene:", bin_df.shape[1])
print("  avg terms per gene:", float(bin_df.sum(axis=1).mean()))
print("DONE.")