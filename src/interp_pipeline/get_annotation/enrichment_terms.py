from __future__ import annotations

from typing import Dict, Sequence

import pandas as pd

from interp_pipeline.get_annotation.gprofiler_client import GProfilerClient, GProfilerSpec


def gprofiler_go_term_qvals(
    gp: GProfilerClient,
    gene_symbols: Sequence[str],
    spec: GProfilerSpec,
    query_name: str = "panel",
) -> Dict[str, float]:
    """
    Query g:Profiler enrichment for the panel and return GO term q-values keyed by GO:XXXXXXX.

    This is used as a *term prior / mask*, not as gene×term membership.
    """
    res = gp.profile(gene_symbols, spec=spec, query_name=query_name)

    if not hasattr(res, "columns"):
        res = pd.DataFrame(res)

    cols = set(res.columns)

    # Term IDs are usually in 'native' for gprofiler-official output
    term_col = "native" if "native" in cols else ("term_id" if "term_id" in cols else None)
    if term_col is None:
        raise ValueError(f"Cannot find term id column. Columns: {list(res.columns)[:30]}")

    # Adjusted p-value column can vary; be defensive
    q_col = None
    for cand in ["p_value", "p_value_fdr", "p_value_adj", "p_value_corrected"]:
        if cand in cols:
            q_col = cand
            break
    if q_col is None:
        # last resort
        if "p_value" in cols:
            q_col = "p_value"
        else:
            raise ValueError(f"Cannot find p/q column. Columns: {list(res.columns)[:30]}")

    # GO terms only
    go_mask = res[term_col].astype(str).str.startswith("GO:")
    res_go = res.loc[go_mask, [term_col, q_col]].copy()
    res_go[term_col] = res_go[term_col].astype(str)

    # Keep best q if duplicates
    out: Dict[str, float] = {}
    for tid, grp in res_go.groupby(term_col):
        out[tid] = float(grp[q_col].min())
    return out