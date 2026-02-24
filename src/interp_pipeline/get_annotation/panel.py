from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

@dataclass(frozen=True)
class GenePanel:
    """
    Canonical gene panel representation for a dataset.

    - ensembl_ids: stable IDs (e.g., ENSG...)
    - symbols: gene names/symbols (for g:Profiler queries and readability)
    - ens_to_sym: mapping for convenience
    """
    ensembl_ids: List[str]
    symbols: List[str]
    ens_to_sym: Dict[str, str]


def panel_from_cosmx_adata(adata, symbol_col: str = "index") -> GenePanel:
    """
    Your convention:
      - Ensembl IDs in adata.var.index
      - gene symbols in adata.var['index']

    This helper standardizes that.
    """
    ensembl_ids = adata.var.index.astype(str).tolist()

    if symbol_col not in adata.var.columns:
        raise ValueError(
            f"symbol_col='{symbol_col}' not found in adata.var. "
            f"Available columns: {list(adata.var.columns)[:50]}"
        )

    symbols = adata.var[symbol_col].astype(str).tolist()
    ens_to_sym = dict(zip(ensembl_ids, symbols))
    return GenePanel(ensembl_ids=ensembl_ids, symbols=symbols, ens_to_sym=ens_to_sym)