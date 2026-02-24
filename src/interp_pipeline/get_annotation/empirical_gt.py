from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Set, Tuple, Union

import pandas as pd


DEFAULT_HIGH_CONF_EVIDENCE: Set[str] = {"EXP", "IDA", "IPI", "IMP", "IGI", "IEP"}


@dataclass(frozen=True)
class EmpiricalGTSpec:
    """
    How to interpret your gprofiler annotation matrix CSV.
    """
    term_col: str = "term_name"
    gene_col_start: int = 10
    high_conf_evidence: Set[str] = field(
        default_factory=lambda: set(DEFAULT_HIGH_CONF_EVIDENCE)
    )
    sort_terms: bool = True


def _has_high_conf(cell: object, high_conf_evidence: Set[str]) -> int:
    if pd.isna(cell):
        return 0
    codes = [c.strip() for c in str(cell).split(",")]
    return int(any(c in high_conf_evidence for c in codes))


def build_binary_empirical_gt(
    gprofiler_annotation_csv: str,
    gene_panel: Sequence[str],
    spec: Optional[EmpiricalGTSpec] = None,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Returns:
      binary_df: DataFrame [n_terms, n_genes] with 0/1 entries
      terms: list of term names (rows)
      genes: list of genes (columns) that were retained from panel
    """
    spec = spec or EmpiricalGTSpec()
    df = pd.read_csv(gprofiler_annotation_csv)

    if spec.term_col not in df.columns:
        raise ValueError(f"term column '{spec.term_col}' not found. Columns: {list(df.columns)[:30]}...")

    if spec.sort_terms:
        df = df.sort_values(by=spec.term_col, ascending=True)

    # Gene columns are everything from gene_col_start onward (your notebook assumption)
    gene_cols = list(df.columns[spec.gene_col_start:])

    # Intersect with cosmx panel
    panel_set = set(gene_panel)
    keep_gene_cols = [g for g in gene_cols if g in panel_set]

    if len(keep_gene_cols) == 0:
        raise ValueError(
            "No overlap between gene_panel and gene columns in gprofiler annotation table. "
            "Check gene naming (symbols vs Ensembl) and gene_col_start."
        )

    # Evidence-code cells -> binary (high-confidence only)
    binary_df = df[keep_gene_cols].applymap(lambda x: _has_high_conf(x, spec.high_conf_evidence)).astype("int8")

    terms = df[spec.term_col].astype(str).tolist()
    genes = keep_gene_cols

    return binary_df, terms, genes