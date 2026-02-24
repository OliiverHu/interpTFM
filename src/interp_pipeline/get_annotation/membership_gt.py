from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Common “high-confidence” experimental evidence codes
HIGH_CONF = {"EXP", "IDA", "IPI", "IMP", "IGI", "IEP"}


def _normalize_uniprot(x: object) -> str:
    """
    Normalize UniProt-like identifiers to a canonical accession:
      - strip whitespace
      - remove prefixes like 'UniProtKB:'
      - remove isoform suffix '-2'
      - remove dot suffix '.123' (defensive)
    """
    s = str(x).strip()
    if not s:
        return s

    if s.startswith("UniProtKB:"):
        s = s.split(":", 1)[1]

    # isoforms often appear as Q9H799-2
    if "-" in s:
        s = s.split("-", 1)[0]

    # some tools return O00182.208; QuickGO won't, but keep defensive
    if "." in s:
        s = s.split(".", 1)[0]

    return s


@dataclass(frozen=True)
class MembershipGTSpec:
    """
    Controls how membership ground truth is built from QuickGO annotations.
    """
    high_conf_only: bool = True
    keep_aspects: Optional[Sequence[str]] = None  # ["P","F","C"] optional


def build_go_membership_gt(
    genes_ens: Sequence[str],                       # length = 960
    ensg_to_uniprot: Dict[str, List[str]],
    quickgo_annotations: Sequence[dict],
    spec: MembershipGTSpec,
) -> pd.DataFrame:
    """
    Build a binary membership matrix of shape [n_genes, n_terms].

    Rows: Ensembl gene IDs (adata.var.index)
    Cols: GO IDs (GO:xxxxxxx)

    A cell is 1 if:
      - QuickGO says the UniProt product is annotated to GO term
      - it maps back to one of our Ensembl genes
      - and it passes evidence/aspect filters
    """
    genes_ens = list(genes_ens)
    gene_index = {g: i for i, g in enumerate(genes_ens)}

    # Reverse map UniProt -> Ensembl (normalize keys!)
    uniprot_to_ensg: Dict[str, str] = {}
    for ensg, accs in ensg_to_uniprot.items():
        for a in accs:
            a_norm = _normalize_uniprot(a)
            if a_norm:
                uniprot_to_ensg.setdefault(a_norm, ensg)

    edges: List[Tuple[int, str]] = []  # (gene_row_index, go_id)
    keep_aspects = set(spec.keep_aspects) if spec.keep_aspects else None

    # Debug counters (helps if it fails again)
    n_total = 0
    n_missing_go = 0
    n_missing_gp = 0
    n_gp_unmapped = 0
    n_filtered_ev = 0
    n_filtered_aspect = 0

    for ann in quickgo_annotations:
        n_total += 1

        go_id = ann.get("goId") or ann.get("goID") or ann.get("go_id")
        if not go_id:
            n_missing_go += 1
            continue

        aspect = ann.get("aspect")
        if keep_aspects is not None and aspect and aspect not in keep_aspects:
            n_filtered_aspect += 1
            continue

        ev = ann.get("evidenceCode")
        if spec.high_conf_only and ev and ev not in HIGH_CONF:
            n_filtered_ev += 1
            continue

        gp = ann.get("geneProductId")
        if not gp:
            n_missing_gp += 1
            continue

        gp_norm = _normalize_uniprot(gp)
        ensg = uniprot_to_ensg.get(gp_norm)
        if ensg is None:
            n_gp_unmapped += 1
            continue

        i = gene_index.get(ensg)
        if i is None:
            continue

        edges.append((i, str(go_id)))

    if not edges:
        raise ValueError(
            "No GO edges found after filtering.\n"
            f"Debug: total={n_total} missing_go={n_missing_go} missing_gp={n_missing_gp} "
            f"unmapped_gp={n_gp_unmapped} filtered_ev={n_filtered_ev} filtered_aspect={n_filtered_aspect}\n"
            "Most likely: QuickGO geneProductId has prefixes/isoforms and didn't match your UniProt mapping.\n"
            "This function now normalizes UniProt IDs; if still failing, print a few geneProductId values."
        )

    terms = sorted({t for _, t in edges})
    term_index = {t: j for j, t in enumerate(terms)}

    mat = np.zeros((len(genes_ens), len(terms)), dtype=np.int8)
    for i, t in edges:
        mat[i, term_index[t]] = 1

    return pd.DataFrame(mat, index=genes_ens, columns=terms)