#!/usr/bin/env python3
from __future__ import annotations

"""
Post-hoc analysis for 3-model niche validation.

Assumes you already ran test_niche_validation_3models.py / relabel version
with shared settings and tumor13-based niche relabeling.

It does two things:

1. Enriched interpretable terms per niche, per model
   - For each model and each niche, compare cells in the niche vs all other cells.
   - Compute mean_in, mean_out, effect_size, fraction active, delta fraction active,
     Wilcoxon rank-sum p-value, BH q-value.
   - Rank enriched terms by positive effect size / activity difference.

2. Broad biological category summary of top concepts
   - Assign each enriched concept to broad categories using transparent keyword rules.
   - Compare category distributions across models by niche.

3. Cross-model niche consistency
   - Merge global_labels.csv across models by cell.
   - Compute pairwise ARI, NMI, label accuracy, confusion matrices, and per-niche Jaccard.
   - Compute cell-level consensus agreement across the 3 models.

Typical command:
python test_analyze_niche_terms_and_consistency_3models.py \
  --labels scgpt c2sscale geneformer \
  --validation-dirs \
    /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/niche_validation_3models_shared/scgpt/layer_4.norm2/r120p0_xm_gmm_k3 \
    /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/niche_validation_3models_shared/c2sscale/layer_17/r120p0_xm_gmm_k3 \
    /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/niche_validation_3models_shared/geneformer/layer_4/r120p0_xm_gmm_k3 \
  --out-dir /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/niche_validation_3models_shared/posthoc_terms_consistency \
  --top-k 25 \
  --active-threshold 0.0
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import ranksums
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def dense_X(adata) -> np.ndarray:
    X = adata.X
    if not isinstance(X, np.ndarray):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def save_json(path: str | Path, obj: Dict[str, Any]) -> None:
    def conv(x):
        if hasattr(x, "item"):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, (set, tuple)):
            return list(x)
        return x
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=conv)


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    qvals = np.full_like(pvals, np.nan, dtype=float)
    ok = np.isfinite(pvals)
    if ok.sum() == 0:
        return qvals
    p = pvals[ok]
    order = np.argsort(p)
    ranked = p[order]
    n = len(ranked)
    q_ranked = ranked * n / (np.arange(n) + 1)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0.0, 1.0)
    q = np.empty_like(p)
    q[order] = q_ranked
    qvals[ok] = q
    return qvals


def infer_labels_from_adata_or_csv(adata, validation_dir: Path, label_col: str) -> np.ndarray:
    if label_col in adata.obs.columns:
        return adata.obs[label_col].astype(int).to_numpy()

    labels_csv = validation_dir / "global_labels.csv"
    if labels_csv.exists():
        lab = pd.read_csv(labels_csv)
        if label_col not in lab.columns:
            raise RuntimeError(
                f"{labels_csv} does not contain label_col={label_col!r}. "
                f"Available columns: {list(lab.columns)}"
            )
        if "cell" not in lab.columns:
            raise RuntimeError(f"{labels_csv} must contain a 'cell' column.")
        lab = lab.set_index("cell").loc[adata.obs_names.astype(str)]
        return lab[label_col].astype(int).to_numpy()

    raise RuntimeError(f"Could not find label_col={label_col!r} in adata.obs or {labels_csv}")


def get_var_strings(adata, col: Optional[str], fallback_prefix: str) -> List[str]:
    if col is not None and col.strip().lower() not in {"", "none", "null", "na"}:
        if col not in adata.var.columns:
            raise RuntimeError(
                f"Requested adata.var column {col!r} is missing. "
                f"Available columns: {list(adata.var.columns)}"
            )
        vals = adata.var[col].astype(str).tolist()
    else:
        vals = adata.var_names.astype(str).tolist()

    out = []
    for i, v in enumerate(vals):
        v = str(v)
        if v.strip() == "" or v.lower() == "nan":
            v = f"{fallback_prefix}_{i}"
        out.append(v)
    return out


def load_model_payload(
    label: str,
    validation_dir: str,
    label_col: str,
    term_col: Optional[str],
    feature_id_col: Optional[str],
) -> Dict[str, Any]:
    vdir = Path(validation_dir)
    h5ad = vdir / "adata_with_niche_labels.h5ad"
    if not h5ad.exists():
        raise RuntimeError(f"Missing {h5ad}")

    adata = sc.read_h5ad(h5ad)
    X = dense_X(adata)
    y = infer_labels_from_adata_or_csv(adata, vdir, label_col)
    terms = get_var_strings(adata, term_col, "term")
    features = get_var_strings(adata, feature_id_col, "feature")

    return {
        "label": label,
        "validation_dir": str(vdir),
        "adata": adata,
        "X": X,
        "y": y.astype(int),
        "terms": terms,
        "features": features,
        "cells": adata.obs_names.astype(str).tolist(),
    }


def compute_niche_term_enrichment(
    label: str,
    X: np.ndarray,
    y: np.ndarray,
    terms: Sequence[str],
    features: Sequence[str],
    active_threshold: float,
    min_cells: int,
) -> pd.DataFrame:
    rows = []
    niches = sorted(np.unique(y).astype(int).tolist())

    for niche in niches:
        in_mask = y == niche
        out_mask = ~in_mask
        n_in = int(in_mask.sum())
        n_out = int(out_mask.sum())
        if n_in < min_cells or n_out < min_cells:
            print(f"[warn] {label} niche={niche}: too few cells (n_in={n_in}, n_out={n_out})")
            continue

        Xin = X[in_mask, :]
        Xout = X[out_mask, :]

        mean_in = Xin.mean(axis=0)
        mean_out = Xout.mean(axis=0)
        effect = mean_in - mean_out

        frac_in = (Xin > active_threshold).mean(axis=0)
        frac_out = (Xout > active_threshold).mean(axis=0)
        delta_frac = frac_in - frac_out

        pvals = np.empty(X.shape[1], dtype=float)
        for j in range(X.shape[1]):
            try:
                _stat, p = ranksums(Xin[:, j], Xout[:, j])
                pvals[j] = float(p)
            except Exception:
                pvals[j] = np.nan

        qvals = bh_fdr(pvals)
        score = effect * np.maximum(delta_frac, 0.0)

        df = pd.DataFrame(
            {
                "model": label,
                "niche": int(niche),
                "term": list(terms),
                "feature_id": list(features),
                "concept_name": [parse_concept_parts(t)["concept_name"] for t in terms],
                "concept_id": [parse_concept_parts(t)["concept_id"] for t in terms],
                "concept_key_id": [parse_concept_parts(t)["concept_key_id"] for t in terms],
                "concept_key_name_id": [parse_concept_parts(t)["concept_key_name_id"] for t in terms],
                "n_in": n_in,
                "n_out": n_out,
                "mean_in": mean_in,
                "mean_out": mean_out,
                "effect_size": effect,
                "frac_active_in": frac_in,
                "frac_active_out": frac_out,
                "delta_frac_active": delta_frac,
                "score": score,
                "pval": pvals,
                "qval": qvals,
            }
        )
        df = df.sort_values(
            ["effect_size", "delta_frac_active", "score"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        df["rank_effect"] = np.arange(1, len(df) + 1)
        rows.append(df)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def parse_concept_parts(term: str) -> Dict[str, str]:
    """
    Parse concept strings like:
      calcium ion binding|GO:0005509|feat3991
      Signaling by ALK in cancer|REAC:R-HSA-9700206|feat11228
      MicroRNAs in cancer|KEGG:05206|feat15697

    Returns:
      concept_name: text before first |
      concept_id: first stable ID-like part, e.g. GO:..., REAC:..., KEGG:...
      feature_suffix: featXXXX if present
      concept_key_name_id: concept_name|concept_id, with feat suffix removed
    """
    raw = str(term).strip()
    parts = [p.strip() for p in raw.split("|")]
    concept_name = parts[0] if parts else raw

    concept_id = ""
    feature_suffix = ""
    for p in parts[1:]:
        pl = p.lower()
        if pl.startswith("feat"):
            feature_suffix = p
        elif (
            p.startswith("GO:")
            or p.startswith("REAC:")
            or p.startswith("KEGG:")
            or p.startswith("WP:")
            or p.startswith("HP:")
            or p.startswith("MIRNA:")
            or p.startswith("TF:")
        ):
            concept_id = p

    if not concept_id:
        # Fallback: if no known ID prefix, use the second field unless it is feat.
        for p in parts[1:]:
            if not p.lower().startswith("feat"):
                concept_id = p
                break

    if concept_id:
        concept_key_name_id = f"{concept_name}|{concept_id}"
        concept_key_id = concept_id
    else:
        # Fallback removes trailing |feat... if present.
        non_feat = [p for p in parts if not p.lower().startswith("feat")]
        concept_key_name_id = "|".join(non_feat) if non_feat else raw
        concept_key_id = concept_key_name_id

    return {
        "raw_term": raw,
        "concept_name": concept_name,
        "concept_id": concept_id,
        "feature_suffix": feature_suffix,
        "concept_key_name_id": concept_key_name_id,
        "concept_key_id": concept_key_id,
    }


def normalize_term_for_overlap(term: str, mode: str = "id") -> str:
    """
    mode='id':
      GO:0005509 / REAC:R-HSA-... / KEGG:05200 only.
      Falls back to name|id or stripped term if no ID exists.

    mode='name_id':
      concept name + ID, with |featXXXX removed.

    mode='raw':
      exact original string except whitespace/lowercase normalization.
    """
    parsed = parse_concept_parts(term)
    if mode == "id":
        key = parsed["concept_key_id"]
    elif mode == "name_id":
        key = parsed["concept_key_name_id"]
    elif mode == "raw":
        key = parsed["raw_term"]
    else:
        raise ValueError(f"Unknown overlap key mode: {mode}")
    return str(key).strip().lower()


BROAD_CATEGORY_RULES: Dict[str, List[str]] = {
    # v2 rules: expanded to reduce uninformative "Other" calls while keeping the
    # same broad category set used in the first category analysis.
    "Tumor/cancer": [
        "cancer", "tumor", "tumour", "carcinoma", "neoplasm", "oncogene",
        "malign", "metastasis", "alk in cancer", "erbb2", "tp53", "p53",
        "micrornas in cancer", "pancreatic cancer", "pathways in cancer",
        "death receptors and ligands", "cell death genes",
    ],
    "Cell cycle/proliferation": [
        "cell cycle", "mitotic", "mitosis", "nuclear envelope", "g1", "g1/s", "g1_s", "s phase",
        "proliferation", "dna replication", "chromosome segregation",
        "cell division", "cellular component disassembly",
    ],
    "Growth factor/kinase signaling": [
        "signaling", "signal transduction", "kinase", "phosphatase",
        "receptor tyrosine kinase", "growth factor", "egfr", "erbb",
        "alk", "mapk", "pi3k", "akt", "tgf", "tgf-beta", "tgf beta",
        "activin receptor", "insulin receptor", "receptor signaling",
        "receptor internalization", "protein localization to plasma membrane",
        "phosphorylation", "gtpase", "calcium ion binding", "hormone",
        "molecular function activator", "activator activity", "stat5", "stat",
        "jak-stat", "jak stat", "membrane raft",
    ],
    "Immune/inflammatory": [
        "immune", "immuno", "inflammatory", "inflammation", "antigen", "mhc",
        "toll-like", "toll like", "tlr", "nf-kappa", "nf kappa", "complement",
        "fc receptor", "chemokine", "cytokine", "leukocyte", "lymphocyte",
        "antiviral", "innate", "adaptive", "interferon", "interleukin",
        "inflammasome", "infection", "infectious", "virus", "virion",
        "viral", "hiv", "cytomegalovirus", "coronavirus", "covid",
        "shigellosis", "malaria", "antimicrobial", "biotic stimulus",
        "peptide secretion", "mediator of immune response", "cd28", "co-stimulation",
        "costimulation", "cell killing", "stat5", "stat activation",
    ],
    "Cytokine/interleukin/interferon": [
        "interleukin", "il-", "il1", "il-1", "il-3", "il-4", "il-6", "il-13",
        "il-15", "il-17", "interferon", "cytokine", "chemokine", "tnf",
        "tgf-beta", "tgf beta", "monocyte chemotactic protein", "mcp-1",
    ],
    "T/NK/B/myeloid": [
        "t cell", "t-cell", "tcr", "cd4", "cd8", "treg", "nk",
        "natural killer", "b cell", "b-cell", "mhc", "macrophage", "monocyte",
        "myeloid", "dendritic", "mast cell", "granule", "cytolytic",
        "leukocyte", "lymphocyte", "thymocyte", "hematopoietic", "cd28",
        "co-stimulation", "costimulation", "cell killing",
    ],
    "Adhesion/ECM/cytoskeleton": [
        "adhesion", "cell-matrix", "cell matrix", "junction", "cadherin",
        "integrin", "extracellular matrix", "ecm", "collagen", "proteoglycan",
        "glycosaminoglycan", "cytoskeleton", "cytoskeletal", "actin",
        "filament", "keratin", "cornified envelope", "muscle contraction", "smooth muscle", "muscle adaptation",
        "contractile", "migration", "motility", "chemotaxis", "filopodium",
        "scaffold protein", "structural constituent of cytoskeleton",
        "axon guidance", "axon", "fasciculation", "cell recognition",
    ],
    "Stress/death/autophagy": [
        "apoptosis", "apoptotic", "cell death", "death receptor", "necrotic",
        "necroptosis", "programmed cell death", "autophagy", "mitochondrion",
        "mitochondria", "cytochrome c", "reactive oxygen", "ros", "oxidative",
        "stress", "uv", "uv-a", "dna damage", "unfolded protein",
        "inflammasome", "amyloid fibril", "osmotic stress", "abiotic stimulus",
        "metal ion", "cellular response to uv", "endopeptidase inhibitor",
        "protease inhibitor", "peptidase inhibitor",
    ],
    "Metabolism/transport": [
        "metabolic", "metabolism", "transport", "transporter", "homeostasis",
        "lipid", "cholesterol", "fatty acid", "anion", "amino acid",
        "glucose", "monosaccharide", "atp binding", "chemical homeostasis",
        "surfactant", "digestion", "absorption", "vascular permeability",
        "fluid shear", "laminar fluid shear", "endosome", "endocytosis",
        "vesicle", "microparticle", "membrane raft", "entry", "binding and entry",
        "nitrogen compound", "circadian rhythm", "post-transcriptional",
        "gene silencing",
    ],
    "Development/differentiation": [
        "development", "differentiation", "morphogenesis", "fate commitment",
        "organogenesis", "organ morphogenesis", "axis elongation", "mesoderm",
        "neural crest", "endocardial cushion", "bronchus", "trabecula",
        "amelogenesis", "epithelial cell fate", "hematopoietic stem cell",
        "reproduction", "ovulation", "cardiocyte", "endocardial",
        "neuron", "nervous system", "commissural neuron", "developmental biology", "cornified envelope", "muscle adaptation",
    ],
}


def assign_broad_categories(concept_name: str, concept_id: str = "") -> List[str]:
    """
    Rule-based, multi-label broad theme assignment.

    This is intended as a transparent descriptive summary of enriched concepts,
    not as a formal ontology/pathway enrichment test.
    """
    text = f"{concept_name} {concept_id}".lower().replace("_", " ")
    cats: List[str] = []
    for cat, keywords in BROAD_CATEGORY_RULES.items():
        if any(k.lower() in text for k in keywords):
            cats.append(cat)
    if not cats:
        cats = ["Other"]
    return cats


def primary_broad_category(categories: Sequence[str]) -> str:
    # Deterministic primary category for plots/tables.
    # Keep category priority in the same order as BROAD_CATEGORY_RULES, then Other.
    priority = list(BROAD_CATEGORY_RULES.keys()) + ["Other"]
    s = set(categories)
    for cat in priority:
        if cat in s:
            return cat
    return "Other"


def add_broad_categories(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cats = []
    prim = []
    for _, r in out.iterrows():
        c = assign_broad_categories(str(r.get("concept_name", r.get("term", ""))), str(r.get("concept_id", "")))
        cats.append(";".join(c))
        prim.append(primary_broad_category(c))
    out["broad_categories"] = cats
    out["primary_category"] = prim
    out["n_broad_categories"] = [len(x.split(";")) for x in cats]
    return out


def summarize_categories(top_terms: pd.DataFrame, labels: Sequence[str], niches: Sequence[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Multi-label category summary.
    Each concept contributes 1 count to each matching category.
    Fractions are count / number of top concepts for that model+niche.
    """
    rows = []
    all_categories = list(BROAD_CATEGORY_RULES.keys()) + ["Other"]

    for model in labels:
        for niche in niches:
            sub = top_terms[(top_terms["model"] == model) & (top_terms["niche"] == niche)].copy()
            denom = max(1, len(sub))
            counts = {cat: 0 for cat in all_categories}
            for cats in sub["broad_categories"].astype(str):
                for cat in cats.split(";"):
                    if cat:
                        counts[cat] = counts.get(cat, 0) + 1
            for cat in all_categories:
                rows.append(
                    {
                        "model": model,
                        "niche": int(niche),
                        "category": cat,
                        "count": int(counts.get(cat, 0)),
                        "fraction_of_top_terms": float(counts.get(cat, 0) / denom),
                        "n_top_terms": int(len(sub)),
                    }
                )

    long = pd.DataFrame(rows)
    wide = long.pivot_table(index=["model", "niche"], columns="category", values="fraction_of_top_terms", fill_value=0.0).reset_index()
    return long, wide


def category_vector_similarity(category_wide: pd.DataFrame, labels: Sequence[str], niches: Sequence[int]) -> pd.DataFrame:
    cat_cols = [c for c in category_wide.columns if c not in {"model", "niche"}]
    rows = []
    for niche in niches:
        sub = category_wide[category_wide["niche"] == niche].set_index("model")
        for i, a in enumerate(labels):
            for b in labels[i + 1:]:
                if a not in sub.index or b not in sub.index:
                    continue
                va = sub.loc[a, cat_cols].to_numpy(dtype=float)
                vb = sub.loc[b, cat_cols].to_numpy(dtype=float)
                denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
                cosine = float(np.dot(va, vb) / denom) if denom > 0 else np.nan
                pa = set([c for c, v in zip(cat_cols, va) if v > 0])
                pb = set([c for c, v in zip(cat_cols, vb) if v > 0])
                union = pa | pb
                inter = pa & pb
                jacc = float(len(inter) / len(union)) if union else np.nan
                rows.append(
                    {
                        "niche": int(niche),
                        "model_a": a,
                        "model_b": b,
                        "category_cosine": cosine,
                        "category_presence_jaccard": jacc,
                        "n_categories_a": len(pa),
                        "n_categories_b": len(pb),
                        "n_categories_intersection": len(inter),
                        "shared_categories": ";".join(sorted(inter)),
                    }
                )
    return pd.DataFrame(rows)


def dominant_categories(category_long: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    rows = []
    for (model, niche), g in category_long.groupby(["model", "niche"], sort=True):
        gg = g.sort_values(["count", "fraction_of_top_terms", "category"], ascending=[False, False, True]).head(top_n)
        for rank, (_, r) in enumerate(gg.iterrows(), start=1):
            rows.append(
                {
                    "model": model,
                    "niche": int(niche),
                    "rank": rank,
                    "category": r["category"],
                    "count": int(r["count"]),
                    "fraction_of_top_terms": float(r["fraction_of_top_terms"]),
                    "n_top_terms": int(r["n_top_terms"]),
                }
            )
    return pd.DataFrame(rows)


def top_terms_by_model_niche(
    enrich: pd.DataFrame,
    top_k: int,
    require_positive_effect: bool,
    overlap_key_mode: str,
) -> pd.DataFrame:
    df = enrich.copy()
    if require_positive_effect:
        df = df[df["effect_size"] > 0].copy()

    outs = []
    for (model, niche), g in df.groupby(["model", "niche"], sort=True):
        gg = g.sort_values(
            ["effect_size", "delta_frac_active", "score"],
            ascending=[False, False, False],
        ).head(top_k).copy()
        gg["top_k"] = int(top_k)
        gg["term_norm"] = gg["term"].map(lambda x: normalize_term_for_overlap(x, mode=overlap_key_mode))
        gg["overlap_key_mode"] = overlap_key_mode
        outs.append(gg)

    if not outs:
        return pd.DataFrame()
    return pd.concat(outs, ignore_index=True)


def compute_term_overlap(
    top_terms: pd.DataFrame,
    labels: Sequence[str],
    niches: Sequence[int],
    top_k: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pair_rows = []
    shared_rows = []
    threeway_rows = []

    for niche in niches:
        term_sets: Dict[str, set] = {}
        display: Dict[str, Dict[str, str]] = {}

        for model in labels:
            sub = top_terms[(top_terms["model"] == model) & (top_terms["niche"] == niche)]
            s = set(sub["term_norm"].astype(str).tolist())
            term_sets[model] = s
            display[model] = dict(zip(sub["term_norm"].astype(str), sub["term"].astype(str)))

        for i, a in enumerate(labels):
            for b in labels[i + 1 :]:
                A, B = term_sets[a], term_sets[b]
                inter = A & B
                union = A | B
                jacc = len(inter) / len(union) if union else np.nan
                overlap_coef = len(inter) / min(len(A), len(B)) if min(len(A), len(B)) else np.nan
                pair_rows.append(
                    {
                        "niche": int(niche),
                        "model_a": a,
                        "model_b": b,
                        "top_k": int(top_k),
                        "n_terms_a": len(A),
                        "n_terms_b": len(B),
                        "n_intersection": len(inter),
                        "n_union": len(union),
                        "jaccard": jacc,
                        "overlap_coefficient": overlap_coef,
                    }
                )

                for t in sorted(inter):
                    shared_rows.append(
                        {
                            "niche": int(niche),
                            "model_a": a,
                            "model_b": b,
                            "term_norm": t,
                            "term_a": display[a].get(t, t),
                            "term_b": display[b].get(t, t),
                        }
                    )

        all_inter = set.intersection(*(term_sets[m] for m in labels)) if labels else set()
        all_union = set.union(*(term_sets[m] for m in labels)) if labels else set()
        threeway_rows.append(
            {
                "niche": int(niche),
                "top_k": int(top_k),
                "n_threeway_intersection": len(all_inter),
                "n_threeway_union": len(all_union),
                "threeway_jaccard": len(all_inter) / len(all_union) if all_union else np.nan,
            }
        )
        for t in sorted(all_inter):
            row = {"niche": int(niche), "model_a": "ALL3", "model_b": "ALL3", "term_norm": t}
            for m in labels:
                row[f"term_{m}"] = display[m].get(t, t)
            shared_rows.append(row)

    return pd.DataFrame(pair_rows), pd.DataFrame(shared_rows), pd.DataFrame(threeway_rows)


def read_labels_for_consistency(label: str, validation_dir: str, label_col: str) -> pd.DataFrame:
    vdir = Path(validation_dir)
    labels_csv = vdir / "global_labels.csv"
    if labels_csv.exists():
        df = pd.read_csv(labels_csv)
        if "cell" not in df.columns:
            raise RuntimeError(f"{labels_csv} must contain a cell column.")
        if label_col not in df.columns:
            raise RuntimeError(
                f"{labels_csv} missing label_col={label_col!r}. "
                f"Available columns: {list(df.columns)}"
            )
        out = df[["cell", label_col]].copy()
        out = out.rename(columns={label_col: f"{label}_niche"})
        out[f"{label}_niche"] = out[f"{label}_niche"].astype(int)
        return out

    h5ad = vdir / "adata_with_niche_labels.h5ad"
    if not h5ad.exists():
        raise RuntimeError(f"Missing both {labels_csv} and {h5ad}")
    adata = sc.read_h5ad(h5ad)
    if label_col not in adata.obs.columns:
        raise RuntimeError(f"{h5ad} adata.obs missing {label_col!r}")
    return pd.DataFrame({"cell": adata.obs_names.astype(str), f"{label}_niche": adata.obs[label_col].astype(int).to_numpy()})


def per_niche_jaccard(a: np.ndarray, b: np.ndarray, niche: int) -> float:
    A = a == niche
    B = b == niche
    union = np.logical_or(A, B).sum()
    inter = np.logical_and(A, B).sum()
    return float(inter / union) if union > 0 else np.nan


def compute_consistency(
    labels: Sequence[str],
    validation_dirs: Sequence[str],
    label_col: str,
    niches: Sequence[int],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    merged = None
    for label, vdir in zip(labels, validation_dirs):
        df = read_labels_for_consistency(label, vdir, label_col)
        merged = df if merged is None else merged.merge(df, on="cell", how="inner")

    if merged is None or merged.empty:
        raise RuntimeError("No overlapping cells across model label files.")

    pair_rows = []
    confusion_mats: Dict[str, pd.DataFrame] = {}
    for i, a in enumerate(labels):
        for b in labels[i + 1 :]:
            ca = f"{a}_niche"
            cb = f"{b}_niche"
            ya = merged[ca].astype(int).to_numpy()
            yb = merged[cb].astype(int).to_numpy()

            row = {
                "model_a": a,
                "model_b": b,
                "n_cells": int(len(merged)),
                "ARI": float(adjusted_rand_score(ya, yb)),
                "NMI": float(normalized_mutual_info_score(ya, yb)),
                "label_accuracy": float((ya == yb).mean()),
            }
            for niche in niches:
                row[f"jaccard_niche{int(niche)}"] = per_niche_jaccard(ya, yb, int(niche))
            pair_rows.append(row)

            cm = pd.crosstab(pd.Series(ya, name=f"{a}_niche"), pd.Series(yb, name=f"{b}_niche"), dropna=False)
            confusion_mats[f"{a}_vs_{b}"] = cm

    niche_cols = [f"{m}_niche" for m in labels]
    arr = merged[niche_cols].astype(int).to_numpy()

    consensus = []
    n_agree = []
    all_agree = []
    at_least_two = []
    for row in arr:
        vals, counts = np.unique(row, return_counts=True)
        max_count = int(counts.max())
        winners = vals[counts == max_count]
        cons = int(winners[0]) if len(winners) == 1 and max_count >= 2 else -1
        consensus.append(cons)
        n_agree.append(max_count)
        all_agree.append(bool(max_count == len(row)))
        at_least_two.append(bool(max_count >= 2))

    merged["consensus_niche"] = consensus
    merged["n_models_agree"] = n_agree
    merged["all_models_agree"] = all_agree
    merged["at_least_two_models_agree"] = at_least_two

    summary_rows = [
        {
            "summary_type": "overall",
            "n_cells": int(len(merged)),
            "n_models": int(len(labels)),
            "fraction_all_models_agree": float(np.mean(all_agree)),
            "fraction_at_least_two_models_agree": float(np.mean(at_least_two)),
        }
    ]

    for niche in niches:
        sub = merged[merged["consensus_niche"] == int(niche)]
        summary_rows.append(
            {
                "summary_type": "by_consensus_niche",
                "consensus_niche": int(niche),
                "n_cells": int(len(sub)),
                "n_models": int(len(labels)),
                "fraction_of_all_cells": float(len(sub) / len(merged)),
                "fraction_all_models_agree_with_this_consensus": float(sub["all_models_agree"].mean()) if len(sub) else np.nan,
            }
        )

    return merged, pd.DataFrame(pair_rows), pd.DataFrame(summary_rows), confusion_mats


def maybe_write_term_lists(top_terms: pd.DataFrame, out_dir: Path) -> None:
    list_dir = ensure_dir(out_dir / "top_term_lists")
    for (model, niche), g in top_terms.groupby(["model", "niche"], sort=True):
        path = list_dir / f"{model}_niche{int(niche)}_top_terms.txt"
        with open(path, "w") as f:
            for _, r in g.iterrows():
                f.write(str(r["term"]) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze niche enriched terms and cross-model niche consistency.")
    ap.add_argument("--labels", nargs=3, required=True)
    ap.add_argument("--validation-dirs", nargs=3, required=True)
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--label-col", default="niche", help="Usually 'niche' for relabeled IDs, or 'niche_raw'.")
    ap.add_argument("--term-col", default=None, help="Optional adata.var column to use as term name.")
    ap.add_argument("--feature-id-col", default=None, help="Optional adata.var column to use as feature ID.")
    ap.add_argument("--active-threshold", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=25)
    ap.add_argument(
        "--overlap-key-mode",
        default="id",
        choices=["id", "name_id", "raw"],
        help="Key used for cross-model term overlap. 'id' compares GO/REAC/KEGG IDs only; 'name_id' strips feat suffix; 'raw' is exact.",
    )
    ap.add_argument("--min-cells", type=int, default=20)
    ap.add_argument("--require-positive-effect", action="store_true", default=True)
    ap.add_argument("--no-require-positive-effect", dest="require_positive_effect", action="store_false")
    ap.add_argument("--niches", nargs="+", type=int, default=None, help="Default: infer from labels.")

    args = ap.parse_args()
    out_dir = ensure_dir(args.out_dir)

    print("=" * 100)
    print("[load models]")
    print("=" * 100)

    payloads = []
    inferred_niches = set()
    for label, vdir in zip(args.labels, args.validation_dirs):
        payload = load_model_payload(label, vdir, args.label_col, args.term_col, args.feature_id_col)
        payloads.append(payload)
        inferred_niches.update(np.unique(payload["y"]).astype(int).tolist())
        print(
            f"{label}: X={payload['X'].shape}, "
            f"n_cells={len(payload['cells'])}, "
            f"niches={sorted(np.unique(payload['y']).astype(int).tolist())}, "
            f"validation_dir={vdir}"
        )

    niches = sorted(args.niches if args.niches is not None else inferred_niches)
    print("Using niches:", niches)

    print("=" * 100)
    print("[1] niche term enrichment")
    print("=" * 100)

    enrich_frames = []
    for p in payloads:
        df = compute_niche_term_enrichment(
            label=p["label"],
            X=p["X"],
            y=p["y"],
            terms=p["terms"],
            features=p["features"],
            active_threshold=float(args.active_threshold),
            min_cells=int(args.min_cells),
        )
        enrich_frames.append(df)
        model_out = ensure_dir(out_dir / p["label"])
        df.to_csv(model_out / "niche_term_enrichment.csv", index=False)
        print(f"[OK] {p['label']} enrichment rows={len(df)}")

    enrich = pd.concat(enrich_frames, ignore_index=True)
    enrich.to_csv(out_dir / "combined_niche_term_enrichment.csv", index=False)

    top_terms = top_terms_by_model_niche(
        enrich,
        int(args.top_k),
        bool(args.require_positive_effect),
        overlap_key_mode=str(args.overlap_key_mode),
    )
    top_terms = add_broad_categories(top_terms)
    top_terms.to_csv(out_dir / f"combined_top{int(args.top_k)}_terms_by_model_niche.csv", index=False)

    # Concept-only view: useful for manual sanity check and cross-model comparison.
    concept_view_cols = [
        "model", "niche", "rank_effect", "term", "concept_name", "concept_id",
        "concept_key_id", "concept_key_name_id", "broad_categories", "primary_category",
        "effect_size", "delta_frac_active",
        "frac_active_in", "frac_active_out", "qval",
    ]
    top_terms[[c for c in concept_view_cols if c in top_terms.columns]].to_csv(
        out_dir / f"combined_top{int(args.top_k)}_concepts_by_model_niche.csv",
        index=False,
    )
    maybe_write_term_lists(top_terms, out_dir)

    category_long, category_wide = summarize_categories(top_terms, list(args.labels), niches)
    category_long.to_csv(out_dir / f"category_counts_long_top{int(args.top_k)}.csv", index=False)
    category_wide.to_csv(out_dir / f"category_fractions_wide_top{int(args.top_k)}.csv", index=False)

    category_sim = category_vector_similarity(category_wide, list(args.labels), niches)
    category_sim.to_csv(out_dir / f"category_pairwise_similarity_top{int(args.top_k)}.csv", index=False)

    dom_cats = dominant_categories(category_long, top_n=5)
    dom_cats.to_csv(out_dir / f"dominant_categories_by_model_niche_top{int(args.top_k)}.csv", index=False)

    # Audit remaining "Other" assignments, so the keyword rules can be refined transparently.
    other_terms = top_terms[top_terms["broad_categories"].astype(str).str.contains("Other", regex=False)].copy()
    other_cols = [
        "model", "niche", "rank_effect", "term", "concept_name", "concept_id",
        "effect_size", "delta_frac_active", "qval", "broad_categories", "primary_category",
    ]
    other_terms[[c for c in other_cols if c in other_terms.columns]].to_csv(
        out_dir / f"other_terms_audit_top{int(args.top_k)}.csv",
        index=False,
    )

    print("Category pairwise similarity:")
    if len(category_sim):
        print(category_sim.to_string(index=False))

    pair_overlap, shared_terms, threeway = compute_term_overlap(top_terms, list(args.labels), niches, int(args.top_k))
    pair_overlap.to_csv(out_dir / f"term_overlap_pairwise_{args.overlap_key_mode}_top{int(args.top_k)}.csv", index=False)
    shared_terms.to_csv(out_dir / f"term_overlap_shared_terms_{args.overlap_key_mode}_top{int(args.top_k)}.csv", index=False)
    threeway.to_csv(out_dir / f"term_overlap_threeway_{args.overlap_key_mode}_top{int(args.top_k)}.csv", index=False)

    print("Pairwise top-term overlap:")
    if len(pair_overlap):
        print(pair_overlap.to_string(index=False))

    print("=" * 100)
    print("[2] cross-model niche label consistency")
    print("=" * 100)

    merged, pair_consistency, consensus_summary, confusion_mats = compute_consistency(
        labels=list(args.labels),
        validation_dirs=list(args.validation_dirs),
        label_col=args.label_col,
        niches=niches,
    )
    merged.to_csv(out_dir / "cell_niche_assignments_3models.csv", index=False)
    pair_consistency.to_csv(out_dir / "niche_label_pairwise_consistency.csv", index=False)
    consensus_summary.to_csv(out_dir / "niche_consensus_summary.csv", index=False)

    cm_dir = ensure_dir(out_dir / "confusion_matrices")
    for name, cm in confusion_mats.items():
        cm.to_csv(cm_dir / f"{name}.csv")

    print("Pairwise niche consistency:")
    print(pair_consistency.to_string(index=False))
    print("Consensus summary:")
    print(consensus_summary.to_string(index=False))

    summary = {
        "labels": list(args.labels),
        "validation_dirs": list(args.validation_dirs),
        "out_dir": str(out_dir),
        "label_col": args.label_col,
        "term_col": args.term_col,
        "feature_id_col": args.feature_id_col,
        "active_threshold": float(args.active_threshold),
        "top_k": int(args.top_k),
        "overlap_key_mode": str(args.overlap_key_mode),
        "category_rule_set": "v4_targeted_other_reduction_fixed_dominant_output",
        "niches": niches,
        "n_cells_overlap": int(len(merged)),
        "outputs": {
            "combined_niche_term_enrichment": str(out_dir / "combined_niche_term_enrichment.csv"),
            "combined_top_terms": str(out_dir / f"combined_top{int(args.top_k)}_terms_by_model_niche.csv"),
            "combined_top_concepts_with_categories": str(out_dir / f"combined_top{int(args.top_k)}_concepts_by_model_niche.csv"),
            "category_counts_long": str(out_dir / f"category_counts_long_top{int(args.top_k)}.csv"),
            "category_fractions_wide": str(out_dir / f"category_fractions_wide_top{int(args.top_k)}.csv"),
            "category_pairwise_similarity": str(out_dir / f"category_pairwise_similarity_top{int(args.top_k)}.csv"),
            "dominant_categories": str(out_dir / f"dominant_categories_by_model_niche_top{int(args.top_k)}.csv"),
            "other_terms_audit": str(out_dir / f"other_terms_audit_top{int(args.top_k)}.csv"),
            "term_overlap_pairwise": str(out_dir / f"term_overlap_pairwise_{args.overlap_key_mode}_top{int(args.top_k)}.csv"),
            "term_overlap_shared_terms": str(out_dir / f"term_overlap_shared_terms_{args.overlap_key_mode}_top{int(args.top_k)}.csv"),
            "term_overlap_threeway": str(out_dir / f"term_overlap_threeway_{args.overlap_key_mode}_top{int(args.top_k)}.csv"),
            "cell_assignments": str(out_dir / "cell_niche_assignments_3models.csv"),
            "pairwise_consistency": str(out_dir / "niche_label_pairwise_consistency.csv"),
            "consensus_summary": str(out_dir / "niche_consensus_summary.csv"),
        },
    }
    save_json(out_dir / "analysis_summary.json", summary)

    print("\n[OK] wrote:", out_dir)


if __name__ == "__main__":
    main()
