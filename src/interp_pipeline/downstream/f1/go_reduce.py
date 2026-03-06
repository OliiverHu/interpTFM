from __future__ import annotations

import os
import shutil
import urllib.request
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# =========================
# Download + load GO DAG
# =========================
def ensure_go_basic_obo(
    go_obo_path: str,
    urls: Optional[List[str]] = None,
    timeout_sec: int = 60,
) -> str:
    """
    Ensure go-basic.obo exists locally. If missing, download it.
    Robust to 403/redirects by using a browser-like User-Agent and curl/wget fallbacks.
    """
    if urls is None:
        urls = [
            "https://purl.obolibrary.org/obo/go/go-basic.obo",
            "https://geneontology.org/ontology/go-basic.obo",
            "http://geneontology.org/ontology/go-basic.obo",
        ]

    os.makedirs(os.path.dirname(go_obo_path) or ".", exist_ok=True)
    if os.path.exists(go_obo_path) and os.path.getsize(go_obo_path) > 0:
        return go_obo_path

    tmp_path = go_obo_path + ".tmp"

    def _try_urllib(url: str) -> bool:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
                ),
                "Accept": "*/*",
            },
            method="GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_sec) as r, open(tmp_path, "wb") as f:
                shutil.copyfileobj(r, f)
            return os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0
        except Exception:
            return False

    def _try_cmd(cmd: List[str]) -> bool:
        import subprocess
        try:
            subprocess.check_call(cmd)
            return os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0
        except Exception:
            return False

    print(f"[GO] go-basic.obo not found. Downloading to: {go_obo_path}")
    ok = False
    last_url = None

    for url in urls:
        last_url = url
        if _try_urllib(url):
            ok = True
            break

        curl = shutil.which("curl")
        if curl and _try_cmd([curl, "-L", "-A", "Mozilla/5.0", "-o", tmp_path, url]):
            ok = True
            break

        wget = shutil.which("wget")
        if wget and _try_cmd([wget, "-O", tmp_path, "--user-agent=Mozilla/5.0", url]):
            ok = True
            break

    if not ok:
        raise RuntimeError(
            f"Failed to download go-basic.obo. Last tried: {last_url}\n"
            f"Please download manually and place at: {go_obo_path}\n"
            f"Suggested URL: https://purl.obolibrary.org/obo/go/go-basic.obo"
        )

    os.replace(tmp_path, go_obo_path)
    print("[GO] download complete.")
    return go_obo_path


def load_go_dag(go_obo_path: str):
    """
    Loads goatools GODag, downloading go-basic.obo if needed.
    """
    try:
        from goatools.obo_parser import GODag  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "goatools is required for GO DAG reduction.\n"
            "Install: pip install goatools\n"
            f"Import error: {e}"
        )

    go_obo_path = ensure_go_basic_obo(go_obo_path)
    return GODag(go_obo_path)


# =========================
# Strategy-C parent selection (notebook-aligned)
# =========================
def _is_go_id(x: str) -> bool:
    return isinstance(x, str) and x.startswith("GO:")


def _ancestors(go_dag, go_id: str) -> List[str]:
    term = go_dag.get(go_id)
    if term is None:
        return []
    try:
        return list(term.get_all_parents())
    except Exception:
        return []


def _depth(go_dag, go_id: str) -> int:
    term = go_dag.get(go_id)
    if term is None:
        return -1
    # goatools term often has .depth
    d = getattr(term, "depth", None)
    try:
        return int(d) if d is not None else -1
    except Exception:
        return -1


def build_descendant_count_cache(go_dag, go_ids: Iterable[str]) -> Dict[str, int]:
    """
    Cache descendant counts for GO ids encountered (terms and ancestors).
    """
    cache: Dict[str, int] = {}

    def desc_count(g: str) -> int:
        if g in cache:
            return cache[g]
        term = go_dag.get(g)
        if term is None:
            cache[g] = 10**9
            return cache[g]
        try:
            cache[g] = len(term.get_all_children())
        except Exception:
            cache[g] = 10**9
        return cache[g]

    # warm cache for go_ids + their ancestors
    for gid in go_ids:
        if not _is_go_id(str(gid)):
            continue
        g = str(gid)
        desc_count(g)
        for a in _ancestors(go_dag, g):
            desc_count(a)

    return cache


def build_parent_frequency(
    go_dag,
    go_ids: Iterable[str],
    *,
    max_descendants: int,
    desc_cache: Dict[str, int],
    exclude_self: bool = True,
) -> Dict[str, int]:
    """
    Parent frequency used as a tie-breaker (notebook-style):
    count how often a candidate parent appears as "eligible" across concepts.
    """
    freq: Dict[str, int] = {}
    for gid in go_ids:
        if not _is_go_id(str(gid)):
            continue
        g = str(gid)
        cands = _ancestors(go_dag, g)
        if not cands:
            continue
        for p in cands:
            if exclude_self and p == g:
                continue
            if desc_cache.get(p, 10**9) <= int(max_descendants):
                freq[p] = freq.get(p, 0) + 1
    return freq


@dataclass(frozen=True)
class ParentSelectConfig:
    max_descendants: int
    exclude_self: bool = True           # IMPORTANT: exclude self so reduction can happen
    allow_self_fallback: bool = True    # if no eligible parent, fall back to self


def choose_parent_strategy_c(
    go_dag,
    go_id: str,
    cfg: ParentSelectConfig,
    *,
    desc_cache: Dict[str, int],
    parent_freq: Optional[Dict[str, int]] = None,
) -> str:
    """
    Notebook-aligned Strategy C:
      - candidates are ancestors (optionally exclude self)
      - keep those with descendant_count <= max_descendants
      - pick most specific allowed: MIN descendant_count
      - tie-breakers:
          1) deeper depth (max depth)
          2) higher parent frequency in dataset (max freq)
          3) lexicographic GO id
      - if none eligible: fallback to self (optional)
    """
    if not _is_go_id(go_id):
        return go_id

    cands = _ancestors(go_dag, go_id)
    if not cands:
        return go_id

    eligible: List[str] = []
    for c in cands:
        if cfg.exclude_self and c == go_id:
            continue
        if desc_cache.get(c, 10**9) <= int(cfg.max_descendants):
            eligible.append(c)

    if not eligible:
        return go_id if cfg.allow_self_fallback else ""

    def key_fn(c: str) -> Tuple[int, int, int, str]:
        # smaller descendants => more specific
        d = desc_cache.get(c, 10**9)
        # larger depth preferred
        dep = _depth(go_dag, c)
        # larger freq preferred
        f = parent_freq.get(c, 0) if parent_freq else 0
        # sort by: descendants asc, depth desc, freq desc, id asc
        return (d, -dep, -f, c)

    eligible.sort(key=key_fn)
    return eligible[0]


def reduce_go_terms_strategy_c(
    go_ids: Iterable[str],
    *,
    go_obo_path: str,
    cfg: ParentSelectConfig,
) -> Dict[str, str]:
    """
    Map go_id -> chosen_parent_id using Strategy C and dataset frequency tie-break.
    """
    go_dag = load_go_dag(go_obo_path)
    go_ids = [str(g) for g in go_ids if _is_go_id(str(g))]
    desc_cache = build_descendant_count_cache(go_dag, go_ids)
    parent_freq = build_parent_frequency(
        go_dag, go_ids, max_descendants=cfg.max_descendants, desc_cache=desc_cache, exclude_self=cfg.exclude_self
    )

    mapping: Dict[str, str] = {}
    for gid in go_ids:
        mapping[gid] = choose_parent_strategy_c(
            go_dag, gid, cfg, desc_cache=desc_cache, parent_freq=parent_freq
        )
    return mapping


# =========================
# Notebook-style NMI evaluation
# =========================
def best_feature_per_concept(
    f1_df: pd.DataFrame,
    *,
    concept_col: str = "concept",
    feature_col: str = "feature",
    f1_col: str = "f1",
    f1_min: float = 0.0,
    only_go: bool = True,
) -> pd.DataFrame:
    """
    For each concept, pick feature with maximum F1.
    Returns: concept, best_feature, best_f1
    """
    df = f1_df[[concept_col, feature_col, f1_col]].copy()
    df[concept_col] = df[concept_col].astype(str)
    if only_go:
        df = df[df[concept_col].str.startswith("GO:")].copy()

    df = df.sort_values([concept_col, f1_col], ascending=[True, False])
    best = df.groupby(concept_col, as_index=False).first()
    best = best.rename(columns={feature_col: "best_feature", f1_col: "best_f1"})

    if f1_min > 0:
        best = best[best["best_f1"] >= float(f1_min)].copy()

    return best


def go_parent_per_concept(
    concepts: Iterable[str],
    *,
    go_obo_path: str,
    cfg: ParentSelectConfig,
) -> pd.DataFrame:
    """
    Map each GO concept -> selected parent using Strategy C.
    Returns: concept, parent_go
    """
    concepts = [str(c) for c in concepts if str(c).startswith("GO:")]
    mapping = reduce_go_terms_strategy_c(concepts, go_obo_path=go_obo_path, cfg=cfg)
    out = pd.DataFrame({"concept": concepts})
    out["parent_go"] = out["concept"].map(lambda x: mapping.get(x, x))
    return out


def encode_labels(values: pd.Series) -> np.ndarray:
    vals = values.astype(str).fillna("")
    uniq = {v: i for i, v in enumerate(pd.unique(vals))}
    return vals.map(uniq).to_numpy(dtype=np.int32)


def compute_nmi_ari_ami(labels_a: pd.Series, labels_b: pd.Series) -> Dict[str, float]:
    try:
        from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score
    except Exception as e:
        raise RuntimeError(f"scikit-learn required for NMI/ARI/AMI. Install: pip install scikit-learn. Error: {e}")

    a = encode_labels(labels_a)
    b = encode_labels(labels_b)
    return {
        "nmi": float(normalized_mutual_info_score(a, b)),
        "ari": float(adjusted_rand_score(a, b)),
        "ami": float(adjusted_mutual_info_score(a, b)),
    }


def nmi_sweep(
    f1_df: pd.DataFrame,
    *,
    go_obo_path: str,
    max_descendants_grid: List[int],
    f1_min_grid: List[float],
    concept_col: str = "concept",
    feature_col: str = "feature",
    f1_col: str = "f1",
    only_go: bool = True,
    exclude_self: bool = True,
    allow_self_fallback: bool = True,
) -> pd.DataFrame:
    """
    Notebook-style sweep:
      labels_a = best_feature per concept
      labels_b = go_parent per concept (Strategy C)
      score = NMI(labels_a, labels_b)
    """
    rows = []
    for f1_min in f1_min_grid:
        best = best_feature_per_concept(
            f1_df,
            concept_col=concept_col,
            feature_col=feature_col,
            f1_col=f1_col,
            f1_min=float(f1_min),
            only_go=only_go,
        )
        if best.empty:
            for md in max_descendants_grid:
                rows.append(
                    dict(
                        f1_min=float(f1_min),
                        max_descendants=int(md),
                        n_concepts=0,
                        nmi=np.nan,
                        ari=np.nan,
                        ami=np.nan,
                        n_unique_feature_labels=0,
                        n_unique_parent_labels=0,
                    )
                )
            continue

        concepts = best["concept"].astype(str).tolist()
        f1_labels = best.set_index("concept")["best_feature"].astype(str)

        for md in max_descendants_grid:
            cfg = ParentSelectConfig(
                max_descendants=int(md),
                exclude_self=exclude_self,
                allow_self_fallback=allow_self_fallback,
            )
            parent_df = go_parent_per_concept(concepts, go_obo_path=go_obo_path, cfg=cfg)
            parent_labels = parent_df.set_index("concept")["parent_go"].astype(str)

            common = f1_labels.index.intersection(parent_labels.index)
            a = f1_labels.loc[common]
            b = parent_labels.loc[common]

            mets = compute_nmi_ari_ami(a, b)

            rows.append(
                dict(
                    f1_min=float(f1_min),
                    max_descendants=int(md),
                    n_concepts=int(len(common)),
                    nmi=mets["nmi"],
                    ari=mets["ari"],
                    ami=mets["ami"],
                    n_unique_feature_labels=int(a.nunique()),
                    n_unique_parent_labels=int(b.nunique()),
                )
            )

    return pd.DataFrame(rows).sort_values(["f1_min", "max_descendants"])