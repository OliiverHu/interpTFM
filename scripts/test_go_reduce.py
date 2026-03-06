# scripts/test_go_reduce.py
from __future__ import annotations

import os
import pandas as pd

from interp_pipeline.downstream.f1.go_reduce import (
    ensure_go_basic_obo,
    nmi_sweep,
    reduce_go_terms_strategy_c,
    ParentSelectConfig,
)

# =========================
# CONFIG (EDIT HERE)
# =========================
RUNS_ROOT = "runs/full_scgpt_cosmx"
LAYER = "layer_4.norm2"

F1_TABLE = os.path.join(RUNS_ROOT, "heldout_report", LAYER, "valid_concept_f1_scores.csv")
GO_OBO_PATH = "resources/go-basic.obo"

OUTDIR = os.path.join(RUNS_ROOT, "f1_analysis", "go_parent_nmi", LAYER)
os.makedirs(OUTDIR, exist_ok=True)

MAX_DESC_SWEEP = [50, 100, 200, 500, 1000, 2000, 5000]
F1_MIN_SWEEP = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

# Strategy-C settings (notebook aligned)
EXCLUDE_SELF = True
ALLOW_SELF_FALLBACK = True

# quick sanity mapping check
SANITY_MAX_DESC = 500
SANITY_PRINT_N = 15
# =========================


def main():
    # 0) ensure GO DAG file exists
    ensure_go_basic_obo(GO_OBO_PATH)

    # 1) load heldout F1 table
    if not os.path.exists(F1_TABLE):
        raise FileNotFoundError(f"Missing F1 table: {F1_TABLE}")
    f1_df = pd.read_csv(F1_TABLE)

    # expected columns in your file
    needed = {"concept", "feature", "f1"}
    missing = needed - set(f1_df.columns)
    if missing:
        raise RuntimeError(f"{F1_TABLE} missing columns {missing}. have={list(f1_df.columns)}")

    # 2) run sweep (concept-level NMI between best-feature labels and GO-parent labels)
    sweep = nmi_sweep(
        f1_df,
        go_obo_path=GO_OBO_PATH,
        max_descendants_grid=MAX_DESC_SWEEP,
        f1_min_grid=F1_MIN_SWEEP,
        concept_col="concept",
        feature_col="feature",
        f1_col="f1",
        only_go=True,
        exclude_self=EXCLUDE_SELF,
        allow_self_fallback=ALLOW_SELF_FALLBACK,
    )
    out_path = os.path.join(OUTDIR, "nmi_sweep.csv")
    sweep.to_csv(out_path, index=False)

    # 3) pick best (highest NMI, then most concepts)
    sweep_valid = sweep.dropna(subset=["nmi"]).copy()
    if len(sweep_valid):
        best = sweep_valid.sort_values(["nmi", "n_concepts"], ascending=[False, False]).head(1)
        best_path = os.path.join(OUTDIR, "best_setting.csv")
        best.to_csv(best_path, index=False)
        print("[OK] wrote:", out_path)
        print("[OK] wrote:", best_path)
        print(best.to_string(index=False))
    else:
        print("[OK] wrote:", out_path)
        print("[warn] no valid sweep rows (did filtering remove everything?)")

    # 4) sanity-check: does reduction actually change anything?
    # Build mapping for unique GO concepts appearing in the file
    unique_go = sorted(set([c for c in f1_df["concept"].astype(str).unique().tolist() if c.startswith("GO:")]))
    cfg = ParentSelectConfig(
        max_descendants=SANITY_MAX_DESC,
        exclude_self=EXCLUDE_SELF,
        allow_self_fallback=ALLOW_SELF_FALLBACK,
    )
    mapping = reduce_go_terms_strategy_c(unique_go, go_obo_path=GO_OBO_PATH, cfg=cfg)

    n_changed = sum(1 for k, v in mapping.items() if k != v)
    print(f"\n[sanity] max_descendants={SANITY_MAX_DESC}  changed={n_changed}/{len(mapping)} ({(n_changed/max(1,len(mapping)))*100:.2f}%)")
    ex = []
    for k in unique_go[:SANITY_PRINT_N]:
        ex.append((k, mapping.get(k, k)))
    print("[sanity] examples (go -> parent):")
    for a, b in ex:
        print(" ", a, "->", b)


if __name__ == "__main__":
    main()