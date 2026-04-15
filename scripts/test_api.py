from __future__ import annotations

import importlib
import inspect
from typing import Dict, List, Tuple


MODULES = [
    "interp_pipeline.downstream.interaction.edges",
    "interp_pipeline.downstream.interaction.preprocess",
    "interp_pipeline.downstream.interaction.grouping",
    "interp_pipeline.downstream.interaction.aggregate",
    "interp_pipeline.downstream.interaction.score",
    "interp_pipeline.downstream.interaction.report",
    "interp_pipeline.downstream.interaction.plot",
]

# What we need for CCC to run (logical roles)
REQUIRED = {
    "edges": ["EdgeConfig", "build_edges_radius", "distance_weights", "tile_ids"],
    "preprocess": ["PreprocessAConfig", "PreprocessBConfig", "preprocess_variant_a", "preprocess_variant_b"],
    "grouping": ["encode_groups", "permute_within_tiles"],
    "aggregate": ["AggConfig", "aggregate_s1_s2"],
    "score": ["ScoreConfig", "neff_from_s1_s2", "compute_z", "intensity_topk_median"],
    "report": ["top_pairs_summary", "top_drivers_for_pair"],
    "plot": ["plot_intensity_scatter", "plot_within_vs_cross_boxplot", "plot_intensity_heatmap", "plot_top_pair_drivers"],
}


def public_names(mod) -> List[str]:
    return sorted([n for n in dir(mod) if not n.startswith("_")])


def is_callable_obj(x) -> bool:
    return callable(x) and not isinstance(x, type)


def fmt_sig(obj) -> str:
    try:
        return str(inspect.signature(obj))
    except Exception:
        return "(signature unavailable)"


def print_module(modpath: str) -> None:
    print("\n" + "=" * 88)
    print(modpath)
    print("=" * 88)
    mod = importlib.import_module(modpath)

    names = public_names(mod)
    print(f"Public symbols ({len(names)}):")
    for n in names:
        obj = getattr(mod, n)
        if isinstance(obj, type):
            print(f"  CLASS  {n}")
        elif callable(obj):
            print(f"  FUNC   {n}{fmt_sig(obj)}")
        else:
            # show simple constants too
            if isinstance(obj, (int, float, str, bool, tuple, list, dict)):
                s = str(obj)
                s = s if len(s) <= 80 else s[:77] + "..."
                print(f"  CONST  {n} = {s}")
            else:
                print(f"  VAR    {n} ({type(obj).__name__})")


def best_guess(mod, want: str) -> str | None:
    """
    Heuristic: find an exported name that contains all key tokens of want.
    """
    names = public_names(mod)
    tokens = want.split("_")
    hits = []
    for n in names:
        ln = n.lower()
        score = sum(t in ln for t in tokens)
        if score > 0:
            hits.append((score, n))
    hits.sort(reverse=True)
    return hits[0][1] if hits else None


def main():
    mods = {}
    for m in MODULES:
        mods[m] = importlib.import_module(m)

    # print everything with signatures
    for m in MODULES:
        print_module(m)

    # provide a mapping suggestion per module
    print("\n" + "#" * 88)
    print("SUGGESTED API NAME MAPPING (heuristic)")
    print("#" * 88)

    short = {
        "edges": mods["interp_pipeline.downstream.interaction.edges"],
        "preprocess": mods["interp_pipeline.downstream.interaction.preprocess"],
        "grouping": mods["interp_pipeline.downstream.interaction.grouping"],
        "aggregate": mods["interp_pipeline.downstream.interaction.aggregate"],
        "score": mods["interp_pipeline.downstream.interaction.score"],
        "report": mods["interp_pipeline.downstream.interaction.report"],
        "plot": mods["interp_pipeline.downstream.interaction.plot"],
    }

    for section, wants in REQUIRED.items():
        mod = short[section]
        names = set(public_names(mod))
        print(f"\n[{section}]")
        for w in wants:
            if w in names:
                print(f"  OK      {w}")
            else:
                guess = best_guess(mod, w)
                print(f"  MISSING {w}  -> maybe: {guess}")

    print("\nDone. Use this output to update scripts/test_ccc.py imports precisely.")


if __name__ == "__main__":
    main()