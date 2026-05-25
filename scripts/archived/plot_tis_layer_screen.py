#!/usr/bin/env python
from __future__ import annotations

import argparse

from interp_pipeline.tis.layer_plot import write_tis_layer_plots


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate and plot TIS layer-screen outputs.")
    p.add_argument("--tis-root", default="runs/tis_layer_screen_3models_cosmx")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--metric", default="tis_mean", choices=["tis_mean", "tis_median", "tis_p90", "tis_gap_mean"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outputs = write_tis_layer_plots(tis_root=args.tis_root, out_dir=args.out_dir, metric=args.metric)
    for label, path in outputs.items():
        print(f"Wrote {label}: {path}")


if __name__ == "__main__":
    main()
