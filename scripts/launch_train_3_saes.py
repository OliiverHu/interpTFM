#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path


DEFAULT_LABELS = ["scgpt", "c2sscale", "geneformer"]
DEFAULT_STORE_ROOTS = [
    "runs/full_scgpt_cosmx",
    "runs/full_c2sscale_cosmx",
    "runs/full_geneformer_cosmx",
]
DEFAULT_LAYERS = ["layer_4.norm2", "layer_17", "layer_4"]
DEFAULT_OUT_DIRS = [
    "runs/full_scgpt_cosmx/sae/layer_4.norm2",
    "runs/full_c2sscale_cosmx/sae/layer_17",
    "runs/full_geneformer_cosmx/sae/layer_4",
]


def fmt_float(x: float) -> str:
    s = f"{x:.0e}" if x < 1e-2 else str(x)
    return s.replace("+0", "").replace("+", "").replace("-0", "-").replace(".", "p")


def main():
    ap = argparse.ArgumentParser(
        description="Launch multiple 3-model SAE training runs from the current trainer."
    )
    ap.add_argument("--trainer-script", default="test_train_3saes.py")
    ap.add_argument("--labels", nargs=3, default=DEFAULT_LABELS)
    ap.add_argument("--store-roots", nargs=3, default=DEFAULT_STORE_ROOTS)
    ap.add_argument("--layers", nargs=3, default=DEFAULT_LAYERS)
    ap.add_argument("--base-out-dirs", nargs=3, default=DEFAULT_OUT_DIRS)

    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--latent-multiplier", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-steps", type=int, default=1000)
    ap.add_argument("--resample-steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--best-metric", choices=["loss", "recon", "sparsity"], default="loss")
    ap.add_argument("--save-every", type=int, default=0)

    # Sweep values
    ap.add_argument("--steps-list", nargs="+", type=int, default=[8000])
    ap.add_argument("--l1-list", nargs="+", type=float, default=[1e-3, 3e-3, 1e-2])

    # Default to no-resample for this iteration
    ap.add_argument("--no-resample", action="store_true", default=True)
    ap.add_argument("--allow-resample", action="store_true", help="Override and keep resampling on")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    use_no_resample = args.no_resample and not args.allow_resample

    trainer_script = Path(args.trainer_script)
    if not trainer_script.exists():
        raise FileNotFoundError(f"Trainer script not found: {trainer_script}")

    for steps in args.steps_list:
        for l1 in args.l1_list:
            tag = f"nr_{'on' if use_no_resample else 'off'}__steps_{steps}__l1_{fmt_float(l1)}"
            out_dirs = [os.path.join(base, tag) for base in args.base_out_dirs]

            cmd = [
                "python",
                str(trainer_script),
                "--labels", *args.labels,
                "--store-roots", *args.store_roots,
                "--layers", *args.layers,
                "--out-dirs", *out_dirs,
                "--device", args.device,
                "--batch-size", str(args.batch_size),
                "--latent-multiplier", str(args.latent_multiplier),
                "--l1", str(l1),
                "--lr", str(args.lr),
                "--steps", str(steps),
                "--warmup-steps", str(args.warmup_steps),
                "--resample-steps", str(args.resample_steps),
                "--seed", str(args.seed),
                "--best-metric", args.best_metric,
                "--save-every", str(args.save_every),
            ]
            if use_no_resample:
                cmd.append("--no-resample")

            print("\n" + "=" * 120)
            print("RUN TAG:", tag)
            print("CMD:")
            print(" ".join(shlex.quote(x) for x in cmd))
            print("=" * 120)

            if not args.dry_run:
                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

# python launch_train_3saes_sweep.py \
#   --steps-list 8000 \
#   --l1-list 1e-3 3e-3 1e-2