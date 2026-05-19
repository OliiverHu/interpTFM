#!/usr/bin/env python3
from __future__ import annotations

"""
Batch-create F1-compatible max1-normalized SAE checkpoints for 3 models.

This is the standalone normalization test script. It calls:
  interp_pipeline.sae.normalize.normalize_sae_checkpoint_for_f1

It does NOT run heldout F1. Use test_run_f1heldout_3models_max1norm.py if you
want normalize + heldout in one command.
"""

import argparse

from interp_pipeline.sae.normalize import normalize_sae_checkpoint_for_f1


def main() -> None:
    ap = argparse.ArgumentParser(description="Create max1-normalized SAE checkpoints for 3-model F1 sensitivity.")
    ap.add_argument("--labels", nargs=3, required=True)
    ap.add_argument("--store-roots", nargs=3, required=True)
    ap.add_argument("--layers", nargs=3, required=True)
    ap.add_argument("--input-sae-ckpts", nargs=3, required=True)
    ap.add_argument("--output-sae-ckpts", nargs=3, required=True)

    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--shard-offset", type=int, default=0)

    ap.add_argument("--min-scale", type=float, default=1e-3)
    ap.add_argument("--min-active-rate", type=float, default=1e-4)

    ap.add_argument("--scale-decoder", action="store_true", default=True)
    ap.add_argument("--no-scale-decoder", dest="scale_decoder", action="store_false")

    args = ap.parse_args()

    for label, store_root, layer, input_ckpt, output_ckpt in zip(
        args.labels,
        args.store_roots,
        args.layers,
        args.input_sae_ckpts,
        args.output_sae_ckpts,
    ):
        print("=" * 100)
        print(f"[normalize SAE for F1] {label} | {layer}")
        print(f"  input : {input_ckpt}")
        print(f"  output: {output_ckpt}")
        print("=" * 100)

        out = normalize_sae_checkpoint_for_f1(
            ckpt_path=input_ckpt,
            store_root=store_root,
            layer=layer,
            output_path=output_ckpt,
            label=label,
            device=args.device,
            batch_size=args.batch_size,
            max_shards=args.max_shards,
            shard_offset=args.shard_offset,
            min_scale=args.min_scale,
            min_active_rate=args.min_active_rate,
            scale_decoder=args.scale_decoder,
        )
        print(f"[OK] saved: {out}")

    print("\nDONE")


if __name__ == "__main__":
    main()
