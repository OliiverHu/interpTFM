#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from interp_pipeline.sae.normalize import normalize_sae_checkpoint


def resolve_input_ckpt(run_dir: Path, layer: str, ckpt_mode: str, ckpt_name: str | None) -> Path:
    if ckpt_name is not None:
        ckpt = run_dir / ckpt_name
    elif ckpt_mode == "best":
        ckpt = run_dir / f"sae_{layer}_best.pt"
    elif ckpt_mode == "last":
        ckpt = run_dir / f"sae_{layer}_last.pt"
    else:
        raise ValueError(f"Unsupported ckpt_mode: {ckpt_mode}")
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
    return ckpt


def resolve_output_ckpt(input_ckpt: Path, explicit_output_name: str | None) -> Path:
    if explicit_output_name is not None:
        return input_ckpt.parent / explicit_output_name
    return input_ckpt.with_name(f"{input_ckpt.stem}_normalized.pt")


def main():
    ap = argparse.ArgumentParser(description="Batch-normalize SAE checkpoints across 3-model sweep runs.")
    ap.add_argument("--labels", nargs=3, required=True, help="Model labels, e.g. scgpt c2sscale geneformer")
    ap.add_argument("--sae-base-dirs", nargs=3, required=True, help="Base SAE dirs for the 3 models")
    ap.add_argument("--store-roots", nargs=3, required=True, help="ActivationStore roots for the 3 models")
    ap.add_argument("--layers", nargs=3, required=True, help="Layer name for each model")
    ap.add_argument("--run-tags", nargs="+", required=True, help="Sweep run tags")
    ap.add_argument("--ckpt-mode", choices=["best", "last"], default="best")
    ap.add_argument("--ckpt-name", default=None, help="Optional explicit checkpoint filename inside each run dir")
    ap.add_argument("--output-name", default=None, help="Optional explicit output filename inside each run dir")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--token-chunk-size", type=int, default=4096)
    ap.add_argument("--feature-chunk-size", type=int, default=1024)
    args = ap.parse_args()

    for run_tag in args.run_tags:
        print("\n" + "=" * 100)
        print(f"RUN TAG: {run_tag}")
        print("=" * 100)

        for label, base_dir, store_root, layer in zip(args.labels, args.sae_base_dirs, args.store_roots, args.layers):
            run_dir = Path(base_dir) / run_tag
            input_ckpt = resolve_input_ckpt(run_dir, layer, args.ckpt_mode, args.ckpt_name)
            output_ckpt = resolve_output_ckpt(input_ckpt, args.output_name)

            print(f"[normalize] model={label} layer={layer}")
            print(f"  input : {input_ckpt}")
            print(f"  output: {output_ckpt}")

            norm_path = normalize_sae_checkpoint(
                ckpt_path=str(input_ckpt),
                store_root=str(store_root),
                layer=layer,
                output_path=str(output_ckpt),
                device=args.device,
                max_shards=args.max_shards,
                token_chunk_size=args.token_chunk_size,
                feature_chunk_size=args.feature_chunk_size,
            )

            print(f"  saved : {norm_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()


# python test_normalize_sae_latents.py \
#   --labels scgpt c2sscale geneformer \
#   --sae-base-dirs \
#     runs/full_scgpt_cosmx/sae/layer_4.norm2 \
#     runs/full_c2sscale_cosmx/sae/layer_17 \
#     runs/full_geneformer_cosmx/sae/layer_4 \
#   --store-roots \
#     runs/full_scgpt_cosmx \
#     runs/full_c2sscale_cosmx \
#     runs/full_geneformer_cosmx \
#   --layers layer_4.norm2 layer_17 layer_4 \
#   --run-tags \
#     nr_on__steps_6000__l1_1e-3 \
#     nr_on__steps_6000__l1_3e-3 \
#     nr_on__steps_6000__l1_0p01 \
#   --ckpt-mode best \
#   --device cuda