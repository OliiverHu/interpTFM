from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any

import torch
from tqdm import tqdm

from interp_pipeline.adapters.models.c2s_scale import C2SScaleAdapter
from interp_pipeline.adapters.model_base import ModelSpec
from interp_pipeline.types.dataset import StandardDataset


def save_processed_batch_c2s(
    processed: Dict[str, Dict[str, Any]],
    output_dir: Path,
    shard: int,
    batch_idx: int,
) -> None:
    """
    Save one processed C2S batch.

    Per layer:
      - gene-level activations + mapping
      - cell-level activations + mapping
    """
    for layer_name, out in processed.items():
        layer_num = int(layer_name.split("_")[1])

        # gene-level
        gene_out_dir = output_dir / "activations" / f"layer_{layer_num}" / f"shard_{shard}"
        gene_out_dir.mkdir(parents=True, exist_ok=True)

        torch.save(out["acts"], gene_out_dir / f"batch_{batch_idx:05d}_gene_acts.pt")

        with open(gene_out_dir / f"batch_{batch_idx:05d}_cell_gene_pairs.txt", "w") as f:
            for cell_id, gene in zip(out["ex"], out["tok"]):
                f.write(f"{cell_id}\t{gene}\n")

        # cell-level
        if "cell_acts" in out and "cell_ids" in out:
            cell_out_dir = output_dir / "cell_activations" / f"layer_{layer_num}" / f"shard_{shard}"
            cell_out_dir.mkdir(parents=True, exist_ok=True)

            torch.save(out["cell_acts"], cell_out_dir / f"batch_{batch_idx:05d}_cell_acts.pt")

            with open(cell_out_dir / f"batch_{batch_idx:05d}_cell_ids.txt", "w") as f:
                for cell_id in out["cell_ids"]:
                    f.write(f"{cell_id}\n")


def extract_c2s_shard(
    adata,
    model_path: str,
    output_dir: str,
    layers: List[str],
    shard: int,
    batch_size: int = 8,
    max_genes: int = 256,
    device: str = "cuda:0",
    pooling: str = "last",
    save_dtype: str = "fp16",
    pool_dtype: str = "fp32",
    normalize: bool = True,
    cache_dir: str | None = None,
) -> None:
    """
    Run C2S extraction for one shard AnnData subset.
    """
    output_path = Path(output_dir)
    dataset = StandardDataset(adata=adata, obs_key_map={})

    spec = ModelSpec(
        name="c2s-scale",
        checkpoint=model_path,
        device=device,
        options={
            "max_genes": max_genes,
            "cache_dir": cache_dir,
        },
    )

    adapter = C2SScaleAdapter()
    handle = adapter.load(spec)

    (output_path / "metadata").mkdir(parents=True, exist_ok=True)
    with open(output_path / "metadata" / f"shard_{shard}.json", "w") as f:
        json.dump(
            {
                "model": model_path,
                "layers": layers,
                "shard": shard,
                "batch_size": batch_size,
                "max_genes": max_genes,
                "pooling": pooling,
                "save_dtype": save_dtype,
                "pool_dtype": pool_dtype,
                "normalize": normalize,
            },
            f,
            indent=2,
        )

    batch_iter = adapter.make_batches(
        dataset=dataset,
        model_handle=handle,
        batch_size=batch_size,
        max_genes=max_genes,
        normalize=normalize,
    )

    for batch_idx, batch in enumerate(tqdm(batch_iter, desc=f"Shard {shard} extracting")):
        batch["pooling"] = pooling
        batch["save_dtype"] = save_dtype
        batch["pool_dtype"] = pool_dtype

        captured = adapter.forward_and_capture(
            model_handle=handle,
            batch=batch,
            layers=layers,
            capture_cfg={},
        )

        processed = adapter.process_captured(
            captured=captured,
            batch=batch,
        )

        save_processed_batch_c2s(
            processed=processed,
            output_dir=output_path,
            shard=shard,
            batch_idx=batch_idx,
        )


def extract_c2s_dataset(
    adata,
    model_path: str,
    output_dir: str,
    layers: List[str],
    shards: int = 60,
    shard_key: str = "shards",
    batch_size: int = 8,
    max_genes: int = 256,
    device: str = "cuda:0",
    pooling: str = "last",
    save_dtype: str = "fp16",
    pool_dtype: str = "fp32",
    normalize: bool = True,
    cache_dir: str | None = None,
) -> None:
    """
    Run C2S extraction over all shard subsets in an AnnData.
    """
    print(f"Loaded adata: {adata.n_obs} cells, {adata.n_vars} genes")
    print(f"Using layers: {layers}")
    print(
        f"shards={shards}, batch_size={batch_size}, max_genes={max_genes}, "
        f"pooling={pooling}, save_dtype={save_dtype}, pool_dtype={pool_dtype}"
    )

    for s in range(shards):
        subset = adata[adata.obs[shard_key] == f"shard_{s}"].copy()
        print(f"\nProcessing shard_{s}: {subset.n_obs} cells")

        if subset.n_obs == 0:
            continue

        extract_c2s_shard(
            adata=subset,
            model_path=model_path,
            output_dir=output_dir,
            layers=layers,
            shard=s,
            batch_size=batch_size,
            max_genes=max_genes,
            device=device,
            pooling=pooling,
            save_dtype=save_dtype,
            pool_dtype=pool_dtype,
            normalize=normalize,
            cache_dir=cache_dir,
        )