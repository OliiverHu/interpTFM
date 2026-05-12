from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple

import torch
from tqdm import tqdm

from interp_pipeline.adapters.models.c2s_scale import C2SScaleAdapter
from interp_pipeline.adapters.model_base import ModelSpec
from interp_pipeline.io.activation_store import ActivationStore, ActivationStoreSpec
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

    for batch_idx, batch in enumerate(
        tqdm(batch_iter, desc=f"Shard {shard} batches", position=1, leave=False)
    ):
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

    shard_names = [f"shard_{s}" for s in range(shards)]

    for s_name in tqdm(shard_names, desc="Shards", position=0):
        s = int(s_name.split("_")[1])
        subset = adata[adata.obs[shard_key] == s_name].copy()

        tqdm.write(f"Processing {s_name}: {subset.n_obs} cells")

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


def _read_pairs_file(path: str) -> Tuple[List[str], List[str]]:
    example_ids: List[str] = []
    token_ids: List[str] = []
    with open(path, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise RuntimeError(f"Malformed cell/gene pair line in {path!r}: {line!r}")
            cell_id, gene = parts
            example_ids.append(cell_id)
            token_ids.append(gene)
    return example_ids, token_ids


def convert_c2s_layer_to_activation_store(
    c2s_root: str,
    out_root: str,
    layer: str,
    overwrite: bool = False,
) -> int:
    """
    Convert C2S raw extraction output for one layer into a generic ActivationStore.

    C2S raw format (from extract_c2s_shard / extract_c2s_dataset):
      c2s_root/activations/{layer}/shard_{i}/batch_XXXXX_gene_acts.pt
      c2s_root/activations/{layer}/shard_{i}/batch_XXXXX_cell_gene_pairs.txt

    Generic ActivationStore format written to out_root:
      out_root/activations/{layer}/shard_{i}/activations.pt
      out_root/activations/{layer}/shard_{i}/index.pt   (token_ids = gene name strings)

    Returns the number of shards written.
    """
    layer_dir = os.path.join(c2s_root, "activations", layer)
    if not os.path.isdir(layer_dir):
        raise FileNotFoundError(f"Missing extracted c2s layer directory: {layer_dir}")

    store = ActivationStore(ActivationStoreSpec(root=out_root))
    shard_dirs = sorted(
        [p for p in glob.glob(os.path.join(layer_dir, "shard_*")) if os.path.isdir(p)]
    )
    if not shard_dirs:
        raise RuntimeError(f"No c2s shard dirs found for {layer} under {layer_dir}")

    written = 0
    for shard_dir in tqdm(shard_dirs, desc=f"convert:{layer}"):
        shard_name = os.path.basename(shard_dir)
        shard_id = int(shard_name.split("_")[-1])
        generic_dir = os.path.join(out_root, "activations", layer, shard_name)
        acts_out = os.path.join(generic_dir, "activations.pt")
        idx_out = os.path.join(generic_dir, "index.pt")
        if (not overwrite) and os.path.exists(acts_out) and os.path.exists(idx_out):
            continue

        act_files = sorted(glob.glob(os.path.join(shard_dir, "batch_*_gene_acts.pt")))
        if not act_files:
            continue

        acts_parts: List[torch.Tensor] = []
        example_ids: List[str] = []
        token_ids: List[str] = []
        n_source_batches = 0

        for act_path in act_files:
            stem = os.path.basename(act_path).replace("_gene_acts.pt", "")
            pair_path = os.path.join(shard_dir, f"{stem}_cell_gene_pairs.txt")
            if not os.path.exists(pair_path):
                raise FileNotFoundError(f"Missing pair file for {act_path}: {pair_path}")

            acts = torch.load(act_path, map_location="cpu")
            if not isinstance(acts, torch.Tensor) or acts.ndim != 2:
                raise RuntimeError(
                    f"Expected 2D tensor in {act_path}, got {type(acts)} shape={getattr(acts, 'shape', None)}"
                )
            ex_ids, tok_ids = _read_pairs_file(pair_path)
            if len(ex_ids) != acts.shape[0]:
                raise RuntimeError(
                    f"Row mismatch in {act_path}: acts_rows={acts.shape[0]} pairs={len(ex_ids)}"
                )
            acts_parts.append(acts)
            example_ids.extend(ex_ids)
            token_ids.extend(tok_ids)
            n_source_batches += 1

        if not acts_parts:
            continue

        acts_cat = torch.cat(acts_parts, dim=0)
        store.write_token_shard(
            layer=layer,
            shard_id=shard_id,
            acts=acts_cat,
            example_ids=example_ids,
            token_ids=token_ids,
            token_unit="gene",
            meta={
                "source_format": "c2s_gene_batches",
                "source_layer": layer,
                "source_batches": n_source_batches,
                "source_root": c2s_root,
            },
        )
        written += 1
    return written