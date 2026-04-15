from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scanpy as sc
import scipy.sparse as sp
import torch
from datasets import load_from_disk

from interp_pipeline.adapters.models.geneformer import GeneformerAdapter
from interp_pipeline.io.activation_store import ActivationStore, ActivationStoreSpec


def prepare_geneformer_h5ad(
    adata_path: str,
    output_path: str,
    shard_key: str = "shards",
    n_shards: int = 60,
    seed: int = 0,
    keep_existing_shards: bool = True,
) -> str:
    ad = sc.read_h5ad(adata_path).copy()

    if "ensembl_id" not in ad.var.columns:
        ad.var["ensembl_id"] = ad.var_names.astype(str)

    if "cell_id" not in ad.obs.columns:
        ad.obs["cell_id"] = ad.obs_names.astype(str)

    if "n_counts" not in ad.obs.columns:
        X = ad.X
        if sp.issparse(X):
            ad.obs["n_counts"] = np.asarray(X.sum(axis=1)).ravel().astype(np.int64)
        else:
            X = X if isinstance(X, np.ndarray) else np.asarray(X)
            ad.obs["n_counts"] = X.sum(axis=1).astype(np.int64)

    if shard_key not in ad.obs.columns or not keep_existing_shards:
        rng = np.random.default_rng(seed)
        ad.obs[shard_key] = rng.choice([f"shard_{i}" for i in range(n_shards)], size=ad.n_obs)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ad.write(out)
    return str(out)


def tokenize_geneformer_dataset(
    prepared_h5ad_path: str,
    output_dir: str,
    output_prefix: str,
    model_version: str = "V1",
    nproc: int = 1,
    shard_key: str = "shards",
) -> str:
    from geneformer.tokenizer import TranscriptomeTokenizer

    prepared_path = Path(prepared_h5ad_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tk = TranscriptomeTokenizer(
        custom_attr_name_dict={shard_key: shard_key, "cell_id": "cell_id"},
        nproc=nproc,
        model_version=model_version,
        keep_counts=False,
        use_h5ad_index=False,
    )

    tk.tokenize_data(
        data_directory=str(prepared_path.parent),
        output_directory=str(out_dir),
        output_prefix=output_prefix,
        file_format="h5ad",
        input_identifier=prepared_path.stem,
    )

    ds_path = out_dir / f"{output_prefix}.dataset"
    return str(ds_path)


def _load_token_dict(token_dictionary_file: str) -> Dict[int, str]:
    with open(token_dictionary_file, "rb") as f:
        d = pickle.load(f)
    inv = {}
    for gene, tok in d.items():
        try:
            inv[int(tok)] = str(gene)
        except Exception:
            continue
    return inv


def _batched_indices(n: int, batch_size: int) -> Iterable[Tuple[int, int]]:
    for start in range(0, n, batch_size):
        yield start, min(start + batch_size, n)


@torch.no_grad()
def extract_geneformer_to_store(
    model_dir: str,
    tokenized_dataset_path: str,
    store_root: str,
    layers: Sequence[str],
    model_version: str = "V1",
    device: str = "cuda",
    forward_batch_size: int = 8,
    token_dictionary_file: Optional[str] = None,
    shard_key: str = "shards",
    save_dtype: str = "fp16",
) -> List[str]:
    adapter = GeneformerAdapter()
    handle = adapter.load(model_dir=model_dir, model_version=model_version, device=device)

    available_layers = set(adapter.list_layers(handle))
    layers = [layer for layer in layers if layer in available_layers]
    if not layers:
        raise ValueError(f"No valid layers requested. Available: {sorted(available_layers)}")

    ds = load_from_disk(str(tokenized_dataset_path))
    if shard_key not in ds.column_names:
        raise ValueError(f"Tokenized dataset missing required shard column: {shard_key}")

    if token_dictionary_file is None:
        from geneformer import TOKEN_DICTIONARY_FILE, TOKEN_DICTIONARY_FILE_30M
        token_dictionary_file = str(
            TOKEN_DICTIONARY_FILE_30M if model_version.upper() == "V1" else TOKEN_DICTIONARY_FILE
        )
    tok2gene = _load_token_dict(token_dictionary_file)

    store = ActivationStore(ActivationStoreSpec(root=store_root))
    dtype = torch.float16 if save_dtype == "fp16" else torch.float32

    # Build shard -> indices once
    print("[extract] building shard index map...")
    all_shards = ds[shard_key]
    shard_to_indices: Dict[str, List[int]] = {}
    for i, s in enumerate(all_shards):
        shard_to_indices.setdefault(str(s), []).append(i)

    unique_shards = sorted(shard_to_indices.keys())
    print(f"[extract] found {len(unique_shards)} shards")

    for shard_name in unique_shards:
        indices = shard_to_indices[shard_name]
        print(f"[extract] {shard_name}: {len(indices)} rows")

        shard_data = ds.select(indices)
        shard_id = int(str(shard_name).split("_")[-1])

        layer_acts: Dict[str, List[torch.Tensor]] = {layer: [] for layer in layers}
        example_ids: List[str] = []
        token_ids: List[str] = []

        n = len(shard_data)
        for start, end in _batched_indices(n, forward_batch_size):
            batch = shard_data.select(range(start, end))
            input_ids_list = batch["input_ids"]
            cell_ids = (
                batch["cell_id"]
                if "cell_id" in batch.column_names
                else [f"{shard_name}_row_{i}" for i in range(start, end)]
            )

            max_len = max(len(x) for x in input_ids_list)
            input_ids = torch.zeros((len(input_ids_list), max_len), dtype=torch.long, device=handle.device)
            attention_mask = torch.zeros_like(input_ids)

            for i, ids in enumerate(input_ids_list):
                ids = [int(t) for t in ids]
                input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=handle.device)
                attention_mask[i, : len(ids)] = 1

            hidden_states = adapter.forward_hidden_states(
                handle,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            for bi, ids in enumerate(input_ids_list):
                ids = [int(t) for t in ids]
                cid = str(cell_ids[bi])
                valid_len = len(ids)

                example_ids.extend([cid] * valid_len)
                token_ids.extend([tok2gene.get(int(t), str(t)) for t in ids])

                for layer in layers:
                    layer_idx = int(layer.split("_")[-1])
                    hs = hidden_states[layer_idx][bi, :valid_len, :].detach().to("cpu", dtype=dtype)
                    layer_acts[layer].append(hs)

        for layer in layers:
            acts = torch.cat(layer_acts[layer], dim=0)
            store.write_token_shard(
                layer=layer,
                shard_id=shard_id,
                acts=acts,
                example_ids=example_ids,
                token_ids=token_ids,
                token_unit="gene",
                meta={
                    "source_format": "geneformer_token_activations",
                    "model_version": model_version,
                    "model_dir": model_dir,
                    "tokenized_dataset_path": tokenized_dataset_path,
                    "shard_name": shard_name,
                },
            )

    return list(layers)