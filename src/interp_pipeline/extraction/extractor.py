from __future__ import annotations
from typing import Any, Dict, List, Sequence, Tuple, Optional

import math
import torch
from tqdm import tqdm
from interp_pipeline.adapters.model_base import ModelAdapter
from interp_pipeline.io.activation_store import ActivationStore
from interp_pipeline.types.activations import ActivationIndex
from interp_pipeline.types.dataset import StandardDataset


def extract_activations(
    dataset: StandardDataset,
    model_handle: Any,
    model_adapter: ModelAdapter,
    layers: Sequence[str],
    store: ActivationStore,
    extraction_cfg: Dict[str, Any],
) -> Tuple[ActivationIndex, List[str]]:
    """
    Generic extractor with bounded shard count.

    Behavior:
      - iterate batches from adapter
      - capture activations per requested layer (token-level: [B,T,H])
      - flatten -> [B*T, H]
      - ACCUMULATE into per-layer buffers
      - flush buffers to disk when token budget reached
      - aim for <= max_shards (default 512) by auto-computing token budget from first batch

    Config keys:
      - batch_size: int
      - max_length: int
      - start_shard: int
      - model_name: str
      - max_shards: int (default 512)
      - target_tokens_per_shard: Optional[int] (if provided, overrides auto budget)
    """

    batch_size = int(extraction_cfg.get("batch_size", 16))
    max_length = int(extraction_cfg.get("max_length", 1200))
    shard_id = int(extraction_cfg.get("start_shard", 0))

    max_shards = extraction_cfg.get("max_shards", 512)
    if max_shards is not None:
        max_shards = int(max_shards)

    target_tokens_per_shard: Optional[int] = extraction_cfg.get("target_tokens_per_shard", None)
    if target_tokens_per_shard is not None:
        target_tokens_per_shard = int(target_tokens_per_shard)

    extracted_layers: List[str] = []

    buf = {
        layer: {"acts": [], "ex": [], "tok": [], "n": 0, "token_unit": None}
        for layer in layers
    }

    global_example_ids: List[str] = []
    global_token_ids: List[str] = []

    def _flush_layer(layer_name: str) -> None:
        nonlocal shard_id
        b = buf[layer_name]
        if b["n"] == 0:
            return

        acts_cat = torch.cat(b["acts"], dim=0)
        store.write_token_shard(
            layer=layer_name,
            shard_id=shard_id,
            acts=acts_cat,
            example_ids=[str(x) for x in b["ex"]],
            token_ids=[str(x) for x in b["tok"]],
            token_unit=b["token_unit"] if b["token_unit"] is not None else model_adapter.infer_token_unit(layer_name),
            meta={"model": extraction_cfg.get("model_name", "unknown")},
        )
        shard_id += 1

        b["acts"].clear()
        b["ex"].clear()
        b["tok"].clear()
        b["n"] = 0

    computed_budget = False

    batch_iter = model_adapter.make_batches(
        dataset,
        model_handle,
        batch_size=batch_size,
        max_length=max_length,
    )

    n_obs = int(getattr(dataset.adata, "n_obs", len(dataset.adata.obs_names)))
    total_batches = max(1, math.ceil(n_obs / batch_size))
    desc = f"extract:{layers[0]}" if len(layers) == 1 else f"extract:{len(layers)}layers"

    for batch_idx, batch in enumerate(
        tqdm(batch_iter, total=total_batches, desc=desc)
    ):
        captured = model_adapter.forward_and_capture(
            model_handle=model_handle,
            batch=batch,
            layers=layers,
            capture_cfg=extraction_cfg,
        )

        if not captured:
            raise RuntimeError(
                "forward_and_capture returned no activations. "
                "This usually means the capture mechanism isn't working for this model."
            )

        buf_entries = model_adapter.process_captured(captured, batch)

        if (not computed_budget) and (target_tokens_per_shard is None):
            any_entry = buf_entries[next(iter(buf_entries))]
            T_list = any_entry["T_list"]
            T0 = sum(T_list) / max(1, len(T_list))

            total_tokens_est = n_obs * T0
            denom = max(1, int(max_shards) if max_shards is not None else 512)
            target_tokens_per_shard = max(1, int(math.ceil(total_tokens_est / denom)))

            computed_budget = True
            print(
                f"[extractor] auto target_tokens_per_shard={target_tokens_per_shard} "
                f"(T≈{T0:.1f}, n_obs={n_obs}, max_shards={max_shards})"
            )

        for layer_name, entry in buf_entries.items():
            b = buf[layer_name]
            if b["token_unit"] is None:
                b["token_unit"] = entry["token_unit"]

            b["acts"].append(entry["acts"])
            b["tok"].extend(entry["tok"])
            b["ex"].extend(entry["ex"])
            b["n"] += entry["acts"].shape[0]

            if layer_name not in extracted_layers:
                extracted_layers.append(layer_name)

            if target_tokens_per_shard is not None and b["n"] >= target_tokens_per_shard:
                _flush_layer(layer_name)

                if max_shards is not None and shard_id >= max_shards:
                    for ln in layers:
                        _flush_layer(ln)
                    index = ActivationIndex(
                        example_ids=global_example_ids,
                        token_ids=global_token_ids if global_token_ids else None,
                        token_unit="other",
                        layer_names=list(extracted_layers),
                    )
                    return index, extracted_layers

    for ln in layers:
        _flush_layer(ln)

    index = ActivationIndex(
        example_ids=global_example_ids,
        token_ids=global_token_ids if global_token_ids else None,
        token_unit="other",
        layer_names=list(extracted_layers),
    )
    return index, extracted_layers