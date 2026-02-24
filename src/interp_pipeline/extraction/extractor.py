from __future__ import annotations
from typing import Any, Dict, List, Sequence, Tuple, Optional

import math
import torch

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

    # If user specifies it, we use it. Otherwise we compute it from first batch.
    target_tokens_per_shard: Optional[int] = extraction_cfg.get("target_tokens_per_shard", None)
    if target_tokens_per_shard is not None:
        target_tokens_per_shard = int(target_tokens_per_shard)

    extracted_layers: List[str] = []

    # Buffers per layer: accumulate until we flush a shard.
    buf = {
        layer: {"acts": [], "ex": [], "tok": [], "n": 0, "token_unit": None}
        for layer in layers
    }

    # Minimal index stub (SAE reads from store)
    global_example_ids: List[str] = []
    global_token_ids: List[str] = []

    # Helper: get token ids from batch (supports scgpt_local and older scGPT)
    def _get_token_ids(batch: Dict[str, Any], B: int, T: int) -> List[str]:
        tok = None
        if "genes" in batch:
            tok = batch["genes"]  # [B,T]
        elif "gene_ids" in batch:
            tok = batch["gene_ids"]  # [B,T]

        if isinstance(tok, torch.Tensor):
            tok = tok.to("cpu").reshape(B * T).tolist()

        if tok is None:
            tok = list(range(B * T))

        return [str(x) for x in tok]

    # Helper: flush a layer buffer as one shard
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

        # reset
        b["acts"].clear()
        b["ex"].clear()
        b["tok"].clear()
        b["n"] = 0

    # Compute auto budget from first batch
    computed_budget = False

    for batch_idx, batch in enumerate(
        model_adapter.make_batches(dataset, model_handle, batch_size=batch_size, max_length=max_length)
    ):
        captured = model_adapter.forward_and_capture(
            model_handle=model_handle,
            batch=batch,
            layers=layers,
            capture_cfg=extraction_cfg,
        )
        cell_ids = batch["cell_ids"]  # list[str], length B

        # Some adapters might return empty dict if capture fails; make that obvious.
        if not captured:
            raise RuntimeError(
                "forward_and_capture returned no activations. "
                "This usually means the capture mechanism isn't working for this model."
            )

        # Determine budget once we see first tensor shape (B,T,H) for any layer.
        if (not computed_budget) and (target_tokens_per_shard is None):
            # pick first captured layer
            any_layer = next(iter(captured))
            acts_btH = captured[any_layer]
            if acts_btH.ndim != 3:
                raise ValueError(f"{any_layer}: expected [B,T,H], got shape={acts_btH.shape}")
            B0, T0, _ = acts_btH.shape

            # estimate total tokens = n_obs * T0
            n_obs = int(getattr(dataset.adata, "n_obs", len(dataset.adata.obs_names)))
            total_tokens_est = n_obs * T0

            # budget so that <= max_shards
            # +1 guard to avoid tiny budgets; also avoid zero
            denom = max(1, int(max_shards) if max_shards is not None else 512)
            target_tokens_per_shard = max(1, int(math.ceil(total_tokens_est / denom)))

            computed_budget = True
            # print a small hint once
            print(f"[extractor] auto target_tokens_per_shard={target_tokens_per_shard} (T≈{T0}, n_obs={n_obs}, max_shards={max_shards})")

        for layer_name, acts_btH in captured.items():
            token_unit = model_adapter.infer_token_unit(layer_name)

            if acts_btH.ndim != 3:
                raise ValueError(f"{layer_name}: expected [B,T,H] for token-level capture, got shape={acts_btH.shape}")

            B, T, H = acts_btH.shape
            flat = acts_btH.reshape(B * T, H).contiguous()

            tok_list = _get_token_ids(batch, B, T)

            # Expand example ids to token occurrences
            ex_list: List[str] = []
            for cid in cell_ids:
                ex_list.extend([str(cid)] * T)

            b = buf[layer_name]
            if b["token_unit"] is None:
                b["token_unit"] = token_unit

            b["acts"].append(flat)
            b["tok"].extend(tok_list)
            b["ex"].extend(ex_list)
            b["n"] += flat.shape[0]

            if layer_name not in extracted_layers:
                extracted_layers.append(layer_name)

            # Flush if we reached budget
            if target_tokens_per_shard is not None and b["n"] >= target_tokens_per_shard:
                _flush_layer(layer_name)

                # stop early if max_shards hit
                if max_shards is not None and shard_id >= max_shards:
                    # flush other layers too (optional: keep consistent)
                    for ln in layers:
                        _flush_layer(ln)
                    index = ActivationIndex(
                        example_ids=global_example_ids,
                        token_ids=global_token_ids if global_token_ids else None,
                        token_unit="other",
                        layer_names=list(extracted_layers),
                    )
                    return index, extracted_layers

    # Final flush leftovers
    for ln in layers:
        _flush_layer(ln)

    index = ActivationIndex(
        example_ids=global_example_ids,
        token_ids=global_token_ids if global_token_ids else None,
        token_unit="other",
        layer_names=list(extracted_layers),
    )
    return index, extracted_layers
