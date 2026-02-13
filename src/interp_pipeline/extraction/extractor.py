from __future__ import annotations
from typing import Any, Dict, List, Sequence, Tuple

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
    Generic extractor:
      - iterate batches from adapter
      - capture activations per requested layer
      - flatten token-level acts -> [N_tokens, H]
      - write per-batch as a shard (simple, safe, resumable)
    """

    batch_size = int(extraction_cfg.get("batch_size", 16))
    max_length = int(extraction_cfg.get("max_length", 1200))
    shard_id = int(extraction_cfg.get("start_shard", 0))

    extracted_layers: List[str] = []
    # We’ll fill a “global” index later; for now return a minimal index stub.
    # Downstream SAE training reads from store anyway.
    global_example_ids: List[str] = []
    global_token_ids: List[str] = []

    for batch in model_adapter.make_batches(dataset, model_handle, batch_size=batch_size, max_length=max_length):
        captured = model_adapter.forward_and_capture(
            model_handle=model_handle,
            batch=batch,
            layers=layers,
            capture_cfg=extraction_cfg,
        )
        cell_ids = batch["cell_ids"]  # list[str], length B

        for layer_name, acts_btH in captured.items():
            token_unit = model_adapter.infer_token_unit(layer_name)

            if acts_btH.ndim != 3:
                raise ValueError(f"{layer_name}: expected [B,T,H] for token-level capture, got shape={acts_btH.shape}")

            B, T, H = acts_btH.shape
            flat = acts_btH.reshape(B * T, H).contiguous()

            # token_ids: best-effort mapping for scGPT
            # If collator produced gene_ids [B,T], we use those; otherwise we store position IDs.
            tok = None
            if "gene_ids" in batch:
                tok = batch["gene_ids"]  # [B,T] on device
                if isinstance(tok, torch.Tensor):
                    tok = tok.to("cpu").reshape(B * T).tolist()
            if tok is None:
                tok = list(range(B * T))

            # Expand example ids to token occurrences
            ex = []
            for cid in cell_ids:
                ex.extend([cid] * T)

            # Store shard per (batch, layer)
            store.write_token_shard(
                layer=layer_name,
                shard_id=shard_id,
                acts=flat,
                example_ids=[str(x) for x in ex],
                token_ids=[str(x) for x in tok],
                token_unit=token_unit,
                meta={"model": extraction_cfg.get("model_name", "unknown")},
            )

            if layer_name not in extracted_layers:
                extracted_layers.append(layer_name)

        shard_id += 1

    index = ActivationIndex(
        example_ids=global_example_ids,
        token_ids=global_token_ids if global_token_ids else None,
        token_unit="other",
        layer_names=list(extracted_layers),
    )
    return index, extracted_layers
