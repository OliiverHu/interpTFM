from __future__ import annotations
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch

from interp_pipeline.types.activations import ActivationIndex, TokenUnit


@dataclass(frozen=True)
class ActivationStoreSpec:
    root: str
    backend: str = "pt_shards"  # keep it simple first
    shard_prefix: str = "shard_"


class ActivationStore:
    """
    Very simple shard-based store:

      root/
        activations/<layer_name>/shard_0/activations.pt
        activations/<layer_name>/shard_0/index.pt
        activations/<layer_name>/shard_0/metadata.json
        activations/<layer_name>/shard_1/...

    Where index.pt is a dict with:
      {"example_ids": [...], "token_ids": [...] or None, "token_unit": "..."}
    """
    def __init__(self, spec: ActivationStoreSpec):
        self.spec = spec
        os.makedirs(self.spec.root, exist_ok=True)

    def _layer_dir(self, layer: str) -> str:
        return os.path.join(self.spec.root, "activations", layer)

    def write_token_shard(
        self,
        layer: str,
        shard_id: int,
        acts: torch.Tensor,                 # [N_tokens, H]
        example_ids: List[str],             # length N_tokens
        token_ids: List[str],               # length N_tokens
        token_unit: TokenUnit,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        out_dir = os.path.join(self._layer_dir(layer), f"{self.spec.shard_prefix}{shard_id}")
        os.makedirs(out_dir, exist_ok=True)

        torch.save(acts, os.path.join(out_dir, "activations.pt"))
        torch.save(
            {"example_ids": example_ids, "token_ids": token_ids, "token_unit": token_unit},
            os.path.join(out_dir, "index.pt"),
        )

        metadata = meta or {}
        metadata.update(
            {
                "layer": layer,
                "shard": shard_id,
                "token_unit": token_unit,
                "n_rows": int(acts.shape[0]),
                "hidden_dim": int(acts.shape[1]),
                "dtype": str(acts.dtype),
            }
        )
        with open(os.path.join(out_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    def list_shards(self, layer: str) -> List[str]:
        layer_dir = self._layer_dir(layer)
        if not os.path.isdir(layer_dir):
            return []
        shards = []
        for name in sorted(os.listdir(layer_dir)):
            if name.startswith(self.spec.shard_prefix):
                shards.append(os.path.join(layer_dir, name))
        return shards

    def iter_token_batches(
        self,
        layer: str,
        batch_size: int,
        shuffle_shards: bool = False,
    ) -> Iterator[Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Yields (acts_batch, index_batch).
        index_batch contains:
          - example_ids: list[str]
          - token_ids: list[str]
          - token_unit: str
        """
        shards = self.list_shards(layer)
        if shuffle_shards:
            import random
            random.shuffle(shards)

        for shard_dir in shards:
            acts = torch.load(os.path.join(shard_dir, "activations.pt"), map_location="cpu")
            idx = torch.load(os.path.join(shard_dir, "index.pt"))

            n = acts.shape[0]
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch_acts = acts[start:end]
                batch_index = {
                    "example_ids": idx["example_ids"][start:end],
                    "token_ids": idx["token_ids"][start:end] if idx["token_ids"] is not None else None,
                    "token_unit": idx["token_unit"],
                }
                yield batch_acts, batch_index
