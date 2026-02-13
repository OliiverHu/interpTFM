from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

from interp_pipeline.types.activations import TokenUnit
from interp_pipeline.types.dataset import StandardDataset

@dataclass(frozen=True)
class ModelSpec:
    name: str
    checkpoint: str
    device: str = "cuda"
    # model-specific options can go into this dict
    options: Optional[Dict[str, Any]] = None

class ModelAdapter(ABC):
    @abstractmethod
    def load(self, spec: ModelSpec) -> Any:
        ...

    @abstractmethod
    def list_layers(self, model_handle: Any) -> List[str]:
        ...

    @abstractmethod
    def infer_token_unit(self, layer_name: str) -> TokenUnit:
        ...

    @abstractmethod
    def make_batches(
        self,
        dataset: StandardDataset,
        model_handle: Any,
        batch_size: int,
        max_length: int,
    ) -> Iterable[Dict[str, Any]]:
        """
        Yield batches ready for forward pass. Each batch MUST include:
          - "cell_ids": list[str] length B
        It may include other fields as needed by forward pass.
        """
        ...

    @abstractmethod
    def forward_and_capture(
        self,
        model_handle: Any,
        batch: Dict[str, Any],
        layers: Sequence[str],
        capture_cfg: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Return activations per layer for this batch.
        For token-level layers: Tensor[B, T, H]
        For cell-level layers:  Tensor[B, H]
        """
        ...
