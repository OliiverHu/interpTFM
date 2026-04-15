from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import BertForMaskedLM


@dataclass
class GeneformerHandle:
    model: BertForMaskedLM
    model_dir: str
    model_version: str
    device: str
    hidden_size: int
    num_hidden_layers: int


class GeneformerAdapter:
    """
    Minimal Geneformer adapter for V1/V2 hidden-state extraction.
    This is intentionally lightweight and meant to mirror the scGPT-style
    workflow shape in this repo rather than reimplementing all Geneformer tools.
    """

    def load(
        self,
        model_dir: str,
        model_version: str = "V1",
        device: str = "cuda",
    ) -> GeneformerHandle:
        dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        model = BertForMaskedLM.from_pretrained(Path(model_dir)).to(dev)
        model.eval()
        return GeneformerHandle(
            model=model,
            model_dir=str(model_dir),
            model_version=model_version,
            device=str(dev),
            hidden_size=int(model.config.hidden_size),
            num_hidden_layers=int(model.config.num_hidden_layers),
        )

    def list_layers(self, handle: GeneformerHandle) -> List[str]:
        return [f"layer_{i}" for i in range(handle.num_hidden_layers)]

    @torch.no_grad()
    def forward_hidden_states(
        self,
        handle: GeneformerHandle,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        out = handle.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # hidden_states[0] is embeddings; use transformer block outputs as layer_0..layer_{L-1}
        return list(out.hidden_states[1:])
