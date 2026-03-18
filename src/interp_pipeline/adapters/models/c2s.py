from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

import torch
from transformers import PreTrainedModel

from interp_pipeline.adapters.model_base import ModelAdapter, ModelSpec
from interp_pipeline.types.activations import TokenUnit
from interp_pipeline.types.dataset import StandardDataset

from interp_pipeline.c2s_local.load_model import load_c2s_model
from interp_pipeline.c2s_local.tokenizer import C2STokenizer
from interp_pipeline.c2s_local.processor import C2SProcessor


@dataclass
class C2SHandle:
    model: PreTrainedModel
    tokenizer: C2STokenizer
    processor: C2SProcessor
    device: torch.device


class C2SScaleAdapter(ModelAdapter):
    """
    Adapter for C2S-Scale models (GPT-NeoX based, e.g., Pythia).

    C2S converts gene expression to "cell sentences" - gene names ordered by
    descending expression, space-separated. The model then processes these
    as text using a causal language model.

    Key points:
    - model.gpt_neox.layers[i] contains transformer layers
    - Input is tokenized cell sentences (gene names as text)
    - Captures hidden states from transformer layers
    """

    def load(self, spec: ModelSpec) -> C2SHandle:
        device = torch.device(spec.device)
        cache_dir = spec.options.get("cache_dir") if spec.options else None
        max_genes = spec.options.get("max_genes") if spec.options else None

        model, tokenizer = load_c2s_model(
            model_name_or_path=spec.checkpoint,
            cache_dir=cache_dir,
            device=device,
        )
        processor = C2SProcessor(max_genes=max_genes)
        return C2SHandle(model=model, tokenizer=tokenizer, processor=processor, device=device)

    def list_layers(self, model_handle: C2SHandle) -> List[str]:
        """
        Return layer names for C2S model.
        Format: "gpt_neox.layers.{i}" for GPT-NeoX based models.
        """
        n = int(model_handle.model._model.config.num_hidden_layers)
        # Stable naming scheme for pipeline
        return [f"layer_{i}" for i in range(n)]

    def infer_token_unit(self, layer_name: str) -> TokenUnit:
        # currently only initialized with "gene"
        return "gene"

    def make_batches(
        self,
        dataset: StandardDataset,
        model_handle: C2SHandle,
        batch_size: int,
        max_length: int,
        normalize: bool = True,
    ) -> Iterable[Dict[str, Any]]:
        """
        Create batches from dataset by converting expression to cell sentences.

        Required output keys for extractor:
          - cell_ids: list[str] length B
          - input_ids: Tensor[B, T]
          - attention_mask: Tensor[B, T]
        """
        adata = dataset.adata
        n = adata.n_obs

        # Build vocabulary once from the full dataset
        processor = model_handle.processor

        # ----- additional for cosmx_human_lung_sec8.h5ad ----- #
        adata.obs["cell_type"] = adata.obs["author_cell_type"]
        adata.obs["cell_name"] = adata.obs_names
        adata.var_names = adata.var["feature_name"].astype(str).values
        # ----- ----- ----- ----- ----- ----- ----- ----- ----- #

        processor.normalize_adata(adata) if normalize else None
        
        arrow_dataset, _ = processor.adata_to_arrow(adata)
        formatted_hf_ds = processor.prompts_generation(arrow_dataset, n_genes=max_length)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            arrow_dataset_batch = arrow_dataset.select(range(start, end))
            formatted_hf_ds_batch = formatted_hf_ds.select(range(start, end))
            cell_ids = [str(x) for x in adata.obs_names[start:end]]
            batch_input = list(formatted_hf_ds_batch["model_input"])
            tokenized = model_handle.tokenizer(batch_input)

            yield {
                "cell_ids": cell_ids,
                "cell_sentences": list(arrow_dataset_batch["cell_sentence"]),
                "batch_input": batch_input,
                "tokenized": tokenized,
            }

    def forward_and_capture(
        self,
        model_handle: C2SHandle,
        batch: Dict[str, Any],
        layers: Sequence[str],
        capture_cfg: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        """
        Run forward pass and capture hidden states from specified layers.

        For GPT-NeoX models, we iterate through gpt_neox.layers and record
        outputs at requested layer indices.
        """
        model = model_handle.model
        captured: Dict[str, torch.Tensor] = {}
        layer_idxs = [int(lname.split("_")[1]) for lname in layers]
        with torch.no_grad(), model_handle.model.trace(batch["tokenized"]):
            for layer_name, idx in zip(layers, layer_idxs):
                captured[layer_name] = model_handle.model.gpt_neox.layers[idx].output.save()

        return captured
    
    def process_captured(
        self,
        captured: Dict[str, Any],
        batch: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        pass