from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch

from interp_pipeline.adapters.model_base import ModelAdapter, ModelSpec
from interp_pipeline.types.activations import TokenUnit
from interp_pipeline.types.dataset import StandardDataset

# Use the vendored scGPT implementation (your clean local version)
from interp_pipeline.scgpt_local.load_model_from_pretrain import (
    create_clean_model_from_pretrain,
)


@dataclass
class ScGPTHandle:
    model: torch.nn.Module
    tokenizer: Any  # interp_pipeline.third_party.scgpt_local.tokenizer.Tokenizer
    device: torch.device


class ScGPTAdapter(ModelAdapter):
    """
    Adapter around your local scGPT clean implementation.

    Key points:
    - model(src, values, src_key_padding_mask) returns token-level hidden states [B,T,H]
    - we capture internal activations via forward hooks on transformer_encoder.layers[i].norm2
      (same behavior as your old extraction script)
    """

    def load(self, spec: ModelSpec) -> ScGPTHandle:
        device = torch.device(spec.device)
        model, tokenizer = create_clean_model_from_pretrain(spec.checkpoint, device=device)
        return ScGPTHandle(model=model, tokenizer=tokenizer, device=device)

    def list_layers(self, model_handle: ScGPTHandle) -> List[str]:
        # Your scGPTConfig stores nlayers in model.config.nlayers
        n = int(model_handle.model.config.nlayers)
        # Stable naming scheme for pipeline
        return [f"layer_{i}.norm2" for i in range(n)]

    def infer_token_unit(self, layer_name: str) -> TokenUnit:
        # At norm2, output is token-level (gene tokens + padding possibly)
        if layer_name.endswith(".norm2"):
            return "gene"
        return "other"

    def make_batches(
        self,
        dataset: StandardDataset,
        model_handle: ScGPTHandle,
        batch_size: int,
        max_length: int,
    ) -> Iterable[Dict[str, Any]]:
        """
        Batches are built by running tokenizer on chunks of adata.X.

        Required output keys for extractor:
          - cell_ids: list[str] length B
          - genes: Tensor[B,T]
          - expressions: Tensor[B,T]
          - src_key_padding_mask: Tensor[B,T] bool
        """
        adata = dataset.adata

        # Choose gene symbol column for tokenizer vocab mapping.
        # Your dataset seems to have 'feature_name' in .var (from your printout).
        gene_key = "feature_name"
        if gene_key not in adata.var.columns:
            # fallback; adjust if your var uses different naming
            if "index" in adata.var.columns:
                gene_key = "index"
            else:
                raise ValueError("AnnData .var must include a gene symbol column like 'feature_name' or 'index'")

        gene_names = adata.var[gene_key].to_numpy()

        X = adata.X
        # tokenizer supports np.ndarray or torch.Tensor; we’ll feed numpy float32
        if not isinstance(X, np.ndarray):
            X = X.toarray()
        X = X.astype(np.float32)

        n = adata.n_obs
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            cell_ids = [str(x) for x in adata.obs_names[start:end]]

            # Tokenize: returns (genes, expressions, src_key_padding_mask)
            genes, expressions, src_key_padding_mask = model_handle.tokenizer(
                X[start:end],
                gene_names,
                max_length=max_length,
                include_zero_genes=False,
                normalize=True,
                add_cls=True,
            )

            yield {
                "cell_ids": cell_ids,
                "genes": genes,
                "expressions": expressions,
                "src_key_padding_mask": src_key_padding_mask,
            }

    def forward_and_capture(
        self,
        model_handle: ScGPTHandle,
        batch: Dict[str, Any],
        layers: Sequence[str],
        capture_cfg: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Capture layer outputs via forward hooks on norm2 modules.
        Returns dict[layer_name -> Tensor[B,T,H]] on CPU.
        """
        model = model_handle.model
        device = model_handle.device

        genes = batch["genes"].to(device)
        expressions = batch["expressions"].to(device)
        src_key_padding_mask = batch["src_key_padding_mask"].to(device)

        # Map layer_name -> module
        layer_to_module = {}
        for lname in layers:
            if not lname.endswith(".norm2"):
                raise ValueError(f"Unsupported layer name for scGPT adapter: {lname}")
            idx = int(lname.split("_")[1].split(".")[0])
            layer_to_module[lname] = model.transformer_encoder.layers[idx].norm2

        captured: Dict[str, torch.Tensor] = {}
        hooks = []

        def make_hook(name: str):
            def hook(_module, _inp, out):
                # out expected: [B,T,H]
                captured[name] = out.detach().to("cpu")
            return hook

        for lname, module in layer_to_module.items():
            hooks.append(module.register_forward_hook(make_hook(lname)))

        with torch.no_grad():
            _ = model(genes, expressions, src_key_padding_mask)

        for h in hooks:
            h.remove()

        return captured
