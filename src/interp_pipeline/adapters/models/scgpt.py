from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

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
        return [f"layer_{i}" for i in range(n)]

    def infer_token_unit(self, layer_name: str) -> TokenUnit:
        # currently only initialized with "gene"
        return "gene"

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
        Reliable capture: manually iterate transformer layers and record outputs.
        Avoids the nested-tensor TransformerEncoder fast-path where hooks don't fire.
        """
        model = model_handle.model
        device = model_handle.device

        genes = batch["genes"].to(device)                  # [B,T]
        expressions = batch["expressions"].to(device)      # [B,T]
        padmask = batch["src_key_padding_mask"].to(device) # [B,T] bool

        captured: Dict[str, torch.Tensor] = {}
        layer_idxs = sorted([int(lname.split("_")[1]) for lname in layers])
        with torch.no_grad(), model.trace(genes, expressions, padmask):
            for layer, idx in zip(layers, layer_idxs):
                captured[layer] = model.transformer_encoder.layers[idx].output.save()

        # Build input embeddings as model.forward does
        # x = model.encoder(genes) + model.value_encoder(expressions)  # [B,T,H]

        # # Parse which layer indices we want
        # want = {}
        # for lname in layers:
        #     idx = int(lname.split("_")[1])
        #     want[idx] = lname

        # captured: Dict[str, torch.Tensor] = {}
        # with torch.no_grad():
        #     for i, layer in enumerate(model.transformer_encoder.layers):
        #         x = layer(x, src_key_padding_mask=padmask)  # [B,T,H]
        #         if i in want:
        #             captured[want[i]] = x.detach().to("cpu")

        return captured

    def process_captured(
        self,
        captured: Dict[str, Any],
        batch: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Convert raw captured activations into buffer-ready entries.
        Handles token id extraction and flattening of [B, T, H] -> [B*T, H].
        """
        genes: torch.Tensor = batch["genes"].to("cpu")   # [B, T]
        cell_ids = batch["cell_ids"]
        padmask = batch["src_key_padding_mask"].to("cpu")  # [B, T] bool, True=padding

        # real token count per cell from padding mask
        T_list: List[int] = (~padmask).sum(dim=1).tolist()
        tok_list = [str(x) for x in genes[~padmask].tolist()]


        ex_list: List[str] = []
        for cid, t in zip(cell_ids, T_list):
            ex_list.extend([str(cid)] * t)

        result: Dict[str, Dict[str, Any]] = {}
        for layer_name, acts_btH in captured.items():
            flat = torch.cat(acts_btH.unbind(), dim=0)   # nested [B, T_i, H] -> [N, H]
            
            result[layer_name] = {
                "acts": flat,
                "tok": tok_list,
                "ex": ex_list,
                "token_unit": self.infer_token_unit(layer_name),
                "T_list": T_list,   # include T_list for potential downstream use
            }
        return result

    # def process_captured(
    #     self,
    #     captured: Dict[str, Any],
    #     batch: Dict[str, Any],
    # ) -> Dict[str, Dict[str, Any]]:
    #     """
    #     Convert raw captured activations into buffer-ready entries.
    #     Handles token id extraction and flattening of [B, T, H] -> [B*T, H].
    #     """
    #     genes: torch.Tensor = batch["genes"].to("cpu")   # [B, T]
    #     cell_ids = batch["cell_ids"]
    #     B, T = genes.shape

    #     tok_list = genes.reshape(B * T).tolist()
    #     tok_list = [str(x) for x in tok_list]

    #     ex_list: List[str] = []
    #     for cid in cell_ids:
    #         ex_list.extend([str(cid)] * T)

    #     result: Dict[str, Dict[str, Any]] = {}
    #     for layer_name, acts_btH in captured.items():
    #         flat = acts_btH.reshape(B * T, acts_btH.shape[-1]).contiguous()
    #         result[layer_name] = {
    #             "acts": flat,
    #             "tok": tok_list,
    #             "ex": ex_list,
    #             "token_unit": self.infer_token_unit(layer_name),
    #         }
    #     return result


