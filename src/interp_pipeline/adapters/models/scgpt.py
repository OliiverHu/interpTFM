from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Optional

import numpy as np
import torch

from interp_pipeline.adapters.model_base import ModelAdapter, ModelSpec
from interp_pipeline.types.activations import TokenUnit
from interp_pipeline.types.dataset import StandardDataset

from interp_pipeline.scgpt_local.load_model_from_pretrain import (
    create_clean_model_from_pretrain,
)


@dataclass
class ScGPTHandle:
    model: Any
    tokenizer: Any
    device: str
    n_layers: int


class ScGPTAdapter(ModelAdapter):
    """
    Adapter for scGPT models.

    Current assumptions:
    - dataset.adata is an AnnData-like object
    - tokenizer returns tokenized gene/expression tensors plus padding mask
    - NNsight is used to trace and capture layer outputs
    """

    def load(self, spec: ModelSpec) -> ScGPTHandle:
        model, tokenizer = create_clean_model_from_pretrain(
            spec.checkpoint,
            device=spec.device,
        )

        n_layers = None

        # The wrapped NNsight model should still expose underlying module attributes.
        if hasattr(model, "transformer_encoder") and hasattr(model.transformer_encoder, "layers"):
            n_layers = len(model.transformer_encoder.layers)
        elif hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
            n_layers = len(model.encoder.layers)
        elif hasattr(model, "layers"):
            n_layers = len(model.layers)

        if n_layers is None:
            raise RuntimeError("Could not infer number of transformer layers for scGPT model.")

        return ScGPTHandle(
            model=model,
            tokenizer=tokenizer,
            device=spec.device,
            n_layers=n_layers,
        )

    def list_layers(self, model_handle: ScGPTHandle) -> List[str]:
        return [f"layer_{i}" for i in range(model_handle.n_layers)]

    def infer_token_unit(self, layer_name: str) -> TokenUnit:
        return "gene"

    def make_batches(
        self,
        dataset: StandardDataset,
        model_handle: ScGPTHandle,
        batch_size: int,
        max_length: int,
        **kwargs: Any,
    ) -> Iterable[Dict[str, Any]]:
        adata = dataset.adata
        n = adata.n_obs

        if "feature_name" in adata.var.columns:
            gene_names = [str(x) for x in adata.var["feature_name"].tolist()]
        else:
            gene_names = [str(x) for x in adata.var_names.tolist()]

        X = adata.X

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            cell_ids = [str(x) for x in adata.obs_names[start:end]]

            # Slice first, densify second
            X_batch = X[start:end]
            if not isinstance(X_batch, np.ndarray):
                X_batch = X_batch.toarray()
            X_batch = X_batch.astype(np.float32, copy=False)

            # Your tokenizer appears to return (genes, expressions, src_key_padding_mask)
            genes, expressions, src_key_padding_mask = model_handle.tokenizer(
                X_batch,
                gene_names,
                max_length=max_length,
            )

            yield {
                "cell_ids": cell_ids,
                "genes": genes,
                "expressions": expressions,
                "src_key_padding_mask": src_key_padding_mask,
            }

    def _layer_name_to_index(self, layer_name: str) -> int:
        if not layer_name.startswith("layer_"):
            raise ValueError(f"Unsupported layer name format: {layer_name}")
        try:
            return int(layer_name.split("_", 1)[1])
        except Exception as e:
            raise ValueError(f"Could not parse layer index from: {layer_name}") from e

    def forward_and_capture(
        self,
        model_handle: ScGPTHandle,
        batch: Dict[str, Any],
        layers: Sequence[str],
        capture_cfg: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        capture_cfg = capture_cfg or {}

        model = model_handle.model
        device = model_handle.device

        genes = batch["genes"].to(device)
        expressions = batch["expressions"].to(device)
        src_key_padding_mask = batch["src_key_padding_mask"].to(device)

        layer_idxs = [self._layer_name_to_index(layer_name) for layer_name in layers]

        saved: Dict[str, Any] = {}

        with torch.no_grad():
            with model.trace(genes, expressions, src_key_padding_mask):
                for idx in layer_idxs:
                    layer_name = f"layer_{idx}"
                    saved[layer_name] = model.transformer_encoder.layers[idx].output.save()

        captured: Dict[str, torch.Tensor] = {}
        B = batch["genes"].shape[0]
        T = batch["genes"].shape[1]

        for layer_name, value in saved.items():
            acts = value.value if hasattr(value, "value") else value

            if not isinstance(acts, torch.Tensor):
                acts = torch.as_tensor(acts)

            # NNsight / transformer may return NestedTensor here.
            # Convert to dense [B, T, H] using zero padding.
            if getattr(acts, "is_nested", False):
                acts = acts.to_padded_tensor(0.0)

            if acts.ndim != 3:
                raise ValueError(
                    f"{layer_name}: expected 3D activations after capture, got ndim={acts.ndim}"
                )

            # Defensive fix in case backend returns [T, B, H]
            if acts.shape[0] == T and acts.shape[1] == B:
                acts = acts.transpose(0, 1).contiguous()

            captured[layer_name] = acts

        return captured

    def process_captured(
        self,
        captured: Dict[str, Any],
        batch: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Normalize captured scGPT outputs into the generic adapter format:
          {
            layer_name: {
              "acts": Tensor[N, H],
              "tok": List[str],
              "ex": List[str],
              "token_unit": "gene",
              "T_list": List[int],
            }
          }
        """
        genes = batch["genes"]
        src_key_padding_mask = batch["src_key_padding_mask"]
        cell_ids = batch["cell_ids"]

        if isinstance(genes, torch.Tensor):
            genes_cpu = genes.detach().cpu()
        else:
            genes_cpu = torch.as_tensor(genes)

        if isinstance(src_key_padding_mask, torch.Tensor):
            padmask_cpu = src_key_padding_mask.detach().cpu().bool()
        else:
            padmask_cpu = torch.as_tensor(src_key_padding_mask).bool()

        if genes_cpu.ndim != 2:
            raise ValueError(f"Expected genes to have shape [B, T], got {tuple(genes_cpu.shape)}")
        if padmask_cpu.shape != genes_cpu.shape:
            raise ValueError(
                f"Padding mask shape {tuple(padmask_cpu.shape)} "
                f"does not match genes shape {tuple(genes_cpu.shape)}"
            )

        out: Dict[str, Dict[str, Any]] = {}

        for layer_name, acts_btH in captured.items():
            if not isinstance(acts_btH, torch.Tensor):
                raise TypeError(f"{layer_name}: expected torch.Tensor, got {type(acts_btH)}")
            if acts_btH.ndim != 3:
                raise ValueError(f"{layer_name}: expected [B, T, H], got shape {tuple(acts_btH.shape)}")

            acts_cpu = acts_btH.detach().cpu()

            B, T, H = acts_cpu.shape
            if tuple(genes_cpu.shape) != (B, T):
                raise ValueError(
                    f"{layer_name}: genes shape {tuple(genes_cpu.shape)} "
                    f"does not match activations shape {(B, T, H)}"
                )

            valid_mask = ~padmask_cpu
            acts_flat = acts_cpu[valid_mask]            # [N, H]
            tok_flat = genes_cpu[valid_mask].tolist()   # length N

            ex_flat: List[str] = []
            valid_counts = [int(x) for x in valid_mask.sum(dim=1).tolist()]
            for cell_id, n_valid in zip(cell_ids, valid_counts):
                ex_flat.extend([str(cell_id)] * n_valid)

            n_rows = int(acts_flat.shape[0])

            if len(tok_flat) != n_rows:
                raise ValueError(
                    f"{layer_name}: token id count {len(tok_flat)} != activation rows {n_rows}"
                )
            if len(ex_flat) != n_rows:
                raise ValueError(
                    f"{layer_name}: example id count {len(ex_flat)} != activation rows {n_rows}"
                )

            out[layer_name] = {
                "acts": acts_flat,
                "tok": [str(x) for x in tok_flat],
                "ex": ex_flat,
                "token_unit": "gene",
                "T_list": valid_counts,
            }

        return out