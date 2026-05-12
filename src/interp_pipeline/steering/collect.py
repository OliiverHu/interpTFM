from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from interp_pipeline.adapters.models.scgpt import ScGPTHandle


def _unwrap(proxy) -> torch.Tensor:
    return proxy.value if hasattr(proxy, "value") else proxy


def _get_gene_names(adata) -> np.ndarray:
    if "gene_name" in adata.var.columns:
        return adata.var["gene_name"].to_numpy()
    return adata.var_names.to_numpy()


def _batch_X(adata, start: int, end: int) -> np.ndarray:
    X = adata.X[start:end]
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def collect_condition_activations(
    handle: ScGPTHandle,
    adata: object,
    condition_col: str = "condition",
    conditions: Optional[List[str]] = None,
    batch_size: int = 128,
    include_zero_genes: bool = False,
    normalize: bool = False,
    output_dir: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """
    Collect final-layer CLS activations for each perturbation condition.

    Runs a standard (unsteered) scGPT forward pass on cells grouped by
    condition and captures the CLS token from the model output.

    Args:
        handle:            ScGPTHandle (NNsight-wrapped model + tokenizer).
        adata:             AnnData filtered to tokenizer-known genes.
        condition_col:     obs column holding condition labels.
        conditions:        Which conditions to collect.  Defaults to all unique.
        batch_size:        Cells per forward pass.
        include_zero_genes: Tokenization mode.  False = only expressed genes
                            (variable-length, matching construct_non_zero_act.py).
        normalize:         Whether to normalise expression in the tokenizer.
        output_dir:        If given, writes {condition}_activations.pt per condition.

    Returns:
        {condition: Tensor[N, H]} — CLS activations per condition.
    """
    gene_names = _get_gene_names(adata)
    device = handle.device
    model = handle.model
    obs_conditions = adata.obs[condition_col].to_numpy()

    if conditions is None:
        conditions = list(np.unique(obs_conditions))

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    results: Dict[str, torch.Tensor] = {}

    for cond in tqdm(conditions, desc="collecting condition activations"):
        cell_mask = obs_conditions == cond
        adata_cond = adata[cell_mask]
        n = adata_cond.n_obs
        cls_list: list = []

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            X_batch = _batch_X(adata_cond, start, end)

            genes, expressions, mask = handle.tokenizer(
                X_batch,
                gene_names,
                include_zero_genes=include_zero_genes,
                normalize=normalize,
                add_cls=True,
            )
            genes = genes.to(device)
            expressions = expressions.to(device)
            if mask is not None:
                mask = mask.to(device)

            with torch.no_grad(), model.trace(genes, expressions, mask):
                out = model.output.cpu().save()

            out_tensor = _unwrap(out)  # [B, T, H]
            cls_list.append(out_tensor[:, 0, :])

        cond_acts = torch.cat(cls_list, dim=0)  # [N_cond, H]
        results[cond] = cond_acts

        if output_dir is not None:
            safe_name = cond.replace("/", "_").replace("+", "_")
            torch.save(cond_acts, os.path.join(output_dir, f"{safe_name}_activations.pt"))

    return results


def collect_per_layer_cls_activations(
    handle: ScGPTHandle,
    adata: object,
    gene_names: Optional[np.ndarray] = None,
    batch_size: int = 128,
    include_zero_genes: bool = True,
    normalize: bool = False,
) -> Dict[int, torch.Tensor]:
    """
    Collect CLS activations from every transformer layer for a set of cells.

    Captures `transformer_encoder.layers[i].output[:, 0, :]` (CLS token) at
    each layer in a single forward pass.  Used as input to analyze_probe_activations().

    Args:
        handle:            ScGPTHandle.
        adata:             AnnData (all cells whose CLS acts are needed).
        gene_names:        Gene name array; inferred from adata.var if None.
        batch_size:        Cells per forward pass.
        include_zero_genes: Whether to include zero-expression genes in tokens.
        normalize:         Whether to normalise expression.

    Returns:
        {layer_idx: Tensor[N, H]} — CLS activations for every layer.
    """
    if gene_names is None:
        gene_names = _get_gene_names(adata)

    device = handle.device
    model = handle.model
    n_layers = handle.n_layers
    n_cells = adata.n_obs

    cls_per_layer: Dict[int, list] = defaultdict(list)

    for start in tqdm(range(0, n_cells, batch_size), desc="collecting per-layer CLS"):
        end = min(start + batch_size, n_cells)
        X_batch = _batch_X(adata, start, end)

        genes, expressions, mask = handle.tokenizer(
            X_batch,
            gene_names,
            include_zero_genes=include_zero_genes,
            normalize=normalize,
            add_cls=True,
        )
        genes = genes.to(device)
        expressions = expressions.to(device)
        if mask is not None:
            mask = mask.to(device)

        saved: Dict[int, object] = {}
        with torch.no_grad(), model.trace(genes, expressions, mask):
            for i in range(n_layers):
                saved[i] = model.transformer_encoder.layers[i].output.cpu().save()

        for i in range(n_layers):
            act = _unwrap(saved[i])   # [B, T, H]
            cls_per_layer[i].append(act[:, 0, :])

    return {i: torch.cat(cls_per_layer[i], dim=0) for i in range(n_layers)}
