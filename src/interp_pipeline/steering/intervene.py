from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from interp_pipeline.adapters.models.scgpt import ScGPTHandle


@dataclass(frozen=True)
class InterventionConfig:
    """Configuration for a single gene-steering experiment."""
    gene_select: str
    scale_list: List[float]
    batch_size: int = 256
    include_zero_genes: bool = True
    normalize: bool = False


def apply_scale_2D(
    resid: torch.Tensor,
    directions: torch.Tensor,
    scale: float,
    pos: int,
    mode: str = "amplify",
) -> torch.Tensor:
    """
    Steer residual[pos] along concept directions in-place.

    Must be called inside a NNsight model.trace() context where `resid` is
    a proxy for a transformer layer's output tensor [B, T, H].

    Args:
        resid:      Layer output proxy [B, T, H].
        directions: Concept direction columns [H, n_directions] on the correct device.
        scale:      Steering strength.
        pos:        Token position of the target gene (gene_index + 1 for CLS offset).
        mode:       "amplify" (default) — scales up the existing projection of the
                    residual onto the concept directions; has no effect when the
                    residual is already orthogonal to those directions.
                    "add" — unconditionally adds scale * direction to the residual,
                    matching the Arena / Geometry-of-Truth causal intervention style.

    Amplify update:
        dirs_normed = directions / ||directions||_col
        alpha       = resid[:, pos] @ dirs_normed          # [B, n_directions]
        resid[:, pos] += scale * (alpha @ dirs_normed.T)   # [B, H]

    Add update:
        dirs_normed = directions / ||directions||_col
        resid[:, pos] += scale * dirs_normed.sum(dim=1)    # [B, H]
    """
    dirs_normed = directions / directions.norm(dim=0, keepdim=True).clamp(min=1e-8)
    if mode == "add":
        # Unconditional additive steering — sum of unit concept directions.
        resid[:, pos] = resid[:, pos] + scale * dirs_normed.sum(dim=1)
    else:
        # Amplify existing projection (established default).
        alpha = resid[:, pos] @ dirs_normed                        # [B, n_directions]
        resid[:, pos] = resid[:, pos] + scale * (alpha @ dirs_normed.T)  # [B, H]
    return resid


def find_gene_position(
    gene_names: np.ndarray,
    gene_select: str,
    add_cls: bool = True,
) -> int:
    """
    Return the token sequence position of gene_select.

    With add_cls=True (default), position 0 is the CLS token so
    gene_select lands at gene_array_index + 1.

    Raises ValueError if gene_select is not in gene_names.
    """
    matches = np.where(gene_names == gene_select)[0]
    if len(matches) == 0:
        raise ValueError(
            f"Gene '{gene_select}' not found in gene_names array. "
            "Ensure adata has been filtered to tokenizer-known genes."
        )
    return int(matches[0]) + (1 if add_cls else 0)


def _unwrap(proxy) -> torch.Tensor:
    """Unwrap an NNsight save proxy to a plain tensor (handles both API versions)."""
    return proxy.value if hasattr(proxy, "value") else proxy


def run_intervention(
    handle: ScGPTHandle,
    adata_ctrl: object,
    probes: Dict[int, torch.Tensor],
    concept_idx_union: List[int],
    gene_position: int,
    cfg: InterventionConfig,
    gene_names: Optional[np.ndarray] = None,
    steering_mode: str = "amplify",
) -> Dict[float, torch.Tensor]:
    """
    Run scGPT forward passes on control cells with linear-probe-guided
    activation steering at every transformer layer, for each scale.

    The probe weight columns for the selected concept indices are used as
    steering directions in the residual stream at the target gene's token
    position.  Steering is applied simultaneously at all n_layers layers
    inside a single NNsight trace.

    Args:
        handle:            ScGPTHandle (model is NNsight-wrapped, has .tokenizer).
        adata_ctrl:        AnnData of control cells, already filtered to
                           tokenizer-known, non-zero-expression genes.
        probes:            {layer_idx: weight Tensor[H, n_concepts]}.
                           Loaded via linear_probe.load_probe().weight.detach().
        concept_idx_union: Column indices in the probe weight matrix corresponding
                           to concepts associated with gene_select.
        gene_position:     Token position of gene_select (from find_gene_position()).
        cfg:               Intervention settings (scales, batch_size, tokenisation).
        gene_names:        Gene name array matching adata_ctrl.var order.
                           Inferred from adata_ctrl.var if None.
        steering_mode:     Passed to apply_scale_2D.  "amplify" (default) scales
                           the existing projection; "add" unconditionally adds
                           scale * direction (Arena / Geometry-of-Truth style).

    Returns:
        {scale: Tensor[N_ctrl, H]} — CLS token activations after steering.
    """
    if gene_names is None:
        if "gene_name" in adata_ctrl.var.columns:
            gene_names = adata_ctrl.var["gene_name"].to_numpy()
        else:
            gene_names = adata_ctrl.var_names.to_numpy()

    device = handle.device
    model = handle.model
    n_layers = handle.n_layers
    n_cells = adata_ctrl.n_obs

    # Pre-move probe direction slices to device.
    directions: Dict[int, torch.Tensor] = {}
    for layer_idx, w in probes.items():
        directions[layer_idx] = w[:, concept_idx_union].to(device)  # [H, n_selected]

    results: Dict[float, torch.Tensor] = {}

    for scale in cfg.scale_list:
        cls_acts: list = []
        for start in tqdm(range(0, n_cells, cfg.batch_size), desc=f"scale={scale}"):
            end = min(start + cfg.batch_size, n_cells)
            X_batch = adata_ctrl.X[start:end]
            if hasattr(X_batch, "toarray"):
                X_batch = X_batch.toarray()
            X_batch = np.asarray(X_batch, dtype=np.float32)

            genes, expressions, mask = handle.tokenizer(
                X_batch,
                gene_names,
                include_zero_genes=cfg.include_zero_genes,
                normalize=cfg.normalize,
                add_cls=True,
            )
            genes = genes.to(device)
            expressions = expressions.to(device)
            if mask is not None:
                mask = mask.to(device)

            with torch.no_grad(), model.trace(genes, expressions, mask):
                for layer_idx in range(n_layers):
                    if layer_idx in directions:
                        acts = model.transformer_encoder.layers[layer_idx].output
                        apply_scale_2D(acts, directions[layer_idx], scale, gene_position, mode=steering_mode)
                output = model.output.cpu().save()

            out_tensor = _unwrap(output)  # [B, T, H]
            cls_acts.append(out_tensor[:, 0, :])  # CLS token

        results[scale] = torch.cat(cls_acts, dim=0)  # [N_ctrl, H]

    return results
