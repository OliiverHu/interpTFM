from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import scanpy as sc

from interp_pipeline.types.dataset import StandardDataset
from interp_pipeline.adapters.model_base import ModelSpec
from interp_pipeline.adapters.models.scgpt import ScGPTAdapter
from interp_pipeline.extraction.extractor import extract_activations
from interp_pipeline.io.activation_store import ActivationStore, ActivationStoreSpec
from interp_pipeline.sae.sae_base import SAESpec
from interp_pipeline.sae.trainers import fit_sae_for_layer


@dataclass(frozen=True)
class PipelineConfig:
    # data / model
    adata_path: str
    model_name: str
    model_ckpt: str
    layers: List[str]
    output_dir: str

    # extraction
    batch_size: int = 16
    max_length: int = 1200
    extraction_cfg: Dict[str, Any] = field(default_factory=dict)

    # SAE
    sae_spec: Optional[SAESpec] = None
    sae_batch_size: Optional[int] = None

    # runtime
    device: str = "cuda"
    model_options: Dict[str, Any] = field(default_factory=dict)


def _validate_config(cfg: PipelineConfig) -> None:
    if not cfg.adata_path:
        raise ValueError("PipelineConfig.adata_path must be non-empty.")
    if not cfg.model_name:
        raise ValueError("PipelineConfig.model_name must be non-empty.")
    if not cfg.model_ckpt:
        raise ValueError("PipelineConfig.model_ckpt must be non-empty.")
    if not cfg.output_dir:
        raise ValueError("PipelineConfig.output_dir must be non-empty.")
    if not cfg.layers:
        raise ValueError("PipelineConfig.layers must contain at least one layer.")

    if cfg.batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {cfg.batch_size}")
    if cfg.max_length <= 0:
        raise ValueError(f"max_length must be > 0, got {cfg.max_length}")


def _build_adapter(model_name: str):
    """
    Lazy imports on purpose.

    Supported for now:
      - scgpt
      - c2s_scale / c2sscale

    geneformer can be added later here.
    """
    key = model_name.strip().lower()

    if key == "scgpt":
        return ScGPTAdapter()

    if key in {"c2s_scale", "c2sscale"}:
        from interp_pipeline.adapters.models.c2s_scale import C2SScaleAdapter
        return C2SScaleAdapter()

    raise ValueError(
        f"Unsupported model_name='{model_name}'. "
        "Supported for now: ['scgpt', 'c2s_scale']"
    )


def _validate_requested_layers(
    requested_layers: List[str],
    available_layers: List[str],
) -> None:
    missing = [layer for layer in requested_layers if layer not in available_layers]
    if missing:
        raise ValueError(
            f"Requested layers not found: {missing}. "
            f"Available layers: {available_layers}"
        )


def _make_model_spec(cfg: PipelineConfig) -> ModelSpec:
    return ModelSpec(
        name=cfg.model_name,
        checkpoint=cfg.model_ckpt,
        device=cfg.device,
        options=dict(cfg.model_options),
    )


def _make_store(output_dir: str) -> ActivationStore:
    return ActivationStore(
        ActivationStoreSpec(root=output_dir)
    )


def _make_extraction_cfg(cfg: PipelineConfig) -> Dict[str, Any]:
    out = {
        "model_name": cfg.model_name,
        "batch_size": cfg.batch_size,
        "max_length": cfg.max_length,
        "device": cfg.device,
    }
    out.update(cfg.extraction_cfg)
    return out


def _make_default_sae_spec() -> SAESpec:
    return SAESpec(
        n_latents=4096,
        l1=1e-3,
        lr=1e-4,
        steps=20_000,
        seed=0,
    )


def run(cfg: PipelineConfig) -> Dict[str, Any]:
    """
    Minimal end-to-end pipeline for the CURRENT repo layout:

      1. Read AnnData
      2. Wrap as StandardDataset
      3. Build model adapter
      4. Validate requested layer names
      5. Extract activations for requested layers
      6. Train one SAE per requested layer
    """
    _validate_config(cfg)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[pipeline] reading AnnData from: {cfg.adata_path}")
    adata = sc.read_h5ad(cfg.adata_path)
    dataset = StandardDataset(adata=adata)

    adapter = _build_adapter(cfg.model_name)
    model_spec = _make_model_spec(cfg)

    print(
        f"[pipeline] loading model '{cfg.model_name}' "
        f"from checkpoint: {cfg.model_ckpt}"
    )
    model_handle = adapter.load(model_spec)

    available_layers = adapter.list_layers(model_handle)
    _validate_requested_layers(cfg.layers, available_layers)

    print(f"[pipeline] using layers: {cfg.layers}")
    store = _make_store(str(output_dir))

    extraction_cfg = _make_extraction_cfg(cfg)
    print(f"[pipeline] extraction config: {extraction_cfg}")

    activation_index = extract_activations(
        dataset=dataset,
        model_adapter=adapter,
        model_handle=model_handle,
        store=store,
        layers=cfg.layers,
        extraction_cfg=extraction_cfg,
    )

    sae_spec = cfg.sae_spec if cfg.sae_spec is not None else _make_default_sae_spec()

    sae_results: Dict[str, Any] = {}
    for layer_name in cfg.layers:
        print(f"[pipeline] training SAE for layer: {layer_name}")

        fit_kwargs: Dict[str, Any] = {
            "store": store,
            "layer": layer_name,
            "spec": sae_spec,
            "output_dir": str(output_dir),
            "device": cfg.device,
        }
        if cfg.sae_batch_size is not None:
            fit_kwargs["batch_size"] = cfg.sae_batch_size

        sae_results[layer_name] = fit_sae_for_layer(**fit_kwargs)

    result = {
        "activation_index": activation_index,
        "sae_results": sae_results,
        "layers": list(cfg.layers),
        "output_dir": str(output_dir),
        "model_name": cfg.model_name,
    }

    print("[pipeline] done")
    return result