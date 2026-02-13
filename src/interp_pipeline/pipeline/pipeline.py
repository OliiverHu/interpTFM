from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List

import scanpy as sc

from types.dataset import StandardDataset
from adapters.model_base import ModelSpec
from adapters.models.scgpt import ScGPTAdapter
from extraction.extractor import extract_activations
from io.activation_store import ActivationStore, ActivationStoreSpec
from sae.sae_base import SAESpec
from sae.trainers import fit_sae_for_layer


@dataclass(frozen=True)
class PipelineConfig:
    adata_path: str
    model_name: str
    model_ckpt: str
    layers: List[str]
    output_dir: str

    batch_size: int = 16
    max_length: int = 1200
    sae_steps: int = 10_000
    sae_latents: int = 4096
    sae_l1: float = 1e-3
    device: str = "cuda"


def run(cfg: PipelineConfig) -> Dict[str, Any]:
    # 1) Dataset
    adata = sc.read_h5ad(cfg.adata_path)
    dataset = StandardDataset(adata=adata, obs_key_map={})
    dataset.validate()

    # 2) Model (scGPT for now)
    if cfg.model_name.lower() != "scgpt":
        raise ValueError("This minimal pipeline implementation currently wires only scGPT.")
    adapter = ScGPTAdapter()
    handle = adapter.load(ModelSpec(name="scgpt", checkpoint=cfg.model_ckpt, device=cfg.device))

    # 3) Extract activations
    store = ActivationStore(ActivationStoreSpec(root=cfg.output_dir))
    _, extracted_layers = extract_activations(
        dataset=dataset,
        model_handle=handle,
        model_adapter=adapter,
        layers=cfg.layers,
        store=store,
        extraction_cfg={
            "batch_size": cfg.batch_size,
            "max_length": cfg.max_length,
            "model_name": "scgpt",
        },
    )

    # 4) Train SAE per layer
    sae_spec = SAESpec(
        n_latents=cfg.sae_latents,
        l1=cfg.sae_l1,
        steps=cfg.sae_steps,
    )
    sae_results = []
    for layer in extracted_layers:
        r = fit_sae_for_layer(
            store=store,
            layer=layer,
            spec=sae_spec,
            output_dir=cfg.output_dir,
            device=cfg.device,
        )
        sae_results.append(r)

    return {"layers": extracted_layers, "sae": [r.summary for r in sae_results]}
