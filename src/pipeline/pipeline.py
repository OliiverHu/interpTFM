# src/interp_pipeline/pipeline.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from interp_pipeline.adapters.datasets.cosmx import CosMxDatasetAdapter
from interp_pipeline.adapters.models.scgpt import ScGPTAdapter
from interp_pipeline.adapters.models.geneformer import GeneformerAdapter
from interp_pipeline.adapters.models.c2s_scale import C2SScaleAdapter

from interp_pipeline.io.activation_store import ActivationStore, ActivationStoreSpec
from interp_pipeline.extraction.extractor import extract_activations

from interp_pipeline.sae.trainers import SAETrainer, SAESpec
from interp_pipeline.tis.runner import run_tis, TISSpec

from interp_pipeline.get_annotation.runner import run_go_annotation, GOAnnotationSpec

from interp_pipeline.downstream.context.niche_discovery import run_context_discovery, ContextSpec
from interp_pipeline.downstream.interaction.crosstalk import run_interaction_crosstalk, InteractionSpec


@dataclass(frozen=True)
class PipelineConfig:
    dataset: Dict[str, Any]
    model: Dict[str, Any]
    activations: Dict[str, Any]
    sae: Dict[str, Any]
    tis: Dict[str, Any]
    get_annotation: Dict[str, Any]
    context: Dict[str, Any]
    interaction: Dict[str, Any]
    output_dir: str


def _get_model_adapter(model_name: str):
    name = model_name.lower()
    if name == "scgpt":
        return ScGPTAdapter()
    if name == "geneformer":
        return GeneformerAdapter()
    if name in ("c2s-scale", "c2s_scale", "c2sscale"):
        return C2SScaleAdapter()
    raise ValueError(f"Unknown model: {model_name}")


def run_pipeline(cfg: PipelineConfig) -> Dict[str, Any]:
    """
    Sequential pipeline:
      1) load dataset (AnnData)
      2) load model
      3) extract activations (store to disk)
      4) fit SAE on selected layer(s)
      5) TIS sanity checks (gate)
      6) get_annotation (GO) for selected layer(s)/latents
      7) downstream context discovery
      8) downstream interaction crosstalk
    Returns a dict of artifact paths + summaries.
    """

    # 1) Dataset
    dataset_adapter = CosMxDatasetAdapter()
    dataset = dataset_adapter.load(cfg.dataset)
    dataset.validate()

    # 2) Model
    model_adapter = _get_model_adapter(cfg.model["name"])
    model = model_adapter.load(cfg.model)

    # 3) Activations → ActivationStore
    store = ActivationStore(ActivationStoreSpec(root=cfg.output_dir, **cfg.activations.get("store", {})))

    # layers can be provided explicitly, or derived from adapter + glob
    layers = cfg.activations.get("layers")
    if not layers:
        layers = model_adapter.list_layers(model)
        layer_glob = cfg.activations.get("layer_glob")
        if layer_glob:
            # TODO: filter layers by glob/regex
            pass

    act_index, act_layers = extract_activations(
        dataset=dataset,
        model=model,
        model_adapter=model_adapter,
        layers=layers,
        store=store,
        extraction_cfg=cfg.activations,
    )

    # 4) SAE
    sae_spec = SAESpec(**cfg.sae)
    sae_trainer = SAETrainer()
    sae_result = sae_trainer.fit(
        store=store,
        index=act_index,
        layers=cfg.sae.get("layers", act_layers),
        spec=sae_spec,
        output_dir=cfg.output_dir,
    )

    # 5) TIS gate
    tis_result = run_tis(
        store=store,
        sae_result=sae_result,
        dataset=dataset,
        spec=TISSpec(**cfg.tis),
        output_dir=cfg.output_dir,
    )

    artifacts: Dict[str, Any] = {
        "activations": {"layers": act_layers},
        "sae": sae_result,
        "tis": tis_result,
    }

    if not tis_result.passed:
        artifacts["stopped_reason"] = "TIS_FAILED"
        return artifacts

    # 6) get_annotation (GO, via g:Profiler)
    go_spec = GOAnnotationSpec(**cfg.get_annotation)
    go_result = run_go_annotation(
        sae_result=sae_result,
        dataset=dataset,
        spec=go_spec,
        output_dir=cfg.output_dir,
    )
    artifacts["get_annotation"] = go_result

    # 7) Context-level downstream
    ctx_result = run_context_discovery(
        dataset=dataset,
        sae_result=sae_result,
        annotations=go_result,
        spec=ContextSpec(**cfg.context),
        output_dir=cfg.output_dir,
    )
    artifacts["context"] = ctx_result

    # 8) Interaction-level downstream
    int_result = run_interaction_crosstalk(
        dataset=dataset,
        sae_result=sae_result,
        annotations=go_result,
        context_result=ctx_result,
        spec=InteractionSpec(**cfg.interaction),
        output_dir=cfg.output_dir,
    )
    artifacts["interaction"] = int_result

    return artifacts
