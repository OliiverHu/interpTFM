from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

DEFAULT_ADATA = "/maiziezhou_lab2/yunfei/Projects/FM_temp/interGFM/ge_shards/cosmx_human_lung_sec8.h5ad"
DEFAULT_SCGPT_FOUNDATION_CKPT = "/maiziezhou_lab/zihang/SpatialFoundationModel/eval/scGPT/whole-human-pretrain"

MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    "scgpt": {
        "store_root": "/maiziezhou_lab2/yunfei/Projects/interpTFM/acts_extraction_scgpt_cosmx",
        "activation_pooling": "token",
        "token_value": "60695",
        "cell_feature_pooling": "mean",
        "layers": [
            "layer_0.norm2", "layer_1.norm2", "layer_2.norm2", "layer_3.norm2",
            "layer_4.norm2", "layer_5.norm2", "layer_6.norm2", "layer_7.norm2",
            "layer_8.norm2", "layer_9.norm2", "layer_10.norm2", "layer_11.norm2",
        ],
    },
    "c2sscale": {
        "store_root": "/maiziezhou_lab2/yunfei/Projects/interpTFM/acts_extraction_c2s_cosmx_store",
        "activation_pooling": "mean",
        "token_value": None,
        "cell_feature_pooling": "mean",
        "layers": ["layer_0", "layer_6", "layer_13", "layer_15", "layer_17", "layer_19", "layer_21", "layer_23", "layer_25"],
    },
    "geneformer": {
        "store_root": "/maiziezhou_lab2/yunfei/Projects/interpTFM/acts_extraction_geneformer_cosmx",
        "activation_pooling": "token",
        "token_value": "<cls>",
        "cell_feature_pooling": "mean",
        "layers": ["layer_1", "layer_4", "layer_7", "layer_10", "layer_12", "layer_14", "layer_16", "layer_17"],
    },
}

@dataclass(frozen=True)
class LayerJob:
    model: str
    layer: str
    store_root: str
    layer_tag: str
    job_dir: str
    activation_pooling: str
    token_value: Optional[str]
    cell_feature_pooling: str


def layer_tag(layer: str) -> str:
    return layer.replace(".", "_").replace("/", "_")


def default_run_tag(steps: int, l1: float, no_resample: bool) -> str:
    l1s = f"{l1:g}".replace(".", "p")
    nr = "nr_off" if no_resample else "nr_on"
    return f"{nr}__steps_{int(steps)}__l1_{l1s}"


def build_jobs(out_root: str, models: Optional[Sequence[str]] = None) -> List[LayerJob]:
    use_models = list(models) if models else list(MODEL_SPECS.keys())
    jobs: List[LayerJob] = []
    for model in use_models:
        if model not in MODEL_SPECS:
            raise ValueError(f"Unknown model={model!r}; valid={sorted(MODEL_SPECS)}")
        spec = MODEL_SPECS[model]
        for layer in spec["layers"]:
            tag = layer_tag(layer)
            jobs.append(
                LayerJob(
                    model=model,
                    layer=layer,
                    store_root=spec["store_root"],
                    layer_tag=tag,
                    job_dir=str(Path(out_root) / model / tag),
                    activation_pooling=str(spec.get("activation_pooling", "token")),
                    token_value=spec.get("token_value"),
                    cell_feature_pooling=str(spec.get("cell_feature_pooling", "mean")),
                )
            )
    return jobs


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def import_script_module(path: str, module_name: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Required backend script not found: {p}")
    spec = importlib.util.spec_from_file_location(module_name, str(p))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import backend script: {p}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def patch_train_backend_api(train_mod: Any) -> None:
    """Compatibility shim for branch drift, without changing training logic.

    The archived all-layer runner deliberately delegates SAE training to
    ``scripts/test_train_3saes.py`` and preserves that script's parameters and
    checkpoint behavior.  The only issue seen across branches is that the
    trainer's resampling helper expects ``AutoEncoder.dict_size``; some checked
    out package versions expose only ``encoder``/``decoder``.  Add read-only
    properties on the imported backend class when missing so the original
    trainer can run unchanged.

    This does not edit ``scripts/test_train_3saes.py`` or
    ``src/interp_pipeline/sae/sae_base.py`` on disk.
    """
    ae_cls = getattr(train_mod, "AutoEncoder", None)
    if ae_cls is None:
        return

    def _dict_size(self) -> int:
        enc = getattr(self, "encoder", None)
        if enc is not None:
            if hasattr(enc, "out_features"):
                return int(enc.out_features)
            weight = getattr(enc, "weight", None)
            if weight is not None:
                return int(weight.shape[0])
        dec = getattr(self, "decoder", None)
        if dec is not None:
            if hasattr(dec, "in_features"):
                return int(dec.in_features)
            weight = getattr(dec, "weight", None)
            if weight is not None:
                return int(weight.shape[1])
        raise AttributeError("Could not infer AutoEncoder.dict_size from encoder/decoder")

    def _activation_dim(self) -> int:
        enc = getattr(self, "encoder", None)
        if enc is not None:
            if hasattr(enc, "in_features"):
                return int(enc.in_features)
            weight = getattr(enc, "weight", None)
            if weight is not None:
                return int(weight.shape[1])
        dec = getattr(self, "decoder", None)
        if dec is not None:
            if hasattr(dec, "out_features"):
                return int(dec.out_features)
            weight = getattr(dec, "weight", None)
            if weight is not None:
                return int(weight.shape[0])
        raise AttributeError("Could not infer AutoEncoder.activation_dim from encoder/decoder")

    if not hasattr(ae_cls, "dict_size"):
        setattr(ae_cls, "dict_size", property(_dict_size))
    if not hasattr(ae_cls, "activation_dim"):
        setattr(ae_cls, "activation_dim", property(_activation_dim))


def marker_path(job: LayerJob, stage: str, run_tag: str) -> Path:
    return Path(job.job_dir) / f".{stage}_{run_tag}_done.json"


def fail_path(job: LayerJob, stage: str, run_tag: str) -> Path:
    return Path(job.job_dir) / f".{stage}_{run_tag}_failed.json"


def sae_dir(job: LayerJob, run_tag: str) -> Path:
    return Path(job.job_dir) / "sae" / run_tag


def find_unique_file(candidates: Sequence[Path], glob_dir: Optional[Path] = None, pattern: str = "") -> Path:
    """Return the first existing candidate, or a unique glob match.

    This keeps the wrapper compatible with the original 3-model trainer while
    tolerating branch differences in checkpoint filename sanitization, e.g.
    ``sae_layer_0.norm2_best.pt`` vs ``sae_layer_0_norm2_best.pt``.
    """
    for p in candidates:
        if p.exists():
            return p
    if glob_dir is not None and pattern:
        matches = sorted(glob_dir.glob(pattern))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise RuntimeError(
                f"Ambiguous checkpoint matches under {glob_dir}: "
                + ", ".join(str(m.name) for m in matches)
            )
    # Return the canonical first candidate for readable FileNotFoundError messages.
    return candidates[0]


def sae_best_ckpt(job: LayerJob, run_tag: str) -> Path:
    d = sae_dir(job, run_tag)
    return find_unique_file(
        [
            d / f"sae_{job.layer}_best.pt",
            d / f"sae_{job.layer_tag}_best.pt",
            d / f"sae_{layer_tag(job.layer)}_best.pt",
        ],
        glob_dir=d,
        pattern="sae_*_best.pt",
    )


def sae_last_ckpt(job: LayerJob, run_tag: str) -> Path:
    d = sae_dir(job, run_tag)
    return find_unique_file(
        [
            d / f"sae_{job.layer}_last.pt",
            d / f"sae_{job.layer_tag}_last.pt",
            d / f"sae_{layer_tag(job.layer)}_last.pt",
        ],
        glob_dir=d,
        pattern="sae_*_last.pt",
    )


def qc_dir(job: LayerJob, run_tag: str) -> Path:
    return Path(job.job_dir) / "activation_qc" / run_tag


def f1_dir(job: LayerJob, run_tag: str) -> Path:
    return Path(job.job_dir) / "f1_heldout" / run_tag


def stage_complete(job: LayerJob, stage: str, run_tag: str) -> bool:
    marker = marker_path(job, stage, run_tag)
    if not marker.exists():
        return False
    if stage == "sae":
        return sae_best_ckpt(job, run_tag).exists()
    if stage == "qc":
        return (qc_dir(job, run_tag) / "dead_neuron_summary_all.json").exists()
    if stage == "f1":
        out = f1_dir(job, run_tag)
        if (out / "test_cell_concept_f1_scores.csv").exists() and (out / "counts_summary_cell.json").exists():
            return True
        return marker.exists() and any(out.glob("**/*.csv"))
    return marker.exists()


def run_sae(job: LayerJob, args: argparse.Namespace, run_tag: str) -> Dict[str, Any]:
    from interp_pipeline.io.activation_store import ActivationStore, ActivationStoreSpec

    train_mod = import_script_module(args.train_script, "archived_backend_train_3saes")
    patch_train_backend_api(train_mod)
    out_dir = sae_dir(job, run_tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    store = ActivationStore(ActivationStoreSpec(root=job.store_root))
    d_in = int(train_mod.infer_d_in(store, job.layer, args.batch_size))
    n_latents = int(args.n_latents) if args.n_latents is not None else int(d_in * args.latent_multiplier)

    train_args = SimpleNamespace(
        l1=args.l1,
        lr=args.lr,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        resample_steps=args.resample_steps,
        no_resample=args.no_resample,
        seed=args.seed,
    )
    spec = train_mod.build_spec(train_args, n_latents=n_latents)

    print(f"[sae] {job.model} {job.layer} d_in={d_in} n_latents={n_latents} out={out_dir}")
    summary = train_mod.fit_sae_for_layer_bestlast(
        store=store,
        layer=job.layer,
        spec=spec,
        output_dir=str(out_dir),
        label=job.model,
        device=args.device,
        batch_size=args.batch_size,
        save_every=args.save_every,
        best_metric=args.best_metric,
    )
    write_json(marker_path(job, "sae", run_tag), {"stage": "sae", "job": asdict(job), "summary": summary})
    return summary


def run_qc(job: LayerJob, args: argparse.Namespace, run_tag: str) -> Dict[str, Any]:
    qc_mod = import_script_module(args.qc_script, "archived_backend_audit_dead_neurons")
    ckpt = sae_best_ckpt(job, run_tag)
    if not ckpt.exists():
        raise FileNotFoundError(f"SAE checkpoint missing for QC: {ckpt}")
    out_dir = qc_dir(job, run_tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[qc] {job.model} {job.layer} ckpt={ckpt} out={out_dir}")
    feat_df, summary = qc_mod.audit_model(
        label=job.model,
        ckpt_path=ckpt,
        store_root=Path(job.store_root),
        layer=job.layer,
        split=args.qc_split,
        val_fraction=args.val_fraction,
        max_shards=None if str(args.qc_max_shards).upper() == "NONE" else int(args.qc_max_shards),
        token_chunk_size=args.token_chunk_size,
        dead_eps=args.dead_eps,
        near_dead_rate=args.near_dead_rate,
    )
    feat_csv = out_dir / f"feature_stats_{args.qc_split}.csv"
    feat_df.to_csv(feat_csv, index=False)
    pd.DataFrame([summary]).to_csv(out_dir / f"dead_neuron_summary_{args.qc_split}.csv", index=False)
    write_json(out_dir / f"dead_neuron_summary_{args.qc_split}.json", [summary])

    # Reuse plotting helpers from the backend script.
    qc_mod.plot_hist(feat_df["active_rate"].tolist(), f"{job.model} active-rate ({args.qc_split})", "Feature active rate", out_dir / f"active_rate_hist_{args.qc_split}.png", bins=50)
    qc_mod.plot_hist(feat_df["encoder_l2_norm"].tolist(), f"{job.model} encoder L2 norm ({args.qc_split})", "Encoder L2 norm", out_dir / f"encoder_l2_norm_hist_{args.qc_split}.png", bins=50)
    qc_mod.plot_scatter(feat_df["active_rate"].tolist(), feat_df["encoder_l2_norm"].tolist(), f"{job.model} active rate vs encoder L2 norm", "Feature active rate", "Encoder L2 norm", out_dir / f"active_rate_vs_encoder_l2_norm_{args.qc_split}.png", log_x=True)
    qc_mod.plot_hist(feat_df["max_activation"].tolist(), f"{job.model} max activation ({args.qc_split})", "Feature max activation", out_dir / f"max_activation_hist_{args.qc_split}.png", bins=50, log_x=True)

    write_json(marker_path(job, "qc", run_tag), {"stage": "qc", "job": asdict(job), "summary": summary})
    return summary


def prepare_adata_schema(adata_path: str, cache_root: Path) -> str:
    """Patch AnnData var schema as expected by heldout_report_for_layer.

    This does not rebuild GT and does not create a heldout dataset.
    """
    out = cache_root / "_tmp" / "schema_patched_same_adata.h5ad"
    manifest = cache_root / "_tmp" / "schema_patch_manifest.json"
    if out.exists() and manifest.exists():
        return str(out)

    import scanpy as sc
    adata = sc.read_h5ad(adata_path).copy()
    adata.var["ensembl_id"] = adata.var_names.astype(str)
    if "feature_name" in adata.var.columns:
        adata.var["gene_symbol"] = adata.var["feature_name"].astype(str)
        adata.var["index"] = adata.var["feature_name"].astype(str)
    elif "index" in adata.var.columns:
        adata.var["gene_symbol"] = adata.var["index"].astype(str)
    else:
        raise ValueError("Need feature_name or index in adata.var for F1 heldout schema patch")
    out.parent.mkdir(parents=True, exist_ok=True)
    adata.write(out)
    write_json(manifest, {"input_adata_path": adata_path, "output_adata_path": str(out), "n_obs": int(adata.n_obs), "n_vars": int(adata.n_vars)})
    return str(out)


def run_gene_heldout_f1(job: LayerJob, args: argparse.Namespace, run_tag: str) -> Dict[str, Any]:
    """Legacy gene-heldout backend retained as an explicit optional mode."""
    from interp_pipeline.get_annotation.f1_alignment import heldout_report_for_layer

    gt_csv = Path(args.gt_csv) if args.gt_csv else None
    if gt_csv is None or not gt_csv.exists():
        raise FileNotFoundError("--gt-csv is required and must point to an existing GT CSV")

    ckpt = sae_best_ckpt(job, run_tag)
    if not ckpt.exists():
        raise FileNotFoundError(f"SAE checkpoint missing for F1: {ckpt}")

    out_dir = f1_dir(job, run_tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    patched_adata = prepare_adata_schema(args.adata_path, Path(args.out_root)) if args.patch_adata_schema else args.adata_path
    aux_ckpt = args.scgpt_foundation_ckpt if job.model == "scgpt" else None

    print(f"[f1:gene-heldout] {job.model} {job.layer} ckpt={ckpt} gt={gt_csv} out={out_dir}")
    heldout_report_for_layer(
        layer=job.layer,
        store_root=job.store_root,
        gt_csv=str(gt_csv),
        sae_ckpt_path=str(ckpt),
        out_dir=str(out_dir),
        latent_thresholds=list(args.latent_thresholds),
        valid_frac=args.valid_frac,
        test_frac=args.test_frac,
        seed=args.f1_seed,
        topM_valid_per_concept_per_threshold=args.topM_valid_per_concept_per_threshold,
        batch_size=args.f1_batch_size,
        adata_path=patched_adata,
        scgpt_ckpt=aux_ckpt,
        device=args.device,
        dev_mode=False,
        dev_max_shards=None,
        dev_max_rows_per_split_per_shard=None,
        dev_only_valid=False,
    )
    summary = {"stage": "f1", "mode": "gene_heldout", "job": asdict(job), "out_dir": str(out_dir), "gt_csv": str(gt_csv), "sae_ckpt": str(ckpt)}
    write_json(marker_path(job, "f1", run_tag), summary)
    return summary


def run_cell_heldout_f1_stage(job: LayerJob, args: argparse.Namespace, run_tag: str) -> Dict[str, Any]:
    from interp_pipeline.layer_experiments.cell_heldout_f1 import CellHeldoutF1Config, run_cell_heldout_f1

    gt_csv = Path(args.gt_csv) if args.gt_csv else None
    if gt_csv is None or not gt_csv.exists():
        raise FileNotFoundError("--gt-csv is required and must point to an existing GT CSV")
    ckpt = sae_best_ckpt(job, run_tag)
    if not ckpt.exists():
        raise FileNotFoundError(f"SAE checkpoint missing for cell-heldout F1: {ckpt}")

    out_dir = f1_dir(job, run_tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = CellHeldoutF1Config(
        model=job.model,
        layer=job.layer,
        store_root=job.store_root,
        sae_ckpt_path=str(ckpt),
        gt_csv=str(gt_csv),
        adata_path=args.adata_path,
        out_dir=str(out_dir),
        token_value=job.token_value,
        activation_pooling=job.activation_pooling,
        c2s_cell_pooling=job.cell_feature_pooling,
        split_seed=args.f1_seed,
        train_frac=args.cell_train_frac,
        valid_frac=args.valid_frac,
        test_frac=args.test_frac,
        concept_score_quantile=args.concept_score_quantile,
        latent_thresholds=tuple(float(x) for x in args.latent_thresholds),
        batch_size=args.f1_batch_size,
        max_shards=None if str(args.f1_max_shards).upper() == "NONE" else int(args.f1_max_shards),
        device=args.device,
        min_concept_genes=args.min_concept_genes,
        min_pos_valid=args.min_pos_valid,
        min_pos_test=args.min_pos_test,
        min_neg_valid=args.min_neg_valid,
        min_neg_test=args.min_neg_test,
        max_concepts=args.max_concepts,
        dedupe_identical_concepts=not args.no_dedupe_identical_concepts,
    )
    print(f"[f1:cell-heldout] {job.model} {job.layer} ckpt={ckpt} gt={gt_csv} out={out_dir}")
    summary = run_cell_heldout_f1(cfg)
    marker = {"stage": "f1", "mode": "cell_heldout", "job": asdict(job), "summary": summary, "out_dir": str(out_dir), "gt_csv": str(gt_csv), "sae_ckpt": str(ckpt)}
    write_json(marker_path(job, "f1", run_tag), marker)
    return marker


def run_f1(job: LayerJob, args: argparse.Namespace, run_tag: str) -> Dict[str, Any]:
    if args.f1_mode == "gene":
        return run_gene_heldout_f1(job, args, run_tag)
    if args.f1_mode == "cell":
        return run_cell_heldout_f1_stage(job, args, run_tag)
    raise ValueError(f"Unsupported --f1-mode: {args.f1_mode}")

def run_f1_downstream(job: LayerJob, args: argparse.Namespace, run_tag: str) -> Dict[str, Any]:
    # This is intentionally conservative because downstream F1 script flags changed during development.
    # The user can provide a template command with placeholders when ready.
    if not args.f1_downstream_template:
        raise ValueError("f1_downstream requires --f1-downstream-template. Keep this stage disabled until the exact branch CLI is confirmed.")
    out_dir = Path(job.job_dir) / "f1_downstream" / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "model": job.model,
        "layer": job.layer,
        "layer_tag": job.layer_tag,
        "job_dir": job.job_dir,
        "store_root": job.store_root,
        "sae_ckpt": str(sae_best_ckpt(job, run_tag)),
        "f1_dir": str(f1_dir(job, run_tag)),
        "out_dir": str(out_dir),
        "run_tag": run_tag,
    }
    cmd = args.f1_downstream_template.format(**mapping)
    print(f"[f1_downstream] {cmd}")
    subprocess.run(cmd, shell=True, check=True)
    summary = {"stage": "f1_downstream", "job": asdict(job), "cmd": cmd, "out_dir": str(out_dir)}
    write_json(marker_path(job, "f1_downstream", run_tag), summary)
    return summary


def run_one_stage(job: LayerJob, stage: str, args: argparse.Namespace, run_tag: str) -> None:
    if stage_complete(job, stage, run_tag) and not args.force:
        print(f"[skip:{stage}] {job.model} {job.layer}")
        return
    try:
        Path(job.job_dir).mkdir(parents=True, exist_ok=True)
        write_json(Path(job.job_dir) / ".job.json", asdict(job))
        if stage == "sae":
            run_sae(job, args, run_tag)
        elif stage == "qc":
            run_qc(job, args, run_tag)
        elif stage == "f1":
            run_f1(job, args, run_tag)
        elif stage == "f1_downstream":
            run_f1_downstream(job, args, run_tag)
        else:
            raise ValueError(f"Unknown stage: {stage}")
        fp = fail_path(job, stage, run_tag)
        if fp.exists():
            fp.unlink()
    except Exception as e:
        write_json(fail_path(job, stage, run_tag), {"stage": stage, "job": asdict(job), "error": repr(e)})
        raise


def status(jobs: Sequence[LayerJob], stages: Sequence[str], run_tag: str) -> pd.DataFrame:
    rows = []
    for job in jobs:
        for stage in stages:
            rows.append({
                "model": job.model,
                "layer": job.layer,
                "stage": stage,
                "complete": stage_complete(job, stage, run_tag),
                "failed_marker": fail_path(job, stage, run_tag).exists(),
                "job_dir": job.job_dir,
            })
    return pd.DataFrame(rows)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run independent all-layer SAE/QC/F1 jobs for interpTFM Phase 2 experiments.")
    ap.add_argument("--out-root", default="runs/all_layer_sae_f1_cosmx")
    ap.add_argument("--models", nargs="+", default=["scgpt", "c2sscale", "geneformer"])
    ap.add_argument("--stages", nargs="+", default=["sae", "qc", "f1"], choices=["sae", "qc", "f1", "f1_downstream"])
    ap.add_argument("--run-tag", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--status-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")

    # Backend scripts used for SAE and QC.
    ap.add_argument("--train-script", default="scripts/test_train_3saes.py")
    ap.add_argument("--qc-script", default="scripts/audit_dead_neurons_3models.py")

    # SAE hyperparameters; aligned with test_train_3saes.py.
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--latent-multiplier", type=int, default=8)
    ap.add_argument("--n-latents", type=int, default=None, help="Explicit n_latents for every layer; otherwise latent_multiplier * d_in")
    ap.add_argument("--l1", type=float, default=1e-3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--warmup-steps", type=int, default=1000)
    ap.add_argument("--resample-steps", type=int, default=2000)
    ap.add_argument("--no-resample", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--best-metric", choices=["loss", "recon", "sparsity"], default="loss")

    # QC args; aligned with audit_dead_neurons_3models.py.
    ap.add_argument("--qc-split", choices=["all", "train", "val"], default="all")
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--qc-max-shards", default="NONE")
    ap.add_argument("--token-chunk-size", type=int, default=8192)
    ap.add_argument("--dead-eps", type=float, default=1e-8)
    ap.add_argument("--near-dead-rate", type=float, default=1e-4)

    # F1 args; uses existing GT, does not rebuild GT.
    # Default is cell-heldout F1. The old gene-heldout backend is retained only as --f1-mode gene.
    ap.add_argument("--f1-mode", choices=["cell", "gene"], default="cell")
    ap.add_argument("--gt-csv", default=None, help="Existing gprofiler_binary_gene_by_term.csv. Required for --stages f1.")
    ap.add_argument("--adata-path", default=DEFAULT_ADATA)
    ap.add_argument("--patch-adata-schema", action="store_true", default=True)
    ap.add_argument("--no-patch-adata-schema", action="store_false", dest="patch_adata_schema")
    ap.add_argument("--latent-thresholds", nargs="+", type=float, default=[0.0, 0.15, 0.3, 0.6])
    ap.add_argument("--valid-frac", type=float, default=0.2)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--f1-seed", type=int, default=0)
    ap.add_argument("--topM-valid-per-concept-per-threshold", type=int, default=200)
    ap.add_argument("--f1-batch-size", type=int, default=8192)
    ap.add_argument("--f1-max-shards", default="NONE", help="Optional shard cap for F1 debugging; default NONE uses all shards.")
    ap.add_argument("--cell-train-frac", type=float, default=0.7)
    ap.add_argument("--concept-score-quantile", type=float, default=0.75, help="Positive-cell threshold estimated on train cells and applied to valid/test.")
    ap.add_argument("--min-concept-genes", type=int, default=3)
    ap.add_argument("--min-pos-valid", type=int, default=20)
    ap.add_argument("--min-pos-test", type=int, default=20)
    ap.add_argument("--min-neg-valid", type=int, default=20)
    ap.add_argument("--min-neg-test", type=int, default=20)
    ap.add_argument("--max-concepts", type=int, default=None, help="Debug cap for number of concepts after filtering/deduplication.")
    ap.add_argument("--no-dedupe-identical-concepts", action="store_true")
    ap.add_argument("--scgpt-foundation-ckpt", default=DEFAULT_SCGPT_FOUNDATION_CKPT)

    # Optional downstream F1 hook; left explicit to avoid guessing branch-specific CLI flags.
    ap.add_argument("--f1-downstream-template", default=None)
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    run_tag = args.run_tag or default_run_tag(args.steps, args.l1, args.no_resample)
    jobs = build_jobs(args.out_root, args.models)
    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    write_json(Path(args.out_root) / "manifest.json", {"run_tag": run_tag, "jobs": [asdict(j) for j in jobs]})

    st = status(jobs, args.stages, run_tag)
    st.to_csv(Path(args.out_root) / f"status_{run_tag}.csv", index=False)
    print("Status summary")
    print(st.groupby(["stage", "complete", "failed_marker"]).size().to_string())
    print(f"run_tag={run_tag}")
    print(f"jobs={len(jobs)} stages={args.stages}")

    if args.status_only:
        return

    for job in jobs:
        for stage in args.stages:
            print("=" * 110)
            print(f"[{stage}] {job.model} {job.layer}")
            print(f"job_dir={job.job_dir}")
            if args.dry_run:
                print("[dry-run] would run")
                continue
            run_one_stage(job, stage, args, run_tag)

    st2 = status(jobs, args.stages, run_tag)
    st2.to_csv(Path(args.out_root) / f"status_{run_tag}.csv", index=False)
    print("\nDONE")
    print(st2.groupby(["stage", "complete", "failed_marker"]).size().to_string())

if __name__ == "__main__":
    main()
