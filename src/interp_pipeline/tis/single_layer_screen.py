from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

DEFAULT_PROJECT_ROOT = Path("/maiziezhou_lab2/yunfei/Projects/interpTFM")
DEFAULT_ADATA = "/maiziezhou_lab2/yunfei/Projects/FM_temp/interGFM/ge_shards/cosmx_human_lung_sec8.h5ad"


@dataclass(frozen=True)
class ModelSpec:
    label: str
    store_root: str
    layers: List[str]
    pooling: str
    token_value: Optional[str]


@dataclass(frozen=True)
class LayerJob:
    idx: int
    model: str
    layer: str
    store_root: str
    pooling: str
    token_value: Optional[str]
    out_dir: str


DEFAULT_MODELS: Dict[str, ModelSpec] = {
    "scgpt": ModelSpec(
        label="scgpt",
        store_root="/maiziezhou_lab2/yunfei/Projects/interpTFM/acts_extraction_scgpt_cosmx",
        layers=[f"layer_{i}.norm2" for i in range(12)],
        pooling="token",
        token_value="60695",
    ),
    "c2sscale": ModelSpec(
        label="c2sscale",
        store_root="/maiziezhou_lab2/yunfei/Projects/interpTFM/acts_extraction_c2s_cosmx_store",
        layers=["layer_0", "layer_6", "layer_13", "layer_15", "layer_17", "layer_19", "layer_21", "layer_23", "layer_25"],
        pooling="mean",
        token_value=None,
    ),
    "geneformer": ModelSpec(
        label="geneformer",
        store_root="/maiziezhou_lab2/yunfei/Projects/interpTFM/acts_extraction_geneformer_cosmx",
        layers=["layer_1", "layer_4", "layer_7", "layer_10", "layer_12", "layer_14", "layer_16", "layer_17"],
        pooling="token",
        token_value="",
    ),
}


def layer_tag(layer: str) -> str:
    return str(layer).replace("/", "_").replace(".", "_")


def parse_csv_or_space(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    out = [x.strip() for x in str(value).replace(",", " ").split() if x.strip()]
    return out or None


def clone_default_models() -> Dict[str, ModelSpec]:
    return {
        k: ModelSpec(
            label=v.label,
            store_root=v.store_root,
            layers=list(v.layers),
            pooling=v.pooling,
            token_value=v.token_value,
        )
        for k, v in DEFAULT_MODELS.items()
    }


def apply_model_overrides(
    models: Dict[str, ModelSpec],
    *,
    scgpt_store_root: Optional[str] = None,
    c2sscale_store_root: Optional[str] = None,
    geneformer_store_root: Optional[str] = None,
    scgpt_layers: Optional[str] = None,
    c2sscale_layers: Optional[str] = None,
    geneformer_layers: Optional[str] = None,
    scgpt_token_value: Optional[str] = None,
    geneformer_token_value: Optional[str] = None,
) -> Dict[str, ModelSpec]:
    patches = {
        "scgpt": {
            "store_root": scgpt_store_root,
            "layers": parse_csv_or_space(scgpt_layers),
            "token_value": scgpt_token_value,
        },
        "c2sscale": {
            "store_root": c2sscale_store_root,
            "layers": parse_csv_or_space(c2sscale_layers),
            "token_value": None,
        },
        "geneformer": {
            "store_root": geneformer_store_root,
            "layers": parse_csv_or_space(geneformer_layers),
            "token_value": geneformer_token_value,
        },
    }

    out = dict(models)
    for name, patch in patches.items():
        if name not in out:
            continue
        old = out[name]
        out[name] = ModelSpec(
            label=old.label,
            store_root=patch["store_root"] or old.store_root,
            layers=patch["layers"] or list(old.layers),
            pooling=old.pooling,
            token_value=old.token_value if patch["token_value"] is None else patch["token_value"],
        )
    return out


def load_backend(backend_script: str):
    backend_script = Path(backend_script)
    if not backend_script.exists():
        raise FileNotFoundError(f"Backend TIS script not found: {backend_script}")
    spec = importlib.util.spec_from_file_location("_interp_tis3_backend", str(backend_script))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load backend module from {backend_script}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    required = ["run_tis_one_model", "TISConfig", "parse_optional_int", "sc"]
    missing = [name for name in required if not hasattr(mod, name)]
    if missing:
        raise RuntimeError(f"Backend script missing required objects: {missing}")
    return mod


def build_jobs(out_root: str, selected_models: Sequence[str], model_specs: Dict[str, ModelSpec]) -> List[LayerJob]:
    out_root_p = Path(out_root)
    jobs: List[LayerJob] = []
    idx = 0
    for model in selected_models:
        if model not in model_specs:
            raise ValueError(f"Unknown model={model!r}. Valid models: {sorted(model_specs)}")
        spec = model_specs[model]
        for layer in spec.layers:
            idx += 1
            jobs.append(
                LayerJob(
                    idx=idx,
                    model=spec.label,
                    layer=layer,
                    store_root=spec.store_root,
                    pooling=spec.pooling,
                    token_value=spec.token_value,
                    out_dir=str(out_root_p / spec.label / layer_tag(layer)),
                )
            )
    return jobs


def summary_path(job: LayerJob) -> Path:
    return Path(job.out_dir) / "summary_row.json"


def done_path(job: LayerJob) -> Path:
    return Path(job.out_dir) / ".tis_done.json"


def failed_path(job: LayerJob) -> Path:
    return Path(job.out_dir) / ".tis_failed.json"


def command_path(job: LayerJob) -> Path:
    return Path(job.out_dir) / ".tis_job.json"


def is_complete(job: LayerJob) -> bool:
    p = summary_path(job)
    if not p.exists():
        return False
    try:
        row = json.loads(p.read_text())
    except Exception:
        return False
    return str(row.get("model")) == job.model and str(row.get("layer")) == job.layer


def status_for_job(job: LayerJob) -> str:
    if is_complete(job):
        return "complete"
    if failed_path(job).exists() or summary_path(job).exists():
        return "incomplete"
    return "pending"


def write_manifest(out_root: str, jobs: Sequence[LayerJob]) -> None:
    p = Path(out_root)
    p.mkdir(parents=True, exist_ok=True)
    (p / "manifest.json").write_text(json.dumps([asdict(j) for j in jobs], indent=2))


def load_summary_rows(out_root: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for p in sorted(Path(out_root).glob("*/*/summary_row.json")):
        try:
            row = json.loads(p.read_text())
            row["summary_path"] = str(p)
            row["out_dir"] = str(p.parent)
            rows.append(row)
        except Exception as e:
            print(f"[warn] could not read {p}: {e}")
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "tis_mean" in df.columns and "shuffle_mean" in df.columns:
        df["tis_gap_mean"] = df["tis_mean"].astype(float) - df["shuffle_mean"].astype(float)
    if "tis_median" in df.columns and "shuffle_median" in df.columns:
        df["tis_gap_median"] = df["tis_median"].astype(float) - df["shuffle_median"].astype(float)
    return df.drop_duplicates(subset=["model", "layer"], keep="last")


def save_current_summary(out_root: str) -> None:
    df = load_summary_rows(out_root)
    if df.empty:
        return
    out_root_p = Path(out_root)
    df.to_csv(out_root_p / "tis_layer_screen_summary.csv", index=False)
    metric = "tis_gap_mean" if "tis_gap_mean" in df.columns else "tis_mean"
    ranked = df.sort_values(["model", metric], ascending=[True, False])
    ranked.to_csv(out_root_p / "tis_layer_screen_ranked.csv", index=False)


def print_status(jobs: Sequence[LayerJob], selected: Sequence[LayerJob]) -> None:
    counts = {"complete": 0, "incomplete": 0, "pending": 0}
    for job in jobs:
        counts[status_for_job(job)] += 1
    print("Resume summary")
    print(f" complete: {counts['complete']}")
    print(f" incomplete: {counts['incomplete']}")
    print(f" pending: {counts['pending']}")
    print(f" run_now: {len(selected)}")


def run_one_job(
    *,
    backend,
    job: LayerJob,
    adata,
    max_shards: Optional[int],
    example_id_key: str,
    exclude_tokens: Optional[List[str]],
    judge_mode: str,
    K: int,
    n_trials: int,
    seed: int,
    q_high: float,
    q_low: float,
    subsample_eval: Optional[int],
    no_pca: bool,
    save_cell_activations: bool,
) -> Dict[str, Any]:
    out_dir = Path(job.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    command_path(job).write_text(json.dumps(asdict(job), indent=2))
    if failed_path(job).exists():
        failed_path(job).unlink()
    cfg = backend.TISConfig(
        K=int(K),
        n_trials=int(n_trials),
        seed=int(seed),
        q_high=float(q_high),
        q_low=float(q_low),
        exclude_query_from_pools=True,
        subsample_eval=subsample_eval,
    )
    row = backend.run_tis_one_model(
        label=job.model,
        store_root=job.store_root,
        layer=job.layer,
        pooling=job.pooling,
        token_value=job.token_value,
        adata=adata,
        out_dir=out_dir,
        max_shards=max_shards,
        example_id_key=example_id_key,
        exclude_tokens=exclude_tokens,
        judge_mode=judge_mode,
        cfg=cfg,
        save_cell_activations=save_cell_activations,
        run_pca=not no_pca,
    )
    done_path(job).write_text(json.dumps({"status": "complete", "job": asdict(job)}, indent=2))
    return row


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one TIS job per model/layer with resume support.")
    p.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    p.add_argument("--backend-script", default="scripts/test_tis_3models.py")
    p.add_argument("--out-root", default="runs/tis_layer_screen_3models_cosmx")
    p.add_argument("--adata-path", default=DEFAULT_ADATA)
    p.add_argument("--models", nargs="+", default=["scgpt", "c2sscale", "geneformer"])
    p.add_argument("--only-model", choices=sorted(DEFAULT_MODELS), default=None)
    p.add_argument("--only-layer", default=None)

    p.add_argument("--scgpt-store-root", default=None)
    p.add_argument("--c2sscale-store-root", default=None)
    p.add_argument("--geneformer-store-root", default=None)
    p.add_argument("--scgpt-layers", default=None, help="Comma/space-separated scGPT layers.")
    p.add_argument("--c2sscale-layers", default=None, help="Comma/space-separated C2S layers.")
    p.add_argument("--geneformer-layers", default=None, help="Comma/space-separated Geneformer layers.")
    p.add_argument("--scgpt-token-value", default=None)
    p.add_argument("--geneformer-token-value", default=None)

    p.add_argument("--max-shards", default="NONE")
    p.add_argument("--example-id-key", default="example_ids")
    p.add_argument("--exclude-tokens", nargs="*", default=["", ""])
    p.add_argument("--judge-mode", default="log1p_cp10k")
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--n-trials", type=int, default=800)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--q-high", type=float, default=0.75)
    p.add_argument("--q-low", type=float, default=0.25)
    p.add_argument("--subsample-eval", type=int, default=None)
    p.add_argument("--no-pca", action="store_true")
    p.add_argument("--save-cell-activations", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--status-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--stop-after", type=int, default=None)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    project_root = Path(args.project_root)
    backend_script = Path(args.backend_script)
    if not backend_script.is_absolute():
        backend_script = project_root / backend_script
    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = project_root / out_root

    model_specs = apply_model_overrides(
        clone_default_models(),
        scgpt_store_root=args.scgpt_store_root,
        c2sscale_store_root=args.c2sscale_store_root,
        geneformer_store_root=args.geneformer_store_root,
        scgpt_layers=args.scgpt_layers,
        c2sscale_layers=args.c2sscale_layers,
        geneformer_layers=args.geneformer_layers,
        scgpt_token_value=args.scgpt_token_value,
        geneformer_token_value=args.geneformer_token_value,
    )

    models = [args.only_model] if args.only_model else args.models
    jobs = build_jobs(str(out_root), models, model_specs=model_specs)
    if args.only_layer:
        jobs = [j for j in jobs if j.layer == args.only_layer]
    write_manifest(str(out_root), jobs)

    selected = []
    for job in jobs:
        if args.force or status_for_job(job) != "complete":
            selected.append(job)
    if args.stop_after is not None:
        selected = selected[: int(args.stop_after)]

    print_status(jobs, selected)
    print(f"Saved manifest: {out_root / 'manifest.json'}")
    for job in selected:
        print(f"[todo] {job.idx:03d} {job.model} {job.layer} status={status_for_job(job)} out={job.out_dir}")

    if args.status_only or args.dry_run:
        if args.dry_run:
            print("DRY RUN: no jobs executed")
        save_current_summary(str(out_root))
        print("DONE")
        return

    if not selected:
        save_current_summary(str(out_root))
        print("DONE")
        return

    backend = load_backend(str(backend_script))
    max_shards = backend.parse_optional_int(args.max_shards)
    print("[load] AnnData:", args.adata_path)
    adata = backend.sc.read_h5ad(args.adata_path)
    print(" n_obs:", adata.n_obs, "n_vars:", adata.n_vars)

    for i, job in enumerate(selected, start=1):
        print("=" * 120)
        print(f"[job {i}/{len(selected)}] {job.model} {job.layer}")
        try:
            run_one_job(
                backend=backend,
                job=job,
                adata=adata,
                max_shards=max_shards,
                example_id_key=args.example_id_key,
                exclude_tokens=args.exclude_tokens,
                judge_mode=args.judge_mode,
                K=args.K,
                n_trials=args.n_trials,
                seed=args.seed,
                q_high=args.q_high,
                q_low=args.q_low,
                subsample_eval=args.subsample_eval,
                no_pca=args.no_pca,
                save_cell_activations=args.save_cell_activations,
            )
            save_current_summary(str(out_root))
        except Exception as e:
            Path(job.out_dir).mkdir(parents=True, exist_ok=True)
            failed_path(job).write_text(
                json.dumps(
                    {
                        "status": "failed",
                        "job": asdict(job),
                        "error": repr(e),
                        "traceback": traceback.format_exc(),
                    },
                    indent=2,
                )
            )
            save_current_summary(str(out_root))
            raise

    save_current_summary(str(out_root))
    print("DONE")


if __name__ == "__main__":
    main()
