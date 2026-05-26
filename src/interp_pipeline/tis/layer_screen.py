from __future__ import annotations

import csv
import json
import shutil
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

DEFAULT_ADATA = "/maiziezhou_lab2/yunfei/Projects/FM_temp/interGFM/ge_shards/cosmx_human_lung_sec8.h5ad"
DEFAULT_BACKEND = "scripts/test_tis_3models.py"

DEFAULT_MODELS: Dict[str, Dict[str, Any]] = {
    "scgpt": {
        "store_root": "/maiziezhou_lab2/yunfei/Projects/interpTFM/acts_extraction_scgpt_cosmx",
        "layers": [
            "layer_0.norm2", "layer_1.norm2", "layer_2.norm2", "layer_3.norm2",
            "layer_4.norm2", "layer_5.norm2", "layer_6.norm2", "layer_7.norm2",
            "layer_8.norm2", "layer_9.norm2", "layer_10.norm2", "layer_11.norm2",
        ],
        "pooling": "token",
        "token_value": "60695",
    },
    "c2sscale": {
        "store_root": "/maiziezhou_lab2/yunfei/Projects/interpTFM/acts_extraction_c2s_cosmx_store",
        "layers": ["layer_0", "layer_6", "layer_13", "layer_15", "layer_17", "layer_19", "layer_21", "layer_23", "layer_25"],
        "pooling": "mean",
        "token_value": "NONE",
    },
    "geneformer": {
        "store_root": "/maiziezhou_lab2/yunfei/Projects/interpTFM/acts_extraction_geneformer_cosmx",
        "layers": ["layer_1", "layer_4", "layer_7", "layer_10", "layer_12", "layer_14", "layer_16", "layer_17"],
        "pooling": "token",
        "token_value": "<cls>",
    },
}


@dataclass(frozen=True)
class TISLayerTask:
    model: str
    layer: str


@dataclass(frozen=True)
class TISLayerScreenParams:
    out_root: str = "runs/tis_layer_screen_3models_cosmx"
    adata_path: str = DEFAULT_ADATA
    backend: str = DEFAULT_BACKEND
    max_shards: str = "NONE"
    K: str = "32"
    n_trials: str = "800"
    seed: str = "42"
    q_high: str = "0.75"
    q_low: str = "0.25"
    no_pca: bool = False
    save_cell_activations: bool = False
    force: bool = False
    dry_run: bool = False
    status_only: bool = False
    retry_incomplete: bool = True
    clean_incomplete: bool = False


def parse_layer_csv(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    out = [x.strip() for x in value.replace(",", " ").split() if x.strip()]
    return out or None


def load_model_config(path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    models = json.loads(json.dumps(DEFAULT_MODELS))
    if path is None:
        return models
    data = json.loads(Path(path).read_text())
    for model, patch in data.get("models", data).items():
        if model not in models:
            models[model] = {}
        models[model].update(patch)
    return models


def apply_layer_overrides(
    models: Dict[str, Dict[str, Any]],
    scgpt_layers: Optional[Sequence[str]] = None,
    c2sscale_layers: Optional[Sequence[str]] = None,
    geneformer_layers: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    models = json.loads(json.dumps(models))
    overrides = {
        "scgpt": list(scgpt_layers) if scgpt_layers is not None else None,
        "c2sscale": list(c2sscale_layers) if c2sscale_layers is not None else None,
        "geneformer": list(geneformer_layers) if geneformer_layers is not None else None,
    }
    for model, layers in overrides.items():
        if layers is not None:
            models.setdefault(model, {})["layers"] = layers
    return models


def chunks3(items: Sequence[TISLayerTask]) -> Iterable[List[TISLayerTask]]:
    for i in range(0, len(items), 3):
        chunk = list(items[i:i + 3])
        while len(chunk) < 3:
            chunk.append(chunk[-1])
        yield chunk


def make_task_list(models: Dict[str, Dict[str, Any]], selected_models: Sequence[str]) -> List[TISLayerTask]:
    tasks: List[TISLayerTask] = []
    seen = set()
    for model in selected_models:
        if model not in models:
            raise ValueError(f"Unknown model {model!r}; available: {sorted(models)}")
        for layer in models[model]["layers"]:
            key = (model, layer)
            if key in seen:
                continue
            seen.add(key)
            tasks.append(TISLayerTask(model=model, layer=layer))
    return tasks


def batch_tag(batch: Sequence[TISLayerTask]) -> str:
    return "__".join(f"{x.model}_{x.layer.replace('.', '_')}" for x in batch)


def task_key(task: TISLayerTask) -> Tuple[str, str]:
    return (task.model, task.layer)


def unique_task_keys(batch: Sequence[TISLayerTask]) -> Set[Tuple[str, str]]:
    return {task_key(x) for x in batch}


def read_summary_keys(summary_csv: Path) -> Set[Tuple[str, str]]:
    if not summary_csv.exists() or summary_csv.stat().st_size == 0:
        return set()
    keys: Set[Tuple[str, str]] = set()
    with summary_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return set()
        for row in reader:
            model = str(row.get("model", "")).strip()
            layer = str(row.get("layer", "")).strip()
            if model and layer:
                keys.add((model, layer))
    return keys


def done_marker(out_dir: Path) -> Path:
    return out_dir / ".tis_done.json"


def fail_marker(out_dir: Path) -> Path:
    return out_dir / ".tis_failed.json"


def summary_csv(out_dir: Path) -> Path:
    return out_dir / "tis_summary_3models.csv"


def batch_is_complete(out_dir: Path, batch: Sequence[TISLayerTask]) -> bool:
    expected = unique_task_keys(batch)
    observed = read_summary_keys(summary_csv(out_dir))
    if expected and expected.issubset(observed):
        return True
    # Backward compatibility for previous completed runs that only had a summary file.
    # This is intentionally weaker and only used when the old summary has rows.
    return False


def batch_status(out_dir: Path, batch: Sequence[TISLayerTask]) -> Dict[str, Any]:
    expected = unique_task_keys(batch)
    observed = read_summary_keys(summary_csv(out_dir))
    complete = expected.issubset(observed)
    return {
        "out_dir": str(out_dir),
        "expected": sorted([{"model": m, "layer": l} for m, l in expected], key=lambda x: (x["model"], x["layer"])),
        "observed": sorted([{"model": m, "layer": l} for m, l in observed], key=lambda x: (x["model"], x["layer"])),
        "missing": sorted([{"model": m, "layer": l} for m, l in expected - observed], key=lambda x: (x["model"], x["layer"])),
        "complete": complete,
        "summary_exists": summary_csv(out_dir).exists(),
        "done_marker_exists": done_marker(out_dir).exists(),
        "failed_marker_exists": fail_marker(out_dir).exists(),
    }


def build_backend_command(
    batch: Sequence[TISLayerTask],
    models: Dict[str, Dict[str, Any]],
    params: TISLayerScreenParams,
    out_dir: Path,
) -> List[str]:
    labels = [x.model for x in batch]
    layers = [x.layer for x in batch]
    roots = [models[m]["store_root"] for m in labels]
    pooling = [models[m]["pooling"] for m in labels]
    tokens = [models[m]["token_value"] for m in labels]

    cmd = [
        "python", params.backend,
        "--labels", *labels,
        "--store-roots", *roots,
        "--layers", *layers,
        "--pooling", *pooling,
        "--token-values", *tokens,
        "--adata-path", params.adata_path,
        "--out-root", str(out_dir),
        "--max-shards", str(params.max_shards),
        "--K", str(params.K),
        "--n-trials", str(params.n_trials),
        "--seed", str(params.seed),
        "--q-high", str(params.q_high),
        "--q-low", str(params.q_low),
    ]
    if params.no_pca:
        cmd.append("--no-pca")
    if params.save_cell_activations:
        cmd.append("--save-cell-activations")
    return cmd


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def run_layer_screen(
    models: Dict[str, Dict[str, Any]],
    selected_models: Sequence[str],
    params: TISLayerScreenParams,
) -> Path:
    out_root = Path(params.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    tasks = make_task_list(models, selected_models)
    batches = list(chunks3(tasks))
    manifest: List[Dict[str, Any]] = []

    print("=" * 100)
    print("interp_pipeline.tis.layer_screen")
    print(f"backend    = {params.backend}")
    print(f"out_root   = {out_root}")
    print(f"n_tasks    = {len(tasks)}")
    print(f"n_batches  = {len(batches)}")
    print(f"force      = {params.force}")
    print(f"status_only= {params.status_only}")
    print("=" * 100)

    n_complete = 0
    n_incomplete = 0
    n_pending = 0
    n_run = 0

    for batch_idx, batch in enumerate(batches):
        tag = batch_tag(batch)
        out_dir = out_root / f"batch_{batch_idx:03d}__{tag}"
        cmd = build_backend_command(batch, models=models, params=params, out_dir=out_dir)
        status = batch_status(out_dir, batch)

        manifest_row = {
            "batch_idx": batch_idx,
            "out_dir": str(out_dir),
            "items": [asdict(x) for x in batch],
            "unique_items": [{"model": m, "layer": l} for m, l in sorted(unique_task_keys(batch))],
            "command": cmd,
            "status": status,
        }
        manifest.append(manifest_row)

        if status["complete"] and not params.force:
            n_complete += 1
            print(f"[skip complete] batch {batch_idx:03d}: {tag}")
            if not done_marker(out_dir).exists() and not params.dry_run and not params.status_only:
                write_json(done_marker(out_dir), {"batch_idx": batch_idx, "completed_at": time.time(), "status": status})
            continue

        if status["summary_exists"] and not status["complete"]:
            n_incomplete += 1
            print(f"[incomplete] batch {batch_idx:03d}: {tag}")
            print(f"  missing: {status['missing']}")
            if params.status_only:
                continue
            if not params.retry_incomplete and not params.force:
                continue
            if params.clean_incomplete and out_dir.exists() and not params.dry_run:
                backup = out_dir.with_name(out_dir.name + f".incomplete_{int(time.time())}")
                print(f"  moving incomplete directory to: {backup}")
                shutil.move(str(out_dir), str(backup))
        else:
            n_pending += 1
            if params.status_only:
                print(f"[pending] batch {batch_idx:03d}: {tag}")
                continue

        print("-" * 100)
        print(f"[run] batch {batch_idx:03d}: {tag}")
        print(" ".join(cmd))
        if params.dry_run or params.status_only:
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(out_dir / ".tis_command.json", {"batch_idx": batch_idx, "command": cmd, "items": [asdict(x) for x in batch]})
        fail = fail_marker(out_dir)
        if fail.exists():
            fail.unlink()
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            write_json(fail, {
                "batch_idx": batch_idx,
                "returncode": e.returncode,
                "failed_at": time.time(),
                "command": cmd,
                "items": [asdict(x) for x in batch],
            })
            raise

        post_status = batch_status(out_dir, batch)
        if not post_status["complete"]:
            write_json(fail, {
                "batch_idx": batch_idx,
                "returncode": 0,
                "failed_at": time.time(),
                "reason": "backend exited successfully but expected rows were not found in tis_summary_3models.csv",
                "status": post_status,
                "command": cmd,
            })
            raise RuntimeError(f"Batch {batch_idx} finished but is incomplete: {post_status['missing']}")
        write_json(done_marker(out_dir), {"batch_idx": batch_idx, "completed_at": time.time(), "status": post_status})
        n_run += 1

    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print("\nResume summary")
    print(f"  complete:   {n_complete}")
    print(f"  incomplete: {n_incomplete}")
    print(f"  pending:    {n_pending}")
    print(f"  run_now:    {n_run}")
    print(f"Saved manifest: {manifest_path}")
    print("DONE")
    return manifest_path
