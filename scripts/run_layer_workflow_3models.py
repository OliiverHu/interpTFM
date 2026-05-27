#!/usr/bin/env python3
from __future__ import annotations

"""
Lean consolidated 3-model layer workflow.

Stage order:
  extract
  tis_pre_sae
  sae_train_or_check
  activation_qc
  f1_heldout
  go_reduce
  f1_qc
  build_adata
  niche_sweep
  niche_validation
  niche_terms
  shuffle_control_crosstalk
  grouped_heatmaps
  immune_followups

Design choices:
  * No standalone max1_norm_qc stage.
  * TIS is before SAE training.
  * Plain test_crosstalk_3models.py is not part of the workflow.
  * Grouped heatmaps come from shuffle-control crosstalk output by default.
  * Boundary/asymmetry needed by immune follow-ups is computed inside
    test_immune_infiltration_followups_3models.py from h5ads, not from old crosstalk CSVs.
"""

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


STAGE_ORDER = [
    "extract",
    "tis_pre_sae",
    "sae_train_or_check",
    "activation_qc",
    "f1_heldout",
    "go_reduce",
    "f1_qc",
    "build_adata",
    "niche_sweep",
    "niche_validation",
    "niche_terms",
    "shuffle_control_crosstalk",
    "grouped_heatmaps",
    "immune_followups",
]


def die(msg: str) -> None:
    raise SystemExit(f"[ERROR] {msg}")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_config(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    text = p.read_text()
    if p.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as e:
            die(f"YAML config needs PyYAML installed, or use JSON. Import error: {e}")
        cfg = yaml.safe_load(text)
    else:
        cfg = json.loads(text)
    if not isinstance(cfg, dict):
        die("Config must parse to a dictionary.")
    cfg["_config_path"] = str(p.resolve())
    return cfg


def q(x: str | Path) -> str:
    return shlex.quote(str(x))


def fmt_cmd(cmd: Sequence[str]) -> str:
    return " ".join(q(str(x)) for x in cmd)


def flatten_kv(extra: Mapping[str, Any]) -> List[str]:
    out: List[str] = []
    for key, val in extra.items():
        flag = "--" + str(key).replace("_", "-")
        if val is None or val is False:
            continue
        if val is True:
            out.append(flag)
        elif isinstance(val, list):
            if len(val) == 0:
                continue
            out.append(flag)
            out.extend(str(v) for v in val)
        else:
            out.extend([flag, str(val)])
    return out


def run_cmd(cmd: Sequence[str], *, cwd: Path, dry_run: bool, log_file: Optional[Path] = None, env: Optional[Dict[str, str]] = None) -> None:
    print("\n[CMD]")
    print(fmt_cmd(cmd))
    if log_file is not None:
        ensure_dir(log_file.parent)
        with open(log_file, "a") as f:
            f.write("\n[CMD]\n")
            f.write(fmt_cmd(cmd) + "\n")
    if dry_run:
        return

    full_env = os.environ.copy()
    if env:
        full_env.update({str(k): str(v) for k, v in env.items()})

    if log_file is None:
        subprocess.run(list(cmd), cwd=str(cwd), check=True, env=full_env)
    else:
        with open(log_file, "a") as f:
            p = subprocess.run(list(cmd), cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT, text=True, env=full_env)
        if p.returncode != 0:
            raise subprocess.CalledProcessError(p.returncode, list(cmd))


def run_shell(command: str, *, cwd: Path, dry_run: bool, log_file: Optional[Path] = None) -> None:
    run_cmd(["bash", "-lc", command], cwd=cwd, dry_run=dry_run, log_file=log_file)


def labels(cfg: Dict[str, Any]) -> List[str]:
    x = [str(v) for v in cfg.get("labels", ["scgpt", "c2sscale", "geneformer"])]
    if len(x) != 3:
        die("This wrapper expects exactly 3 model labels.")
    return x


def layer_set_name(layer_set: Dict[str, Any]) -> str:
    if layer_set.get("name"):
        return str(layer_set["name"])
    return "__".join(str(v).replace("/", "_") for _, v in sorted(layer_set["layers"].items()))


def layer_for(layer_set: Dict[str, Any], label: str) -> str:
    if label not in layer_set.get("layers", {}):
        die(f"Layer set {layer_set_name(layer_set)} missing layer for {label}")
    return str(layer_set["layers"][label])


def layers_for(layer_set: Dict[str, Any], labs: Sequence[str]) -> List[str]:
    return [layer_for(layer_set, x) for x in labs]


def model_cfg(cfg: Dict[str, Any], label: str) -> Dict[str, Any]:
    if label not in cfg.get("models", {}):
        die(f"Missing models.{label}")
    return cfg["models"][label]


def store_roots(cfg: Dict[str, Any], layer_set: Dict[str, Any], labs: Sequence[str]) -> List[str]:
    out = []
    for lab in labs:
        val = layer_set.get("store_roots", {}).get(lab, model_cfg(cfg, lab).get("store_root"))
        if val is None:
            die(f"Missing store_root for {lab}")
        out.append(str(val))
    return out


def render_template(template: str, *, cfg: Dict[str, Any], layer_set: Dict[str, Any], label: str, layer: str) -> str:
    mc = model_cfg(cfg, label)
    ls_name = layer_set_name(layer_set)
    ctx: Dict[str, Any] = {
        "project_root": cfg.get("project_root", ""),
        "runs_root": cfg.get("runs_root", ""),
        "label": label,
        "layer": layer,
        "layer_safe": layer.replace("/", "_"),
        "layer_set": ls_name,
        "store_root": layer_set.get("store_roots", {}).get(label, mc.get("store_root", "")),
        "model_root": mc.get("model_root", mc.get("store_root", "")),
        "sae_run_tag": cfg.get("sae", {}).get("run_tag", ""),
    }
    for k, v in mc.items():
        if isinstance(v, (str, int, float)):
            ctx[f"model_{k}"] = v
    return str(template).format(**ctx)


def script_path(cfg: Dict[str, Any], key: str) -> str:
    if key in cfg.get("scripts", {}):
        return str(cfg["scripts"][key])
    scripts_dir = Path(cfg.get("scripts_dir", "scripts"))
    defaults = {
        "tis_pre_sae": "test_tis_3models.py",
        "tis_seed_grid": "test_tis_hp_search_3models.py",
        "sae_train": "test_train_3saes.py",
        "sae_launch": "launch_train_3_saes.py",
        "activation_qc": "audit_dead_neurons_3models.py",
        "f1_heldout": "test_run_f1heldout_3models.py",
        "f1_heldout_inner": "run_f1heldout_single_backend.py",
        "go_reduce": "test_go_reduce_3models.py",
        "f1_analysis": "test_downstream_f1_3models.py",
        "f1_latents": "test_f1_latents_analysis_3models.py",
        "f1_acts": "test_f1_acts_analysis_3models.py",
        "f1_downstream": "test_downstream_f1_3models.py",
        "plot_latents_vs_acts": "plot_f1_latents_vs_acts_3models.py",
        "build_adata": "test_build_interpretable_adata_3models.py",
        "niche_sweep": "test_niche_sweep_3models.py",
        "niche_validation": "test_niche_validation_3models.py",
        "niche_terms": "test_analyze_niche_terms_and_consistency_3models.py",
        "shuffle_control_crosstalk": "test_crosstalk_shuffle_control_3models.py",
        "immune_followups": "test_immune_infiltration_followups_3models.py",
    }
    if key not in defaults:
        die(f"No script mapping for key={key!r}")
    return str(scripts_dir / defaults[key])


def resolve_script(cfg: Dict[str, Any], key: str, cwd: Path) -> str:
    s = Path(script_path(cfg, key))
    for p in [s, cwd / s, cwd / "scripts" / s.name]:
        if p.exists():
            return str(p)
    return str(s)


def stage_root(cfg: Dict[str, Any], stage: str, ls_name: str) -> Path:
    root = Path(cfg.get("runs_root", "runs/layer_workflow_3models"))
    custom = cfg.get(stage, {}).get("out_root")
    if custom:
        return Path(str(custom).format(runs_root=str(root), layer_set=ls_name))
    return root / stage / ls_name


def status_path(cfg: Dict[str, Any], ls_name: str) -> Path:
    return Path(cfg.get("runs_root", "runs/layer_workflow_3models")) / "workflow_status" / f"{ls_name}.json"


def load_status(cfg: Dict[str, Any], ls_name: str) -> Dict[str, Any]:
    p = status_path(cfg, ls_name)
    if p.exists():
        return json.loads(p.read_text())
    return {"layer_set": ls_name, "stages": {}}


def save_status(cfg: Dict[str, Any], ls_name: str, status: Dict[str, Any]) -> None:
    p = status_path(cfg, ls_name)
    ensure_dir(p.parent)
    p.write_text(json.dumps(status, indent=2))


def done(status: Dict[str, Any], stage: str) -> bool:
    return bool(status.get("stages", {}).get(stage, {}).get("done", False))


def mark(cfg: Dict[str, Any], ls_name: str, status: Dict[str, Any], stage: str, payload: Dict[str, Any]) -> None:
    status.setdefault("stages", {})[stage] = {"done": True, **payload}
    save_status(cfg, ls_name, status)


def sae_base_out_dirs(cfg: Dict[str, Any], layer_set: Dict[str, Any], labs: Sequence[str]) -> List[str]:
    out = []
    sae_cfg = cfg.get("sae", {})
    for lab in labs:
        lyr = layer_for(layer_set, lab)
        direct = layer_set.get("sae_base_out_dirs", {}).get(lab)
        if direct:
            out.append(str(direct))
            continue
        tmpl = model_cfg(cfg, lab).get("sae_base_out_dir_template") or sae_cfg.get("base_out_dir_template")
        if tmpl:
            out.append(render_template(str(tmpl), cfg=cfg, layer_set=layer_set, label=lab, layer=lyr))
        else:
            out.append(str(Path(store_roots(cfg, layer_set, [lab])[0]) / "sae" / lyr))
    return out


def fmt_float_tag(x: float) -> str:
    s = f"{x:.0e}" if float(x) < 1e-2 else str(x)
    return s.replace("+0", "").replace("+", "").replace("-0", "-").replace(".", "p")


def sae_run_tag(cfg: Dict[str, Any]) -> str:
    sae = cfg.get("sae", {})
    if sae.get("run_tag"):
        return str(sae["run_tag"])
    no_resample = bool(sae.get("no_resample", True))
    steps = int(sae.get("steps", 6000))
    l1 = float(sae.get("l1", 3e-3))
    return f"nr_{'on' if no_resample else 'off'}__steps_{steps}__l1_{fmt_float_tag(l1)}"


def sae_ckpts(cfg: Dict[str, Any], layer_set: Dict[str, Any], labs: Sequence[str]) -> List[str]:
    if layer_set.get("sae_ckpts"):
        return [str(layer_set["sae_ckpts"][lab]) for lab in labs]
    out = []
    tag = sae_run_tag(cfg)
    for lab, base, lyr in zip(labs, sae_base_out_dirs(cfg, layer_set, labs), layers_for(layer_set, labs)):
        tmpl = model_cfg(cfg, lab).get("sae_ckpt_template") or cfg.get("sae", {}).get("ckpt_template")
        if tmpl:
            out.append(render_template(str(tmpl), cfg=cfg, layer_set=layer_set, label=lab, layer=lyr))
        else:
            out.append(str(Path(base) / tag / f"sae_{lyr}_best.pt"))
    return out


def check_paths(paths: Sequence[str], what: str, require: bool = True) -> bool:
    ok = True
    for p in paths:
        exists = Path(p).exists()
        print(f"[check] {what}: {p} -> {'OK' if exists else 'MISSING'}")
        ok = ok and exists
    if require and not ok:
        die(f"Missing required {what}.")
    return ok


def csv_has_columns(path: Path, required_groups: List[List[str]]) -> bool:
    try:
        with open(path, newline="") as f:
            header = next(csv.reader(f))
    except Exception:
        return False
    cols = set(header)
    return all(any(c in cols for c in group) for group in required_groups)


def find_f1_table(f1_root: Path, label: str, layer: str) -> Optional[Path]:
    bases = [f1_root / label / layer, f1_root / label / layer.replace("/", "_"), f1_root / label]
    required = [["concept", "term_id", "concept_id", "native"], ["feature", "latent", "feature_id", "latent_id"], ["f1", "best_f1", "F1"]]
    candidates: List[Path] = []
    for base in bases:
        if not base.exists():
            continue
        for p in base.rglob("*.csv"):
            name = p.name.lower()
            if "diagnostic" in name or "duplicate" in name or "gt_" in name:
                continue
            if csv_has_columns(p, required):
                candidates.append(p)
    if not candidates:
        return None

    def score(p: Path) -> Tuple[int, int, str]:
        n = p.name.lower()
        s = sum(kw in n for kw in ["best", "summary", "f1", "heldout", "alignment", "test_concept"])
        return int(s), int(p.stat().st_size), str(p)
    return sorted(candidates, key=score, reverse=True)[0]


def resolve_f1_tables(cfg: Dict[str, Any], layer_set: Dict[str, Any], labs: Sequence[str], status: Dict[str, Any]) -> List[str]:
    if layer_set.get("f1_tables"):
        return [str(layer_set["f1_tables"][lab]) for lab in labs]
    f1_root = Path(status.get("stages", {}).get("f1_heldout", {}).get("out_root", stage_root(cfg, "f1_heldout", layer_set_name(layer_set))))
    out: List[str] = []
    for lab, lyr in zip(labs, layers_for(layer_set, labs)):
        hit = find_f1_table(f1_root, lab, lyr)
        if hit is None:
            die(f"Could not infer F1 table for {lab}/{lyr} under {f1_root}. Add layer_set.f1_tables.")
        out.append(str(hit))
    return out


def heldout_roots(cfg: Dict[str, Any], layer_set: Dict[str, Any], labs: Sequence[str], status: Dict[str, Any]) -> List[str]:
    if layer_set.get("heldout_roots"):
        return [str(layer_set["heldout_roots"][lab]) for lab in labs]
    root = Path(status.get("stages", {}).get("f1_heldout", {}).get("out_root", stage_root(cfg, "f1_heldout", layer_set_name(layer_set))))
    out = []
    for lab, lyr in zip(labs, layers_for(layer_set, labs)):
        for p in [root / lab / lyr, root / lab / lyr.replace("/", "_"), root / lab]:
            if p.exists():
                out.append(str(p))
                break
        else:
            out.append(str(root / lab))
    return out


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def find_interpretable_h5ads(build_root: Path, labs: Sequence[str], lyrs: Sequence[str]) -> List[str]:
    out = []
    for lab, lyr in zip(labs, lyrs):
        bases = [build_root / lab / lyr.replace("/", "_"), build_root / lab / lyr, build_root / lab]
        found = None
        for base in bases:
            if not base.exists():
                continue
            for summary in sorted(base.rglob("summary.json")):
                obj = read_json(summary)
                if obj and obj.get("h5ad_path") and Path(obj["h5ad_path"]).exists():
                    found = Path(obj["h5ad_path"])
                    break
            if found is None:
                hits = sorted(base.rglob("adata_interpretable_*.h5ad"))
                if hits:
                    found = hits[-1]
            if found is not None:
                break
        if found is None:
            die(f"Could not find interpretable h5ad for {lab}/{lyr} under {build_root}")
        out.append(str(found))
    return out


def resolve_interp_h5ads(cfg: Dict[str, Any], layer_set: Dict[str, Any], labs: Sequence[str], status: Dict[str, Any]) -> List[str]:
    if layer_set.get("interp_h5ads"):
        return [str(layer_set["interp_h5ads"][lab]) for lab in labs]
    p = status.get("stages", {}).get("build_adata", {}).get("interp_h5ads")
    if p:
        return [str(x) for x in p]
    return find_interpretable_h5ads(stage_root(cfg, "build_adata", layer_set_name(layer_set)), labs, layers_for(layer_set, labs))


def find_validation_dirs(root: Path, labs: Sequence[str], lyrs: Sequence[str]) -> List[str]:
    out = []
    for lab, lyr in zip(labs, lyrs):
        bases = [root / lab / lyr.replace("/", "_"), root / lab / lyr, root / lab]
        hits: List[Path] = []
        for base in bases:
            if base.exists():
                hits.extend(sorted(base.rglob("validation_summary.json")))
        if not hits:
            die(f"Could not find validation_summary.json for {lab}/{lyr} under {root}")
        out.append(str(hits[-1].parent))
    return out


def resolve_validation_dirs(cfg: Dict[str, Any], layer_set: Dict[str, Any], labs: Sequence[str], status: Dict[str, Any]) -> List[str]:
    if layer_set.get("validation_dirs"):
        return [str(layer_set["validation_dirs"][lab]) for lab in labs]
    p = status.get("stages", {}).get("niche_validation", {}).get("validation_dirs")
    if p:
        return [str(x) for x in p]
    return find_validation_dirs(stage_root(cfg, "niche_validation", layer_set_name(layer_set)), labs, layers_for(layer_set, labs))


def resolve_niche_h5ads(cfg: Dict[str, Any], layer_set: Dict[str, Any], labs: Sequence[str], status: Dict[str, Any]) -> List[str]:
    if layer_set.get("niche_h5ads"):
        return [str(layer_set["niche_h5ads"][lab]) for lab in labs]
    p = status.get("stages", {}).get("niche_validation", {}).get("niche_h5ads")
    if p:
        return [str(x) for x in p]
    return [str(Path(d) / "adata_with_niche_labels.h5ad") for d in resolve_validation_dirs(cfg, layer_set, labs, status)]


def format_custom_command(command: str, cfg: Dict[str, Any], layer_set: Dict[str, Any], labs: Sequence[str], status: Dict[str, Any]) -> str:
    ls = layer_set_name(layer_set)
    ctx: Dict[str, Any] = {
        "project_root": cfg.get("project_root", ""),
        "runs_root": cfg.get("runs_root", ""),
        "layer_set": ls,
        "labels": " ".join(labs),
        "layers": " ".join(layers_for(layer_set, labs)),
        "store_roots": " ".join(store_roots(cfg, layer_set, labs)),
        "sae_ckpts": " ".join(sae_ckpts(cfg, layer_set, labs)),
        "f1_tables": " ".join(resolve_f1_tables(cfg, layer_set, labs, status)) if status.get("stages", {}).get("f1_heldout") else "",
        "heldout_roots": " ".join(heldout_roots(cfg, layer_set, labs, status)) if status.get("stages", {}).get("f1_heldout") else "",
        "interp_h5ads": " ".join(resolve_interp_h5ads(cfg, layer_set, labs, status)) if status.get("stages", {}).get("build_adata") else "",
        "niche_h5ads": " ".join(resolve_niche_h5ads(cfg, layer_set, labs, status)) if status.get("stages", {}).get("niche_validation") else "",
    }
    return command.format(**ctx)


# ---------------- stages ----------------

def stage_extract(cfg, layer_set, labs, cwd, dry, log, force, status):
    ls = layer_set_name(layer_set)
    if done(status, "extract") and not force:
        print(f"[skip] extract already done: {ls}")
        return
    ecfg = cfg.get("extract", {})
    mode = ecfg.get("mode", "check")
    if mode == "skip":
        mark(cfg, ls, status, "extract", {"skipped": True})
        return

    for lab in labs:
        lyr = layer_for(layer_set, lab)
        cmd = layer_set.get("extract_commands", {}).get(lab) or ecfg.get("commands", {}).get(lab) or model_cfg(cfg, lab).get("extract_command")
        if cmd:
            run_shell(render_template(str(cmd), cfg=cfg, layer_set=layer_set, label=lab, layer=lyr), cwd=cwd, dry_run=dry, log_file=log)
        else:
            act_dir = Path(store_roots(cfg, layer_set, [lab])[0]) / "activations" / lyr
            print(f"[extract/check] {lab} {lyr}: {act_dir}")
            if ecfg.get("require_existing", True) and not act_dir.exists():
                die(f"Activation dir missing: {act_dir}. Add extract.commands.{lab} or run extraction first.")
    mark(cfg, ls, status, "extract", {"mode": mode})


def stage_tis_pre_sae(cfg, layer_set, labs, cwd, dry, log, force, status):
    ls = layer_set_name(layer_set)
    if done(status, "tis_pre_sae") and not force:
        print(f"[skip] tis_pre_sae already done: {ls}")
        return
    tcfg = cfg.get("tis_pre_sae", {})
    if not tcfg.get("enabled", True):
        mark(cfg, ls, status, "tis_pre_sae", {"skipped": True})
        return
    root = stage_root(cfg, "tis_pre_sae", ls)
    cmd = [
        sys.executable, resolve_script(cfg, "tis_pre_sae", cwd),
        "--labels", *labs,
        "--store-roots", *store_roots(cfg, layer_set, labs),
        "--layers", *layers_for(layer_set, labs),
        "--pooling", *[str(tcfg.get("pooling", {}).get(lab, model_cfg(cfg, lab).get("tis_pooling", "mean"))) for lab in labs],
        "--token-values", *[str(tcfg.get("token_values", {}).get(lab, model_cfg(cfg, lab).get("tis_token_value", "NONE"))) for lab in labs],
        "--adata-path", str(cfg["adata_path"]),
        "--out-root", str(root),
    ]
    cmd += flatten_kv(tcfg.get("extra_args", {}))
    run_cmd(cmd, cwd=cwd, dry_run=dry, log_file=log)

    if tcfg.get("seed_grid_enabled", False):
        sg_root = stage_root(cfg, "tis_seed_grid", ls)
        cmd2 = [
            sys.executable, resolve_script(cfg, "tis_seed_grid", cwd),
            "--labels", *labs,
            "--store-roots", *store_roots(cfg, layer_set, labs),
            "--layers", *layers_for(layer_set, labs),
            "--pooling", *[str(tcfg.get("pooling", {}).get(lab, model_cfg(cfg, lab).get("tis_pooling", "mean"))) for lab in labs],
            "--token-values", *[str(tcfg.get("token_values", {}).get(lab, model_cfg(cfg, lab).get("tis_token_value", "NONE"))) for lab in labs],
            "--adata-path", str(cfg["adata_path"]),
            "--out-root", str(sg_root),
        ]
        cmd2 += flatten_kv(tcfg.get("seed_grid_extra_args", {}))
        run_cmd(cmd2, cwd=cwd, dry_run=dry, log_file=log)
    mark(cfg, ls, status, "tis_pre_sae", {"out_root": str(root)})


def stage_sae_train_or_check(cfg, layer_set, labs, cwd, dry, log, force, status):
    ls = layer_set_name(layer_set)
    if done(status, "sae_train_or_check") and not force:
        print(f"[skip] sae_train_or_check already done: {ls}")
        return
    scfg = cfg.get("sae", {})
    mode = scfg.get("mode", "check")
    ckpts = sae_ckpts(cfg, layer_set, labs)

    if mode == "check":
        check_paths(ckpts, "SAE checkpoint", require=True)
        mark(cfg, ls, status, "sae_train_or_check", {"mode": mode, "sae_ckpts": ckpts})
        return

    if mode in {"train", "train_if_missing"}:
        if mode == "train_if_missing" and all(Path(p).exists() for p in ckpts):
            print("[sae] all checkpoints already exist; not training")
            mark(cfg, ls, status, "sae_train_or_check", {"mode": "check_existing", "sae_ckpts": ckpts})
            return

        # Use direct 3-SAE trainer by default so output dirs are deterministic.
        tag = sae_run_tag(cfg)
        base_dirs = sae_base_out_dirs(cfg, layer_set, labs)
        out_dirs = [str(Path(b) / tag) for b in base_dirs]
        cmd = [
            sys.executable, resolve_script(cfg, "sae_train", cwd),
            "--labels", *labs,
            "--store-roots", *store_roots(cfg, layer_set, labs),
            "--layers", *layers_for(layer_set, labs),
            "--out-dirs", *out_dirs,
            "--device", str(cfg.get("device", "cuda")),
            "--batch-size", str(scfg.get("batch_size", 1024)),
            "--latent-multiplier", str(scfg.get("latent_multiplier", 8)),
            "--l1", str(scfg.get("l1", 3e-3)),
            "--lr", str(scfg.get("lr", 1e-4)),
            "--steps", str(scfg.get("steps", 6000)),
            "--warmup-steps", str(scfg.get("warmup_steps", 1000)),
            "--resample-steps", str(scfg.get("resample_steps", 2000)),
            "--seed", str(scfg.get("seed", 0)),
            "--best-metric", str(scfg.get("best_metric", "loss")),
            "--save-every", str(scfg.get("save_every", 0)),
        ]
        if scfg.get("n_latents"):
            cmd += ["--n-latents", *[str(x) for x in scfg["n_latents"]]]
        if scfg.get("no_resample", True):
            cmd.append("--no-resample")
        cmd += flatten_kv(scfg.get("extra_args", {}))
        run_cmd(cmd, cwd=cwd, dry_run=dry, log_file=log)
        if not dry:
            check_paths(ckpts, "SAE checkpoint", require=True)
        mark(cfg, ls, status, "sae_train_or_check", {"mode": mode, "sae_ckpts": ckpts, "out_dirs": out_dirs})
        return

    die(f"Unknown sae.mode={mode!r}. Use check, train_if_missing, or train.")


def stage_activation_qc(cfg, layer_set, labs, cwd, dry, log, force, status):
    ls = layer_set_name(layer_set)
    if done(status, "activation_qc") and not force:
        print(f"[skip] activation_qc already done: {ls}")
        return
    qcfg = cfg.get("activation_qc", {})
    if not qcfg.get("enabled", True):
        mark(cfg, ls, status, "activation_qc", {"skipped": True})
        return
    root = stage_root(cfg, "activation_qc", ls)
    cmd = [
        sys.executable, resolve_script(cfg, "activation_qc", cwd),
        "--labels", *labs,
        "--sae-ckpts", *sae_ckpts(cfg, layer_set, labs),
        "--store-roots", *store_roots(cfg, layer_set, labs),
        "--layers", *layers_for(layer_set, labs),
        "--out-dir", str(root),
    ]
    cmd += flatten_kv(qcfg.get("extra_args", {}))
    run_cmd(cmd, cwd=cwd, dry_run=dry, log_file=log)
    mark(cfg, ls, status, "activation_qc", {"out_dir": str(root)})


def stage_f1_heldout(cfg, layer_set, labs, cwd, dry, log, force, status):
    ls = layer_set_name(layer_set)
    if done(status, "f1_heldout") and not force:
        print(f"[skip] f1_heldout already done: {ls}")
        return
    fcfg = cfg.get("f1_heldout", {})
    root = stage_root(cfg, "f1_heldout", ls)
    cmd = [
        sys.executable, resolve_script(cfg, "f1_heldout", cwd),
        "--heldout-script", str(fcfg.get("heldout_script", resolve_script(cfg, "f1_heldout_inner", cwd))),
        "--labels", *labs,
        "--store-roots", *store_roots(cfg, layer_set, labs),
        "--layers", *layers_for(layer_set, labs),
        "--sae-ckpts", *sae_ckpts(cfg, layer_set, labs),
        "--adata-path", str(cfg["adata_path"]),
        "--out-root", str(root),
        "--device", str(cfg.get("device", "cuda")),
    ]
    if fcfg.get("scgpt_foundation_ckpt"):
        cmd += ["--scgpt-foundation-ckpt", str(fcfg["scgpt_foundation_ckpt"])]
    cmd += flatten_kv(fcfg.get("extra_args", {}))
    run_cmd(cmd, cwd=cwd, dry_run=dry, log_file=log)
    payload = {"out_root": str(root)}
    if not dry:
        payload["f1_tables"] = resolve_f1_tables(cfg, layer_set, labs, {"stages": {"f1_heldout": {"out_root": str(root)}}})
    mark(cfg, ls, status, "f1_heldout", payload)


def stage_go_reduce(cfg, layer_set, labs, cwd, dry, log, force, status):
    ls = layer_set_name(layer_set)
    if done(status, "go_reduce") and not force:
        print(f"[skip] go_reduce already done: {ls}")
        return
    gcfg = cfg.get("go_reduce", {})
    if not gcfg.get("enabled", True):
        mark(cfg, ls, status, "go_reduce", {"skipped": True})
        return
    root = stage_root(cfg, "go_reduce", ls)
    cmd = [
        sys.executable, resolve_script(cfg, "go_reduce", cwd),
        "--labels", *labs,
        "--layers", *layers_for(layer_set, labs),
        "--f1-tables", *resolve_f1_tables(cfg, layer_set, labs, status),
        "--out-root", str(root),
        "--go-obo-path", str(gcfg.get("go_obo_path", "resources/go-basic.obo")),
    ]
    cmd += flatten_kv(gcfg.get("extra_args", {}))
    run_cmd(cmd, cwd=cwd, dry_run=dry, log_file=log)
    mark(cfg, ls, status, "go_reduce", {"out_root": str(root)})


def stage_f1_qc(cfg, layer_set, labs, cwd, dry, log, force, status):
    ls = layer_set_name(layer_set)
    if done(status, "f1_qc") and not force:
        print(f"[skip] f1_qc already done: {ls}")
        return
    qcfg = cfg.get("f1_qc", {})
    if not qcfg.get("enabled", True):
        mark(cfg, ls, status, "f1_qc", {"skipped": True})
        return
    root = stage_root(cfg, "f1_qc", ls)
    ensure_dir(root)

    # Canonical compact summary. Other legacy QC scripts can be added via custom_commands.
    if qcfg.get("run_f1_analysis", True):
        cmd = [
            sys.executable, resolve_script(cfg, "f1_analysis", cwd),
            "--labels", *labs,
            "--heldout-roots", *heldout_roots(cfg, layer_set, labs, status),
            "--out-dir", str(root / "f1_analysis"),
        ]
        if qcfg.get("term_meta_path"):
            cmd += ["--term-meta-path", str(qcfg["term_meta_path"])]
        cmd += flatten_kv(qcfg.get("f1_analysis_extra_args", {}))
        run_cmd(cmd, cwd=cwd, dry_run=dry, log_file=log)

    for i, command in enumerate(qcfg.get("custom_commands", []), start=1):
        print(f"[f1_qc custom {i}]")
        run_shell(format_custom_command(str(command), cfg, layer_set, labs, status), cwd=cwd, dry_run=dry, log_file=log)

    mark(cfg, ls, status, "f1_qc", {"out_root": str(root)})


def stage_build_adata(cfg, layer_set, labs, cwd, dry, log, force, status):
    ls = layer_set_name(layer_set)
    if done(status, "build_adata") and not force:
        print(f"[skip] build_adata already done: {ls}")
        return
    bcfg = cfg.get("build_adata", {})
    root = stage_root(cfg, "build_adata", ls)
    cmd = [
        sys.executable, resolve_script(cfg, "build_adata", cwd),
        "--labels", *labs,
        "--store-roots", *store_roots(cfg, layer_set, labs),
        "--layers", *layers_for(layer_set, labs),
        "--sae-ckpts", *sae_ckpts(cfg, layer_set, labs),
        "--f1-tables", *resolve_f1_tables(cfg, layer_set, labs, status),
        "--adata-path", str(cfg["adata_path"]),
        "--out-root", str(root),
        "--threshold", str(bcfg.get("threshold", 0.15)),
        "--f1-min", str(bcfg.get("f1_min", 0.4)),
        "--min-true-pos", str(bcfg.get("min_true_pos", 3)),
        "--cell-agg", str(bcfg.get("cell_agg", "mean")),
        "--device", str(cfg.get("device", "cuda")),
        "--cell-type-col", str(cfg.get("celltype_col", "author_cell_type")),
    ]
    if bcfg.get("top_concepts") is not None:
        cmd += ["--top-concepts", str(bcfg["top_concepts"])]
    if bcfg.get("term_meta"):
        cmd += ["--term-meta", str(bcfg["term_meta"])]
    excl = [str(model_cfg(cfg, lab).get("exclude_token_value", "NONE")) for lab in labs]
    if any(x != "NONE" for x in excl):
        cmd += ["--exclude-token-values", *excl]
    cmd += flatten_kv(bcfg.get("extra_args", {}))
    run_cmd(cmd, cwd=cwd, dry_run=dry, log_file=log)
    payload = {"out_root": str(root)}
    if not dry:
        payload["interp_h5ads"] = find_interpretable_h5ads(root, labs, layers_for(layer_set, labs))
    mark(cfg, ls, status, "build_adata", payload)


def stage_niche_sweep(cfg, layer_set, labs, cwd, dry, log, force, status):
    ls = layer_set_name(layer_set)
    if done(status, "niche_sweep") and not force:
        print(f"[skip] niche_sweep already done: {ls}")
        return
    ncfg = cfg.get("niche_sweep", {})
    if not ncfg.get("enabled", False):
        mark(cfg, ls, status, "niche_sweep", {"skipped": True})
        return
    root = stage_root(cfg, "niche_sweep", ls)
    cmd = [
        sys.executable, resolve_script(cfg, "niche_sweep", cwd),
        "--labels", *labs,
        "--layers", *layers_for(layer_set, labs),
        "--interp-h5ads", *resolve_interp_h5ads(cfg, layer_set, labs, status),
        "--out-root", str(root),
    ]
    cmd += flatten_kv(ncfg.get("extra_args", {}))
    run_cmd(cmd, cwd=cwd, dry_run=dry, log_file=log)
    mark(cfg, ls, status, "niche_sweep", {"out_root": str(root)})


def stage_niche_validation(cfg, layer_set, labs, cwd, dry, log, force, status):
    ls = layer_set_name(layer_set)
    if done(status, "niche_validation") and not force:
        print(f"[skip] niche_validation already done: {ls}")
        return
    ncfg = cfg.get("niche_validation", {})
    root = stage_root(cfg, "niche_validation", ls)
    cmd = [
        sys.executable, resolve_script(cfg, "niche_validation", cwd),
        "--labels", *labs,
        "--layers", *layers_for(layer_set, labs),
        "--interp-h5ads", *resolve_interp_h5ads(cfg, layer_set, labs, status),
        "--out-root", str(root),
        "--celltype-col", str(cfg.get("celltype_col", "author_cell_type")),
        "--radius", str(ncfg.get("radius", 120)),
        "--space", str(ncfg.get("space", "xm")),
        "--kernel", str(ncfg.get("kernel", "uniform")),
        "--method", str(ncfg.get("method", "gmm")),
        "--n-clusters", str(ncfg.get("n_clusters", 3)),
        "--seed", str(ncfg.get("seed", 0)),
        "--relabel-by-celltype", str(ncfg.get("relabel_by_celltype", "tumor 13")),
    ]
    if ncfg.get("configs_csv") or layer_set.get("niche_configs_csv"):
        cmd += ["--configs-csv", str(ncfg.get("configs_csv") or layer_set.get("niche_configs_csv"))]
    cmd += flatten_kv(ncfg.get("extra_args", {}))
    run_cmd(cmd, cwd=cwd, dry_run=dry, log_file=log)
    payload = {"out_root": str(root)}
    if not dry:
        dirs = find_validation_dirs(root, labs, layers_for(layer_set, labs))
        payload["validation_dirs"] = dirs
        payload["niche_h5ads"] = [str(Path(d) / "adata_with_niche_labels.h5ad") for d in dirs]
    mark(cfg, ls, status, "niche_validation", payload)


def stage_niche_terms(cfg, layer_set, labs, cwd, dry, log, force, status):
    ls = layer_set_name(layer_set)
    if done(status, "niche_terms") and not force:
        print(f"[skip] niche_terms already done: {ls}")
        return
    ncfg = cfg.get("niche_terms", {})
    if not ncfg.get("enabled", True):
        mark(cfg, ls, status, "niche_terms", {"skipped": True})
        return
    root = stage_root(cfg, "niche_terms", ls)
    cmd = [
        sys.executable, resolve_script(cfg, "niche_terms", cwd),
        "--labels", *labs,
        "--validation-dirs", *resolve_validation_dirs(cfg, layer_set, labs, status),
        "--out-dir", str(root),
    ]
    cmd += flatten_kv(ncfg.get("extra_args", {}))
    run_cmd(cmd, cwd=cwd, dry_run=dry, log_file=log)
    mark(cfg, ls, status, "niche_terms", {"out_dir": str(root)})


def stage_shuffle_control_crosstalk(cfg, layer_set, labs, cwd, dry, log, force, status):
    ls = layer_set_name(layer_set)
    if done(status, "shuffle_control_crosstalk") and not force:
        print(f"[skip] shuffle_control_crosstalk already done: {ls}")
        return
    ccfg = cfg.get("shuffle_control_crosstalk", {})
    if not ccfg.get("enabled", True):
        mark(cfg, ls, status, "shuffle_control_crosstalk", {"skipped": True})
        return
    root = stage_root(cfg, "shuffle_control_crosstalk", ls)
    cmd = [
        sys.executable, resolve_script(cfg, "shuffle_control_crosstalk", cwd),
        "--labels", *labs,
        "--h5ads", *resolve_niche_h5ads(cfg, layer_set, labs, status),
        "--celltype-group-csv", str(cfg["celltype_group_csv"]),
        "--target-interaction-csv", str(cfg["target_interaction_csv"]),
        "--out-root", str(root),
        "--celltype-col", str(cfg.get("celltype_col", "author_cell_type")),
        "--niche-col", str(cfg.get("niche_col", "niche")),
        "--spatial-key", str(cfg.get("spatial_key", "spatial")),
        "--edge-radius", str(ccfg.get("edge_radius", 120)),
        "--tile-size", str(ccfg.get("tile_size", 400)),
        "--n-perm", str(ccfg.get("n_perm", 500)),
        "--min-edges", str(ccfg.get("min_edges", 50)),
    ]
    if ccfg.get("domains"):
        cmd += ["--domains", *[str(x) for x in ccfg["domains"]]]
    cmd += flatten_kv(ccfg.get("extra_args", {}))
    run_cmd(cmd, cwd=cwd, dry_run=dry, log_file=log)
    mark(cfg, ls, status, "shuffle_control_crosstalk", {
        "out_root": str(root),
        "combined_grouped_csv": str(root / "combined_grouped_crosstalk_shuffle_control.csv"),
        "target_axes_csv": str(root / "combined_target_axes_shuffle_control.csv"),
        "plots_dir": str(root / "plots"),
    })


def stage_grouped_heatmaps(cfg, layer_set, labs, cwd, dry, log, force, status):
    ls = layer_set_name(layer_set)
    if done(status, "grouped_heatmaps") and not force:
        print(f"[skip] grouped_heatmaps already done: {ls}")
        return
    hcfg = cfg.get("grouped_heatmaps", {})
    shuffle = status.get("stages", {}).get("shuffle_control_crosstalk", {})
    if not hcfg.get("enabled", True):
        mark(cfg, ls, status, "grouped_heatmaps", {"skipped": True})
        return

    # By default, shuffle-control script already creates grouped heatmaps.
    if not hcfg.get("custom_commands"):
        print("[grouped_heatmaps] using heatmaps already written by shuffle_control_crosstalk")
        mark(cfg, ls, status, "grouped_heatmaps", {"plots_dir": shuffle.get("plots_dir")})
        return

    for i, command in enumerate(hcfg.get("custom_commands", []), start=1):
        print(f"[grouped_heatmaps custom {i}]")
        run_shell(format_custom_command(str(command), cfg, layer_set, labs, status), cwd=cwd, dry_run=dry, log_file=log)
    mark(cfg, ls, status, "grouped_heatmaps", {"custom": True})


def stage_immune_followups(cfg, layer_set, labs, cwd, dry, log, force, status):
    ls = layer_set_name(layer_set)
    if done(status, "immune_followups") and not force:
        print(f"[skip] immune_followups already done: {ls}")
        return
    icfg = cfg.get("immune_followups", {})
    if not icfg.get("enabled", True):
        mark(cfg, ls, status, "immune_followups", {"skipped": True})
        return
    root = stage_root(cfg, "immune_followups", ls)
    shuffle_csv = status.get("stages", {}).get("shuffle_control_crosstalk", {}).get(
        "combined_grouped_csv",
        str(stage_root(cfg, "shuffle_control_crosstalk", ls) / "combined_grouped_crosstalk_shuffle_control.csv"),
    )
    cmd = [
        sys.executable, resolve_script(cfg, "immune_followups", cwd),
        "--labels", *labs,
        "--h5ads", *resolve_niche_h5ads(cfg, layer_set, labs, status),
        "--celltype-group-csv", str(cfg["celltype_group_csv"]),
        "--target-interaction-csv", str(cfg["target_interaction_csv"]),
        "--shuffle-control-csv", str(shuffle_csv),
        "--out-root", str(root),
        "--celltype-col", str(cfg.get("celltype_col", "author_cell_type")),
        "--niche-col", str(cfg.get("niche_col", "niche")),
        "--spatial-key", str(cfg.get("spatial_key", "spatial")),
        "--edge-radius", str(icfg.get("edge_radius", 120)),
    ]
    if icfg.get("tumor_groups"):
        cmd += ["--tumor-groups", *[str(x) for x in icfg["tumor_groups"]]]
    if icfg.get("immune_groups"):
        cmd += ["--immune-groups", *[str(x) for x in icfg["immune_groups"]]]
    if icfg.get("stromal_endothelial_groups"):
        cmd += ["--stromal-endothelial-groups", *[str(x) for x in icfg["stromal_endothelial_groups"]]]
    cmd += flatten_kv(icfg.get("extra_args", {}))
    run_cmd(cmd, cwd=cwd, dry_run=dry, log_file=log)
    mark(cfg, ls, status, "immune_followups", {"out_root": str(root)})


STAGE_FUNCS = {
    "extract": stage_extract,
    "tis_pre_sae": stage_tis_pre_sae,
    "sae_train_or_check": stage_sae_train_or_check,
    "activation_qc": stage_activation_qc,
    "f1_heldout": stage_f1_heldout,
    "go_reduce": stage_go_reduce,
    "f1_qc": stage_f1_qc,
    "build_adata": stage_build_adata,
    "niche_sweep": stage_niche_sweep,
    "niche_validation": stage_niche_validation,
    "niche_terms": stage_niche_terms,
    "shuffle_control_crosstalk": stage_shuffle_control_crosstalk,
    "grouped_heatmaps": stage_grouped_heatmaps,
    "immune_followups": stage_immune_followups,
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Lean consolidated 3-model layer workflow.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--stages", nargs="+", default=STAGE_ORDER)
    ap.add_argument("--skip-stages", nargs="*", default=[])
    ap.add_argument("--only-layer-set", nargs="*", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-stop-on-error", dest="stop_on_error", action="store_false")
    ap.set_defaults(stop_on_error=True)
    ap.add_argument("--no-log-to-file", dest="log_to_file", action="store_false")
    ap.set_defaults(log_to_file=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cwd = Path(cfg.get("project_root", ".")).resolve()
    labs = labels(cfg)

    stages = [s for s in args.stages if s not in set(args.skip_stages)]
    for s in stages:
        if s not in STAGE_FUNCS:
            die(f"Unknown stage {s!r}. Valid stages: {list(STAGE_FUNCS)}")

    layer_sets = list(cfg.get("layer_sets", []))
    if not layer_sets:
        die("Config needs layer_sets.")
    if args.only_layer_set:
        keep = set(args.only_layer_set)
        layer_sets = [x for x in layer_sets if layer_set_name(x) in keep]
        if not layer_sets:
            die(f"No layer_sets matched {sorted(keep)}")

    print("[workflow]")
    print("  cwd:", cwd)
    print("  labels:", labs)
    print("  stages:", stages)
    print("  layer_sets:", [layer_set_name(x) for x in layer_sets])
    print("  dry_run:", args.dry_run)

    for layer_set in layer_sets:
        ls = layer_set_name(layer_set)
        print("\n" + "#" * 120)
        print("# LAYER SET:", ls)
        print("#" * 120)

        status = load_status(cfg, ls)
        log = None
        if args.log_to_file:
            log = Path(cfg.get("runs_root", "runs/layer_workflow_3models")) / "logs" / f"{ls}.log"

        for stage in stages:
            print("\n" + "=" * 100)
            print(f"[stage] {stage} | {ls}")
            print("=" * 100)
            try:
                STAGE_FUNCS[stage](cfg, layer_set, labs, cwd, args.dry_run, log, args.force, status)
                status = load_status(cfg, ls)
            except Exception as e:
                print(f"[FAILED] stage={stage} layer_set={ls}: {e}")
                status.setdefault("stages", {})[stage] = {"done": False, "error": str(e)}
                save_status(cfg, ls, status)
                if args.stop_on_error:
                    raise

    print("\n[OK] workflow finished")


if __name__ == "__main__":
    main()



