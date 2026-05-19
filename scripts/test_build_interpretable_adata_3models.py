#!/usr/bin/env python3
from __future__ import annotations

"""
Build interpretable cell-level AnnData from SAE features and check Leiden-vs-celltype ARI.

For each model:
  1. Load original AnnData for obs/spatial.
  2. Load token-level activations from ActivationStore shards.
  3. Load SAE checkpoint.
  4. Load selected F1 concept-feature associations.
  5. Compute selected SAE latent activations on token rows.
  6. Aggregate selected latent activations to cell level by example_id.
  7. Write AnnData with X = cells x selected interpretable features/concepts.
  8. Run Leiden on this X and compute ARI vs a cell type column.

Default token exclusions:
  scGPT: 60694,60695
  c2s-scale: none
  Geneformer: <cls>,<eos>
"""

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def parse_optional_int(x: str | None) -> Optional[int]:
    if x is None or str(x).strip().lower() in {"none", "null", "na", ""}:
        return None
    return int(x)


def parse_exclude_tokens(x: str | None) -> List[str]:
    if x is None or str(x).strip().lower() in {"none", "null", "na", ""}:
        return []
    return [p.strip() for p in str(x).split(",") if p.strip()]


def default_exclude_tokens(label: str) -> List[str]:
    lab = label.lower()
    if lab in {"scgpt", "sc-gpt"}:
        return ["60694", "60695"]
    if lab == "geneformer":
        return ["<cls>", "<eos>"]
    return []


def infer_col(df: pd.DataFrame, candidates: Sequence[str], required: bool = True) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"Could not infer column from {candidates}. Have columns={list(df.columns)}")
    return None


def load_term_meta(term_meta_path: Optional[str]) -> pd.DataFrame:
    if term_meta_path is None or str(term_meta_path).lower() in {"none", "null", ""}:
        return pd.DataFrame()
    p = Path(term_meta_path)
    sep = "\t" if p.suffix.lower() in {".tsv", ".txt"} else ","
    return pd.read_csv(p, sep=sep)


def select_f1_features(
    f1_table: str,
    *,
    term_meta: pd.DataFrame,
    threshold: Optional[float],
    f1_min: float,
    min_true_pos: Optional[int],
    top_concepts: Optional[int],
    keep_duplicate_features: bool,
) -> pd.DataFrame:
    df = pd.read_csv(f1_table)
    concept_col = infer_col(df, ["concept", "term_id", "concept_id", "native"])
    feature_col = infer_col(df, ["feature", "latent", "feature_id", "latent_id"])
    f1_col = infer_col(df, ["f1", "best_f1", "F1"])
    thr_col = infer_col(df, ["threshold_pct", "threshold", "thr", "latent_threshold"], required=False)
    true_pos_col = infer_col(df, ["true_pos", "n_true", "true", "tp"], required=False)

    out = pd.DataFrame({
        "concept": df[concept_col].astype(str),
        "feature": pd.to_numeric(df[feature_col], errors="coerce"),
        "f1": pd.to_numeric(df[f1_col], errors="coerce"),
    })
    out["threshold"] = pd.to_numeric(df[thr_col], errors="coerce") if thr_col else np.nan
    out["true_pos"] = pd.to_numeric(df[true_pos_col], errors="coerce") if true_pos_col else np.nan

    for c in ["precision", "recall", "pred_pos", "tp", "fp", "fn"]:
        if c in df.columns:
            out[c] = pd.to_numeric(df[c], errors="coerce")

    out = out.dropna(subset=["feature", "f1"]).copy()
    out["feature"] = out["feature"].astype(int)

    if threshold is not None and out["threshold"].notna().any():
        out = out[np.isclose(out["threshold"].astype(float), float(threshold))].copy()
    out = out[out["f1"] >= float(f1_min)].copy()
    if min_true_pos is not None and out["true_pos"].notna().any():
        out = out[out["true_pos"] >= int(min_true_pos)].copy()

    # Best feature per concept.
    out = out.sort_values(["concept", "f1"], ascending=[True, False]).drop_duplicates("concept", keep="first").copy()

    # Avoid duplicate identical columns from same feature unless requested.
    if not keep_duplicate_features:
        out = out.sort_values(["feature", "f1"], ascending=[True, False]).drop_duplicates("feature", keep="first").copy()

    out = out.sort_values("f1", ascending=False).copy()
    if top_concepts is not None:
        out = out.head(int(top_concepts)).copy()

    if not term_meta.empty:
        meta = term_meta.copy()
        if "term_id" in meta.columns:
            out = out.merge(meta, left_on="concept", right_on="term_id", how="left")
        elif "concept" in meta.columns:
            out = out.merge(meta, on="concept", how="left")

    name_col = next((c for c in ["term_name", "name", "description"] if c in out.columns), None)
    names = out[name_col].fillna("").astype(str) if name_col else pd.Series([""] * len(out), index=out.index)

    col_names, seen = [], set()
    for i, row in out.reset_index(drop=True).iterrows():
        concept = str(row["concept"])
        feature = int(row["feature"])
        pretty = str(names.iloc[i]).strip()
        base = f"{pretty}|{concept}|feat{feature}" if pretty else f"{concept}|feat{feature}"
        base = base.replace("/", "_")
        name, j = base, 2
        while name in seen:
            name = f"{base}__{j}"
            j += 1
        seen.add(name)
        col_names.append(name)

    out = out.reset_index(drop=True)
    out["var_name"] = col_names
    return out


def get_state_dict_container(ckpt: dict) -> dict:
    if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]
    if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
        return ckpt["model_state_dict"]
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]
    if all(torch.is_tensor(v) for v in ckpt.values()):
        return ckpt
    raise RuntimeError(f"Could not find state dict. Top-level keys={list(ckpt.keys())[:40]}")


def first_tensor_key(state: dict, keys: Sequence[str]) -> Optional[str]:
    for k in keys:
        if k in state and torch.is_tensor(state[k]):
            return k
    return None


def load_selected_encoder(ckpt_path: str, selected_features: Sequence[int], device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = get_state_dict_container(ckpt)
    w_key = first_tensor_key(state, ["encoder.weight", "W_enc", "enc.weight"])
    if w_key is None:
        raise RuntimeError(f"Could not find encoder weight. Keys={list(state.keys())[:40]}")
    W = state[w_key].detach().float()
    n_lat, d_in = int(W.shape[0]), int(W.shape[1])

    feats = np.asarray(selected_features, dtype=np.int64)
    if feats.min() < 0 or feats.max() >= n_lat:
        raise ValueError(f"Selected feature out of range. n_lat={n_lat}, min={feats.min()}, max={feats.max()}")

    idx = torch.as_tensor(feats, dtype=torch.long)
    W_sel = W[idx].contiguous().to(device)

    b_key = first_tensor_key(state, ["encoder.bias", "b_enc", "enc.bias"])
    b_sel = state[b_key].detach().float()[idx].contiguous().to(device) if b_key else None

    x_bias_key = first_tensor_key(state, ["bias", "x_bias"])
    x_bias = state[x_bias_key].detach().float().contiguous().to(device) if x_bias_key else None
    if x_bias is not None and int(x_bias.numel()) != d_in:
        raise RuntimeError(f"x_bias dim={x_bias.numel()} but d_in={d_in}")

    return W_sel, b_sel, x_bias, d_in, n_lat


def list_shards(store_root: str, layer: str, max_shards: Optional[int]) -> List[str]:
    acts_root = os.path.join(store_root, "activations", layer)
    shards = sorted([p for p in glob.glob(os.path.join(acts_root, "shard_*")) if os.path.isdir(p)])
    if max_shards is not None:
        shards = shards[: int(max_shards)]
    if not shards:
        raise RuntimeError(f"No shards found under {acts_root}")
    return shards


def resolve_example_index(ex_raw: Any, obs_index: Dict[str, int], n_obs: int) -> Optional[int]:
    ex_str = str(ex_raw)
    if ex_str.isdigit():
        i = int(ex_str)
        return i if 0 <= i < n_obs else None
    i = obs_index.get(ex_str)
    return None if i is None else int(i)


@torch.no_grad()
def build_cell_feature_matrix(
    *,
    adata,
    store_root: str,
    layer: str,
    sae_ckpt: str,
    selected_df: pd.DataFrame,
    label: str,
    cell_agg: str,
    exclude_tokens: List[str],
    batch_size: int,
    max_shards: Optional[int],
    device: str,
):
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    obs_names = list(map(str, adata.obs_names.tolist()))
    obs_index = {x: i for i, x in enumerate(obs_names)}
    n_obs = len(obs_names)

    selected_features = selected_df["feature"].astype(int).tolist()
    W_sel, b_sel, x_bias, d_in, n_lat = load_selected_encoder(sae_ckpt, selected_features, dev)
    n_feat = len(selected_features)

    counts = np.zeros(n_obs, dtype=np.int64)
    if cell_agg == "mean":
        X_sum = np.zeros((n_obs, n_feat), dtype=np.float64)
        X_max = None
    elif cell_agg == "max":
        X_sum = None
        X_max = np.full((n_obs, n_feat), -np.inf, dtype=np.float32)
    else:
        raise ValueError("--cell-agg must be mean or max")

    exclude = set(map(str, exclude_tokens))
    shards = list_shards(store_root, layer, max_shards=max_shards)

    n_rows_seen = n_rows_used = n_unmatched = n_excluded_token = 0

    for shard in tqdm(shards, desc=f"{label}:{layer}:SAE->cell"):
        acts_path = os.path.join(shard, "activations.pt")
        idx_path = os.path.join(shard, "index.pt")
        if not os.path.exists(acts_path) or not os.path.exists(idx_path):
            continue

        A = torch.load(acts_path, map_location="cpu")
        idx = torch.load(idx_path, map_location="cpu")
        example_ids = idx.get("example_ids")
        token_ids = idx.get("token_ids", [None] * int(A.shape[0]))

        if example_ids is None:
            continue
        if len(example_ids) != int(A.shape[0]) or len(token_ids) != int(A.shape[0]):
            raise RuntimeError(f"{idx_path}: index length mismatch")
        if int(A.shape[1]) != int(d_in):
            raise RuntimeError(f"{acts_path}: activation dim={A.shape[1]} but SAE d_in={d_in}")

        n_rows_seen += int(A.shape[0])

        row_to_obs, row_ids = [], []
        for r, ex_raw in enumerate(example_ids):
            tok = token_ids[r]
            if tok is not None and str(tok) in exclude:
                n_excluded_token += 1
                continue
            obs_i = resolve_example_index(ex_raw, obs_index, n_obs)
            if obs_i is None:
                n_unmatched += 1
                continue
            row_to_obs.append(obs_i)
            row_ids.append(r)

        for j0 in range(0, len(row_ids), batch_size):
            batch_rows = row_ids[j0:j0 + batch_size]
            if not batch_rows:
                continue
            batch_obs = np.asarray(row_to_obs[j0:j0 + batch_size], dtype=np.int64)
            xb = A[batch_rows].float().to(dev, non_blocking=True)
            if x_bias is not None:
                xb = xb - x_bias.to(device=dev, dtype=xb.dtype)
            z = xb @ W_sel.to(device=dev, dtype=xb.dtype).T
            if b_sel is not None:
                z = z + b_sel.to(device=dev, dtype=xb.dtype)
            z = torch.relu(z).detach().cpu().numpy().astype(np.float32)

            if cell_agg == "mean":
                np.add.at(X_sum, batch_obs, z.astype(np.float64))
                np.add.at(counts, batch_obs, 1)
            else:
                np.maximum.at(X_max, batch_obs, z)
                np.add.at(counts, batch_obs, 1)
            n_rows_used += len(batch_rows)

    has_row = counts > 0
    if cell_agg == "mean":
        X = np.zeros((n_obs, n_feat), dtype=np.float32)
        X[has_row] = (X_sum[has_row] / counts[has_row, None]).astype(np.float32)
    else:
        X = X_max
        X[~np.isfinite(X)] = 0.0
        X = X.astype(np.float32)

    summary = {
        "label": label,
        "store_root": store_root,
        "layer": layer,
        "sae_ckpt": sae_ckpt,
        "n_shards_used": int(len(shards)),
        "n_rows_seen": int(n_rows_seen),
        "n_rows_used": int(n_rows_used),
        "n_unmatched": int(n_unmatched),
        "n_excluded_token": int(n_excluded_token),
        "n_obs": int(n_obs),
        "n_has_row": int(has_row.sum()),
        "d_in": int(d_in),
        "n_latents_total": int(n_lat),
        "n_selected_columns": int(n_feat),
        "cell_agg": cell_agg,
        "exclude_tokens": sorted(exclude),
        "counts_min": int(counts[has_row].min()) if has_row.any() else None,
        "counts_median": float(np.median(counts[has_row])) if has_row.any() else None,
        "counts_max": int(counts[has_row].max()) if has_row.any() else None,
    }
    return X, has_row, summary


def infer_cell_type_col(adata, requested: Optional[str]) -> str:
    if requested and requested.lower() not in {"auto", "none", "null", ""}:
        if requested not in adata.obs.columns:
            raise KeyError(f"{requested!r} not in adata.obs. Columns={list(adata.obs.columns)[:80]}")
        return requested
    candidates = ["cell_type", "celltype", "cell_type_major", "celltype_major", "major_cell_type", "annotation", "annotations", "cell_label", "cell_labels", "celltype_final", "predicted_celltype", "subclass", "class"]
    for c in candidates:
        if c in adata.obs.columns:
            return c
    raise KeyError(f"Could not infer cell-type column. Pass --cell-type-col. Obs columns={list(adata.obs.columns)[:100]}")


def run_leiden_ari(
    adata_int,
    *,
    cell_type_col: str,
    out_dir: Path,
    resolutions: Sequence[float],
    n_neighbors: int,
    n_pcs: int,
    cluster_rep: str,
    scale: bool,
    seed: int,
) -> pd.DataFrame:
    ad = adata_int.copy()
    if scale:
        sc.pp.scale(ad, max_value=10)

    if cluster_rep == "pca":
        n_comps = min(int(n_pcs), ad.n_vars - 1, ad.n_obs - 1)
        if n_comps >= 2:
            sc.pp.pca(ad, n_comps=n_comps, random_state=seed)
            use_rep = "X_pca"
        else:
            ad.obsm["X_use"] = np.asarray(ad.X)
            use_rep = "X_use"
    else:
        ad.obsm["X_use"] = np.asarray(ad.X)
        use_rep = "X_use"

    sc.pp.neighbors(ad, n_neighbors=int(n_neighbors), use_rep=use_rep, random_state=seed)

    y = ad.obs[cell_type_col].astype(str).values
    valid = pd.notna(ad.obs[cell_type_col].values)
    rows = []

    for res in resolutions:
        key = f"leiden_r{str(res).replace('.', 'p')}"
        sc.tl.leiden(ad, resolution=float(res), key_added=key, random_state=seed)
        pred = ad.obs[key].astype(str).values
        rows.append({
            "resolution": float(res),
            "leiden_key": key,
            "ari": float(adjusted_rand_score(y[valid], pred[valid])),
            "n_clusters": int(ad.obs[key].nunique()),
            "cell_type_col": cell_type_col,
            "n_cells": int(valid.sum()),
            "cluster_rep": cluster_rep,
            "n_neighbors": int(n_neighbors),
            "n_pcs": int(n_pcs),
            "scale": bool(scale),
        })
        adata_int.obs[key] = ad.obs[key].values

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "leiden_celltype_ari.csv", index=False)
    return df


def build_one_model(
    *,
    label: str,
    store_root: str,
    layer: str,
    sae_ckpt: str,
    f1_table: str,
    adata,
    term_meta: pd.DataFrame,
    out_root: Path,
    threshold: Optional[float],
    f1_min: float,
    min_true_pos: Optional[int],
    top_concepts: Optional[int],
    keep_duplicate_features: bool,
    cell_agg: str,
    exclude_tokens: List[str],
    batch_size: int,
    max_shards: Optional[int],
    device: str,
    cell_type_col: str,
    resolutions: Sequence[float],
    n_neighbors: int,
    n_pcs: int,
    cluster_rep: str,
    scale: bool,
    seed: int,
) -> Dict[str, Any]:
    out_dir = ensure_dir(out_root / label / layer.replace("/", "_"))
    print("=" * 100)
    print(f"[interpretable AnnData] {label} | {layer}")
    print(f"  out_dir={out_dir}")
    print("=" * 100)

    selected = select_f1_features(
        f1_table,
        term_meta=term_meta,
        threshold=threshold,
        f1_min=f1_min,
        min_true_pos=min_true_pos,
        top_concepts=top_concepts,
        keep_duplicate_features=keep_duplicate_features,
    )
    if selected.empty:
        raise RuntimeError(f"No selected concepts/features for {label}. Try lower --f1-min.")
    selected.to_csv(out_dir / "selected_concept_features.csv", index=False)

    X, has_row, extract_summary = build_cell_feature_matrix(
        adata=adata,
        store_root=store_root,
        layer=layer,
        sae_ckpt=sae_ckpt,
        selected_df=selected,
        label=label,
        cell_agg=cell_agg,
        exclude_tokens=exclude_tokens,
        batch_size=batch_size,
        max_shards=max_shards,
        device=device,
    )

    adata_int = sc.AnnData(
        X=X,
        obs=adata.obs.copy(),
        var=selected.copy(),
        obsm={k: v.copy() for k, v in adata.obsm.items()},
        uns=adata.uns.copy(),
    )
    adata_int.var_names = selected["var_name"].astype(str).values
    adata_int.obs["has_interpretable_row"] = has_row

    adata_cluster = adata_int[has_row].copy()
    ari_df = run_leiden_ari(
        adata_cluster,
        cell_type_col=cell_type_col,
        out_dir=out_dir,
        resolutions=resolutions,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
        cluster_rep=cluster_rep,
        scale=scale,
        seed=seed,
    )

    for key in ari_df["leiden_key"].tolist():
        full = pd.Series(index=adata_int.obs_names, dtype="object")
        full.loc[adata_cluster.obs_names] = adata_cluster.obs[key].astype(str).values
        adata_int.obs[key] = full.values

    h5ad_name = (
        f"adata_interpretable_{layer.replace('/', '_')}"
        f"_saeThr{str(threshold).replace('.', 'p') if threshold is not None else 'all'}"
        f"_f1cut{str(f1_min).replace('.', 'p')}"
        f"_top{top_concepts if top_concepts is not None else 'all'}"
        f"_{cell_agg}.h5ad"
    )
    h5ad_path = out_dir / h5ad_name
    adata_int.write_h5ad(h5ad_path)

    best_ari = ari_df.sort_values("ari", ascending=False).head(1).iloc[0].to_dict()
    summary = {
        "label": label,
        "layer": layer,
        "h5ad_path": str(h5ad_path),
        "n_obs": int(adata_int.n_obs),
        "n_vars": int(adata_int.n_vars),
        "n_has_row": int(has_row.sum()),
        "cell_type_col": cell_type_col,
        "threshold": None if threshold is None else float(threshold),
        "f1_min": float(f1_min),
        "min_true_pos": None if min_true_pos is None else int(min_true_pos),
        "top_concepts": None if top_concepts is None else int(top_concepts),
        "keep_duplicate_features": bool(keep_duplicate_features),
        "best_ari": float(best_ari["ari"]),
        "best_resolution": float(best_ari["resolution"]),
        "best_n_clusters": int(best_ari["n_clusters"]),
        **{f"extract_{k}": v for k, v in extract_summary.items()},
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("[OK]", json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Build 3-model interpretable AnnData from SAE/F1 features and compute Leiden-celltype ARI.")
    ap.add_argument("--labels", nargs=3, required=True)
    ap.add_argument("--store-roots", nargs=3, required=True)
    ap.add_argument("--layers", nargs=3, required=True)
    ap.add_argument("--sae-ckpts", nargs=3, required=True)
    ap.add_argument("--f1-tables", nargs=3, required=True)
    ap.add_argument("--adata-path", required=True)
    ap.add_argument("--term-meta", default=None)
    ap.add_argument("--out-root", required=True)

    ap.add_argument("--threshold", type=float, default=0.15)
    ap.add_argument("--f1-min", type=float, default=0.6)
    ap.add_argument("--min-true-pos", type=int, default=3)
    ap.add_argument("--top-concepts", type=int, default=None)
    ap.add_argument("--keep-duplicate-features", action="store_true")

    ap.add_argument("--cell-agg", choices=["mean", "max"], default="mean")
    ap.add_argument("--exclude-token-values", nargs=3, default=None)

    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--max-shards", default="NONE")
    ap.add_argument("--device", default="cuda")

    ap.add_argument("--cell-type-col", default="auto")
    ap.add_argument("--leiden-resolutions", nargs="+", type=float, default=[0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    ap.add_argument("--n-neighbors", type=int, default=15)
    ap.add_argument("--n-pcs", type=int, default=50)
    ap.add_argument("--cluster-rep", choices=["pca", "raw"], default="pca")
    ap.add_argument("--scale", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_root = ensure_dir(args.out_root)
    max_shards = parse_optional_int(args.max_shards)

    print("[load] AnnData:", args.adata_path)
    adata = sc.read_h5ad(args.adata_path)
    print("  adata:", adata.shape)
    if "spatial" in adata.obsm:
        print("  spatial:", adata.obsm["spatial"].shape)

    cell_type_col = infer_cell_type_col(adata, args.cell_type_col)
    print("[cell type]", cell_type_col)
    term_meta = load_term_meta(args.term_meta)

    summaries = []
    for i, (label, store_root, layer, sae_ckpt, f1_table) in enumerate(zip(args.labels, args.store_roots, args.layers, args.sae_ckpts, args.f1_tables)):
        exclude_tokens = default_exclude_tokens(label) if args.exclude_token_values is None else parse_exclude_tokens(args.exclude_token_values[i])
        summary = build_one_model(
            label=label,
            store_root=store_root,
            layer=layer,
            sae_ckpt=sae_ckpt,
            f1_table=f1_table,
            adata=adata,
            term_meta=term_meta,
            out_root=out_root,
            threshold=args.threshold,
            f1_min=args.f1_min,
            min_true_pos=args.min_true_pos,
            top_concepts=args.top_concepts,
            keep_duplicate_features=args.keep_duplicate_features,
            cell_agg=args.cell_agg,
            exclude_tokens=exclude_tokens,
            batch_size=args.batch_size,
            max_shards=max_shards,
            device=args.device,
            cell_type_col=cell_type_col,
            resolutions=args.leiden_resolutions,
            n_neighbors=args.n_neighbors,
            n_pcs=args.n_pcs,
            cluster_rep=args.cluster_rep,
            scale=args.scale,
            seed=args.seed,
        )
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out_root / "combined_summary.csv", index=False)
    print("\n[OK] wrote:", out_root)
    print(summary_df[["label", "layer", "n_vars", "n_has_row", "best_ari", "best_resolution", "best_n_clusters", "h5ad_path"]].to_string(index=False))


if __name__ == "__main__":
    main()


# python test_build_interpretable_adata_3models.py \
#   --labels scgpt c2sscale geneformer \
#   --store-roots \
#     /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/full_scgpt_cosmx \
#     /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/full_c2sscale_cosmx \
#     /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/full_geneformer_cosmx \
#   --layers layer_4.norm2 layer_17 layer_4 \
#   --sae-ckpts \
#     /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/full_scgpt_cosmx/sae/layer_4.norm2/nr_on__steps_6000__l1_3e-3/sae_layer_4.norm2_best.pt \
#     /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/full_c2sscale_cosmx/sae/layer_17/nr_on__steps_6000__l1_3e-3/sae_layer_17_best.pt \
#     /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/full_geneformer_cosmx/sae/layer_4/nr_on__steps_6000__l1_3e-3/sae_layer_4_best.pt \
#   --f1-tables \
#     /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/heldout_3models_l1_3e-3_geneheldout/scgpt/layer_4.norm2/test_concept_f1_scores.csv \
#     /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/heldout_3models_l1_3e-3_geneheldout/c2sscale/layer_17/test_concept_f1_scores.csv \
#     /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/heldout_3models_l1_3e-3_geneheldout/geneformer/layer_4/test_concept_f1_scores.csv \
#   --term-meta /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/heldout_3models_l1_3e-3_geneheldout/gprofiler/gprofiler_terms.tsv \
#   --adata-path /maiziezhou_lab2/yunfei/Projects/FM_temp/interGFM/ge_shards/cosmx_human_lung_sec8.h5ad \
#   --out-root /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/interpretable_adata_3models_smoke \
#   --threshold 0.15 \
#   --f1-min 0.4 \
#   --min-true-pos 3 \
#   --top-concepts 300 \
#   --max-shards None \
#   --cell-type-col author_cell_type