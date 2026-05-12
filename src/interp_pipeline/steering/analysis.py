from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch


def score_steering_regression(
    ctrl_act: torch.Tensor,
    pert_act: torch.Tensor,
    intv_acts: Dict[float, torch.Tensor],
    random_state: int = 42,
) -> Dict[float, float]:
    """
    Score each steered condition by how much it resembles the perturbed phenotype.

    Trains an SGD linear regressor on ctrl (label=0) vs perturbed (label=1)
    cell embeddings, then predicts the mean score for each steered condition.
    A score closer to 1.0 indicates the steered embeddings look more perturbed.

    Args:
        ctrl_act:    Tensor[N_ctrl, H] — unsteered control CLS activations.
        pert_act:    Tensor[N_pert, H] — true perturbed CLS activations.
        intv_acts:   {scale: Tensor[N_ctrl, H]} — steered CLS activations.
        random_state: Seed for the SGD regressor.

    Returns:
        {scale: float} — mean predicted perturbation score per scale.
    """
    from sklearn import linear_model

    X_train = np.vstack([ctrl_act.numpy(), pert_act.numpy()])
    y_train = np.array([0] * len(ctrl_act) + [1] * len(pert_act), dtype=np.float32)

    clf = linear_model.SGDRegressor(max_iter=1000, tol=1e-3, random_state=random_state)
    clf.fit(X_train, y_train)

    return {
        scale: float(clf.predict(act.numpy()).mean())
        for scale, act in intv_acts.items()
    }


def plot_steering_umap(
    ctrl_act: torch.Tensor,
    pert_act: torch.Tensor,
    intv_acts: Dict[float, torch.Tensor],
    gene_select: str,
    scale_palette: Optional[Dict[str, str]] = None,
    save_path: Optional[str] = None,
) -> Any:
    """
    Build a UMAP of ctrl + steered + perturbed CLS embeddings and plot it.

    Args:
        ctrl_act:      Tensor[N_ctrl, H].
        pert_act:      Tensor[N_pert, H].
        intv_acts:     {scale: Tensor[N_ctrl, H]}.
        gene_select:   Name of the perturbed gene (used as label and in title).
        scale_palette: Optional colour palette {label_string: hex_colour}.
                       Defaults to a built-in scheme.
        save_path:     If given, the UMAP figure is saved here (PDF or PNG).

    Returns:
        sc.AnnData with UMAP coordinates in .obsm["X_umap"].
    """
    import scanpy as sc

    scale_list = sorted(intv_acts.keys())

    ctrl_label = ["ctrl"] * ctrl_act.shape[0]
    pert_label = [gene_select] * pert_act.shape[0]
    intv_labels = [f"scale_{s}" for s in scale_list for _ in range(intv_acts[s].shape[0])]

    obs_labels = ctrl_label + pert_label + intv_labels
    all_acts = torch.cat(
        [ctrl_act, pert_act] + [intv_acts[s] for s in scale_list], dim=0
    )

    import anndata as ad
    adata_plot = ad.AnnData(X=all_acts.numpy())
    adata_plot.obs["label"] = obs_labels

    sc.pp.neighbors(adata_plot, use_rep="X")
    sc.tl.umap(adata_plot)

    if scale_palette is None:
        import matplotlib.cm as cm
        cmap = cm.get_cmap("tab10", len(scale_list))
        scale_palette = {f"scale_{s}": f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
                         for i, s in enumerate(scale_list)
                         for r, g, b, _ in [cmap(i)]}
        scale_palette["ctrl"] = "#aaaaaa"      # light gray — background reference
        scale_palette[gene_select] = "#000000"  # black — pert anchor

    point_standard = 120_000 / adata_plot.shape[0]
    point_sizes = [
        point_standard * 2 if lbl == gene_select   # pert: slightly larger to stand out
        else point_standard * 0.8 if lbl == "ctrl"  # ctrl: small, background reference
        else point_standard * 1.5                   # steered: foreground
        for lbl in adata_plot.obs["label"]
    ]

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        sc.settings.figdir = os.path.dirname(os.path.abspath(save_path))
        fname = os.path.basename(save_path)
        sc.pl.umap(
            adata_plot,
            color="label",
            title=f"Steering UMAP — {gene_select}",
            palette=scale_palette,
            size=point_sizes,
            save=f"_{fname}",
            show=False,
        )
    else:
        sc.pl.umap(
            adata_plot,
            color="label",
            title=f"Steering UMAP — {gene_select}",
            palette=scale_palette,
            size=point_sizes,
        )

    return adata_plot


def analyze_probe_activations(
    per_layer_cls: Dict[int, torch.Tensor],
    probes: Dict[int, torch.Tensor],
    concept_idx_union: List[int],
    concept_names: List[str],
    condition_labels: List[str],
    save_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Project per-layer CLS activations through probe concept directions and
    produce violin plots showing activation distributions by condition.

    Args:
        per_layer_cls:   {layer_idx: Tensor[N, H]} from collect_per_layer_cls_activations().
        probes:          {layer_idx: Tensor[H, n_concepts]} probe weight matrices.
        concept_idx_union: Indices into the n_concepts dimension to analyse.
        concept_names:   Human-readable names for each index in concept_idx_union.
                         Length must equal len(concept_idx_union).
        condition_labels: Condition string for each of the N cells (length N).
        save_dir:        If given, saves one violin PNG per concept here.

    Returns:
        Long-form DataFrame with columns [feature, layer, condition, activation].
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    if len(concept_names) != len(concept_idx_union):
        raise ValueError("concept_names length must match concept_idx_union length")

    n_layers = len(per_layer_cls)
    records = []

    for layer_idx, cls_acts in per_layer_cls.items():
        if layer_idx not in probes:
            continue
        w = probes[layer_idx]                                         # [H, n_concepts]
        directions = w[:, concept_idx_union]                          # [H, n_selected]
        projections = cls_acts @ directions                           # [N, n_selected]

        for ci, (c_idx, c_name) in enumerate(zip(concept_idx_union, concept_names)):
            proj_col = projections[:, ci].tolist()
            for val, cond in zip(proj_col, condition_labels):
                records.append(
                    {"feature": c_name, "layer": layer_idx, "condition": cond, "activation": val}
                )

    df = pd.DataFrame(records)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        unique_conditions = df["condition"].unique().tolist()
        n_cols = min(6, n_layers)
        n_rows = (n_layers + n_cols - 1) // n_cols

        for c_name in concept_names:
            df_feat = df[df["feature"] == c_name]
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), sharey=False)
            axs_flat = np.array(axs).flatten()

            for layer_idx in sorted(per_layer_cls.keys()):
                ax = axs_flat[layer_idx]
                df_layer = df_feat[df_feat["layer"] == layer_idx]
                sns.violinplot(data=df_layer, x="condition", y="activation", ax=ax)
                ax.set_title(f"Layer {layer_idx}")
                ax.set_xlabel("")
                ax.tick_params(axis="x", rotation=30)

            for ax in axs_flat[n_layers:]:
                ax.set_visible(False)

            fig.suptitle(f"LP activations — {c_name}", fontsize=13)
            fig.tight_layout()
            safe = c_name.replace("/", "_").replace(" ", "_")[:60]
            fig.savefig(os.path.join(save_dir, f"lp_act_{safe}.png"), dpi=150)
            plt.close(fig)

    return df
