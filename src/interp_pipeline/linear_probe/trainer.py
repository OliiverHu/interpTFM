from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from interp_pipeline.io.activation_store import ActivationStore
from interp_pipeline.linear_probe.dataset import ConceptFilteredDataset
from interp_pipeline.linear_probe.probe_base import LinearProbe, LinearProbeSpec, save_probe


def train_probe_for_layer(
    store: ActivationStore,
    layer: str,
    concept_matrix: pd.DataFrame,
    id_to_gene: Dict[str, str],
    spec: LinearProbeSpec,
    output_dir: str,
    device: str,
    use_wandb: bool = False,
    wandb_project: str = "linear-probe",
    wandb_entity: Optional[str] = None,
    wandb_name: Optional[str] = None,
) -> LinearProbe:
    """
    Train one linear probe for a single transformer layer.

    Args:
        store:          ActivationStore holding pre-extracted token activations.
        layer:          Layer name to read from the store (e.g. "layer_4").
        concept_matrix: Binary DataFrame [n_concepts × n_genes] from
                        build_binary_empirical_gt().  Columns = gene names.
        id_to_gene:     {str(token_id): gene_name} from build_id_to_gene().
        spec:           Training hyperparameters.
        output_dir:     Directory where probe_{layer}.pt will be written.
        device:         Torch device string.
        use_wandb:      Whether to log loss/accuracy to Weights & Biases.
        wandb_project:  W&B project name (used when use_wandb=True).
        wandb_name:     W&B run name; defaults to "probe-{layer}".

    Returns:
        Trained LinearProbe (weight already saved to disk).
    """
    train_ds = ConceptFilteredDataset(
        store=store,
        layer=layer,
        concept_matrix=concept_matrix,
        id_to_gene=id_to_gene,
        split="train",
        test_fraction=spec.test_fraction,
        seed=spec.seed,
    )
    test_ds = ConceptFilteredDataset(
        store=store,
        layer=layer,
        concept_matrix=concept_matrix,
        id_to_gene=id_to_gene,
        split="test",
        test_fraction=spec.test_fraction,
        seed=spec.seed,
    )

    pin = device != "cpu"
    # IterableDataset: shuffle=False (shard order is shuffled at construction).
    train_loader = DataLoader(
        train_ds, batch_size=spec.batch_size, shuffle=False, num_workers=0, pin_memory=pin
    )
    test_loader = DataLoader(
        test_ds, batch_size=spec.batch_size, shuffle=False, num_workers=0, pin_memory=pin
    )

    probe = LinearProbe(spec.hidden_size, spec.n_concepts, device=device, seed=spec.seed)
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=spec.lr,
        betas=spec.betas,
        weight_decay=spec.weight_decay,
    )

    if use_wandb:
        import wandb
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_name or f"probe-{layer}",
            config={
                "layer": layer,
                "n_concepts": spec.n_concepts,
                "hidden_size": spec.hidden_size,
                "epochs": spec.epochs,
                "batch_size": spec.batch_size,
                "lr": spec.lr,
                "weight_decay": spec.weight_decay,
            },
        )

    step = 0
    for epoch in range(spec.epochs):
        for acts, labels in tqdm(train_loader, desc=f"[{layer}] epoch {epoch + 1}/{spec.epochs}"):
            acts = acts.to(device, dtype=torch.float32)
            labels = labels.to(device)

            optimizer.zero_grad()
            loss = _multilabel_loss(probe, acts, labels)
            loss.backward()
            optimizer.step()

            if use_wandb:
                import wandb
                wandb.log({"loss": loss.item()}, step=step)
            step += 1

        metrics = evaluate_probe(probe, test_loader, device)
        print(f"[{layer}] epoch {epoch + 1}: top-k acc = {metrics['top_k_acc']:.4f}")
        if use_wandb:
            import wandb
            wandb.log({"test_top_k_acc": metrics["top_k_acc"]}, step=step)

    if use_wandb:
        import wandb
        wandb.finish()

    save_probe(probe, layer, output_dir)
    return probe


def _multilabel_loss(probe: LinearProbe, acts: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Multi-label BCE loss with per-concept positive class weighting.

    Each concept column is trained as an independent sigmoid binary classifier,
    so W[:, c] is the uncontaminated decision-boundary normal for concept c.
    This is important for steering: apply_scale_2D uses W[:, c] as a direction,
    which requires it to be geometrically meaningful in isolation.

    pos_weight = (n_neg / n_pos) per concept, clamped to [1, 100] to prevent
    numerical blow-up on rare concepts.
    """
    logits = probe.project(acts)                       # [N, n_concepts]
    pos_weight = (
        (1 - labels).sum(0) / labels.sum(0).clamp(min=1)
    ).clamp(max=100).to(acts.device)
    return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)


def evaluate_probe(
    probe: LinearProbe,
    test_loader: DataLoader,
    device: str,
    max_eval_samples: int = 4096,
) -> Dict[str, float]:
    """
    Returns {"auroc": macro-AUROC, "top_k_acc": top-k accuracy}.

    auroc: macro-average over concepts with at least one positive and one
           negative example.  Computed on at most max_eval_samples rows
           (early-stop) to avoid materialising the full test set in RAM.

    top_k_acc: per-sample recall where k = number of true concepts for that
               sample — fraction of true concepts recovered in the top-k
               predicted logits.  Same cap applies.
    """
    all_logits: list = []
    all_labels: list = []
    n_collected = 0

    with torch.no_grad():
        for acts, labels in test_loader:
            acts = acts.to(device, dtype=torch.float32)
            logits = probe.project(acts)               # [B, n_concepts]
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            n_collected += acts.shape[0]
            if n_collected >= max_eval_samples:
                break

    logits_t = torch.cat(all_logits, dim=0)[:max_eval_samples]  # [N, C]
    labels_t = torch.cat(all_labels, dim=0)[:max_eval_samples]  # [N, C]

    # # Macro-AUROC — uncomment when needed (slow for large concept matrices).
    # logits_np = logits_t.numpy()
    # labels_np = labels_t.numpy()
    # col_sum = labels_np.sum(0)
    # valid = (col_sum > 0) & (col_sum < labels_np.shape[0])
    # auroc = (
    #     float(roc_auc_score(labels_np[:, valid], logits_np[:, valid], average="macro"))
    #     if valid.sum() > 0 else 0.0
    # )
    auroc = 0.0

    # Top-k accuracy (variable k = true positives per sample).
    total_correct = 0
    total_true = 0
    for i in range(labels_t.shape[0]):
        k = int(labels_t[i].sum().item())
        if k == 0:
            continue
        _, topk_idx = logits_t[i].topk(k)
        total_correct += int(labels_t[i, topk_idx].sum().item())
        total_true += k
    top_k_acc = total_correct / total_true if total_true > 0 else 0.0

    return {"auroc": auroc, "top_k_acc": top_k_acc}
