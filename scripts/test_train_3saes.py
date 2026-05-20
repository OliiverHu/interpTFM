#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch

from interp_pipeline.io.activation_store import ActivationStore, ActivationStoreSpec
from interp_pipeline.sae.sae_base import SAESpec, AutoEncoder


class ConstrainedAdam(torch.optim.Adam):
    def __init__(self, params, constrained_params, lr: float):
        super().__init__(params, lr=lr)
        self.constrained_params = list(constrained_params)

    @torch.no_grad()
    def step(self, closure=None):
        for p in self.constrained_params:
            if p.grad is None:
                continue
            normed_p = p / p.norm(dim=0, keepdim=True).clamp_min(1e-12)
            p.grad -= (p.grad * normed_p).sum(dim=0, keepdim=True) * normed_p

        super().step(closure=closure)

        for p in self.constrained_params:
            p /= p.norm(dim=0, keepdim=True).clamp_min(1e-12)


def warmup_lr_multiplier(step: int, warmup_steps: int, resample_steps: Optional[int]) -> float:
    if warmup_steps <= 0:
        return 1.0
    if resample_steps is None or resample_steps <= 0:
        return min((step + 1) / warmup_steps, 1.0)
    local_step = (step + 1) % resample_steps
    return min(max(local_step, 1) / warmup_steps, 1.0)


@torch.no_grad()
def resample_neurons(ae: AutoEncoder, optimizer: ConstrainedAdam, deads: torch.Tensor, activations: torch.Tensor) -> int:
    if deads.sum().item() == 0:
        return 0

    x_hat, _ = ae(activations)
    losses = (activations - x_hat).norm(dim=-1)

    n_resample = min(int(deads.sum().item()), int(losses.shape[0]))
    if n_resample == 0:
        return 0

    idx = torch.multinomial(losses, num_samples=n_resample, replacement=False)
    sampled_vecs = activations[idx]

    alive = ~deads
    if alive.any():
        alive_norm = ae.encoder.weight[alive].norm(dim=-1).mean()
    else:
        alive_norm = torch.tensor(1.0, device=activations.device)

    dead_idx = deads.nonzero(as_tuple=False).squeeze(-1)[:n_resample]

    ae.encoder.weight[dead_idx] = sampled_vecs * alive_norm * 0.2
    ae.decoder.weight[:, dead_idx] = (
        sampled_vecs / sampled_vecs.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    ).T
    if ae.encoder.bias is not None:
        ae.encoder.bias[dead_idx] = 0.0

    for state in optimizer.state.values():
        for _, value in list(state.items()):
            if not torch.is_tensor(value):
                continue
            if value.ndim == 1 and value.shape[0] == ae.dict_size:
                value[dead_idx] = 0.0
            elif value.ndim == 2:
                if value.shape[0] == ae.dict_size:
                    value[dead_idx] = 0.0
                if value.shape[1] == ae.dict_size:
                    value[:, dead_idx] = 0.0

    return int(n_resample)


def save_checkpoint(
    path: str,
    ae: AutoEncoder,
    spec: SAESpec,
    d_in: int,
    summary: Dict,
) -> None:
    torch.save(
        {
            "state_dict": ae.state_dict(),
            "d_in": d_in,
            "n_latents": spec.n_latents,
            "spec": asdict(spec),
            "summary": summary,
        },
        path,
    )


def build_spec(args, n_latents: int) -> SAESpec:
    return SAESpec(
        n_latents=n_latents,
        l1=args.l1,
        lr=args.lr,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        resample_steps=None if args.no_resample else args.resample_steps,
        seed=args.seed,
    )


def infer_d_in(store: ActivationStore, layer: str, batch_size: int) -> int:
    first_batch, _ = next(store.iter_token_batches(layer=layer, batch_size=min(batch_size, 256)))
    return int(first_batch.shape[1])


def fit_sae_for_layer_bestlast(
    store: ActivationStore,
    layer: str,
    spec: SAESpec,
    output_dir: str,
    label: str,
    device: str = "cuda",
    batch_size: int = 2048,
    save_every: int = 0,
    best_metric: str = "loss",
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    dev = torch.device(device)

    d_in = infer_d_in(store, layer, batch_size)

    torch.manual_seed(spec.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(spec.seed)

    ae = AutoEncoder(d_in=d_in, n_latents=spec.n_latents).to(dev)

    opt = ConstrainedAdam(
        params=ae.parameters(),
        constrained_params=ae.decoder.parameters(),
        lr=spec.lr,
    )

    data_iter = store.iter_token_batches(layer=layer, batch_size=batch_size, shuffle_shards=True)

    steps_since_active = torch.zeros(spec.n_latents, dtype=torch.long, device=dev) if spec.resample_steps else None
    resampled_total = 0

    best_path = os.path.join(output_dir, f"sae_{layer}_best.pt")
    last_path = os.path.join(output_dir, f"sae_{layer}_last.pt")

    best_value = float("inf")
    best_step = -1
    best_summary: Dict = {}

    history_rows: List[Dict] = []

    for step in range(spec.steps):
        try:
            x, _ = next(data_iter)
        except StopIteration:
            data_iter = store.iter_token_batches(layer=layer, batch_size=batch_size, shuffle_shards=True)
            x, _ = next(data_iter)

        x = x.to(dev).float()

        lr_mult = warmup_lr_multiplier(step, spec.warmup_steps, spec.resample_steps)
        for group in opt.param_groups:
            group["lr"] = spec.lr * lr_mult

        opt.zero_grad(set_to_none=True)
        x_hat, z = ae(x)
        recon = (x - x_hat).norm(dim=-1).mean()
        sparsity = z.abs().sum(dim=-1).mean()
        loss = recon + spec.l1 * sparsity
        loss.backward()
        opt.step()

        if steps_since_active is not None:
            deads = (z == 0).all(dim=0)
            steps_since_active[deads] += 1
            steps_since_active[~deads] = 0

            if (step + 1) % spec.resample_steps == 0:
                to_resample = steps_since_active > max(spec.resample_steps // 2, 1)
                n = resample_neurons(ae, opt, to_resample, x)
                resampled_total += n
                if n > 0:
                    steps_since_active[to_resample] = 0

        l0 = float((z > 0).float().sum(dim=-1).mean().item())
        l0_pct = 100.0 * l0 / spec.n_latents

        row = {
            "label": label,
            "layer": layer,
            "step": step + 1,
            "loss": float(loss.item()),
            "recon": float(recon.item()),
            "sparsity": float(sparsity.item()),
            "l0": l0,
            "l0_pct": l0_pct,
            "lr": float(opt.param_groups[0]["lr"]),
            "resampled_total": int(resampled_total),
            "d_in": d_in,
            "n_latents": spec.n_latents,
        }
        history_rows.append(row)

        metric_value = row[best_metric]
        if metric_value < best_value:
            best_value = metric_value
            best_step = step + 1
            best_summary = {
                **row,
                "best_metric": best_metric,
            }
            save_checkpoint(best_path, ae, spec, d_in, best_summary)

        if save_every and (step + 1) % save_every == 0:
            periodic_path = os.path.join(output_dir, f"sae_{layer}_step{step+1}.pt")
            save_checkpoint(
                periodic_path,
                ae,
                spec,
                d_in,
                {**row, "checkpoint_type": "periodic"},
            )

        if (step + 1) % 500 == 0:
            print(
                f"[{label}:{layer}] step={step+1} "
                f"loss={loss.item():.6f} recon={recon.item():.6f} "
                f"sparsity={sparsity.item():.6f} l0={l0:.2f} l0_pct={l0_pct:.2f}% "
                f"lr={opt.param_groups[0]['lr']:.2e} best_{best_metric}={best_value:.6f}@{best_step}"
            )

    last_summary = {
        **history_rows[-1],
        "checkpoint_type": "last",
        "best_metric": best_metric,
        "best_value": best_value,
        "best_step": best_step,
    }
    save_checkpoint(last_path, ae, spec, d_in, last_summary)

    hist_df = pd.DataFrame(history_rows)
    hist_df.to_csv(os.path.join(output_dir, f"sae_{layer}_training_history.csv"), index=False)

    summary = {
        "label": label,
        "layer": layer,
        "best_path": best_path,
        "last_path": last_path,
        "best_metric": best_metric,
        "best_step": best_step,
        "best_value": best_value,
        "final_loss": history_rows[-1]["loss"],
        "final_recon": history_rows[-1]["recon"],
        "final_sparsity": history_rows[-1]["sparsity"],
        "final_l0": history_rows[-1]["l0"],
        "final_l0_pct": history_rows[-1]["l0_pct"],
        "resampled_total": resampled_total,
        "d_in": d_in,
        "n_latents": spec.n_latents,
        "history_csv": os.path.join(output_dir, f"sae_{layer}_training_history.csv"),
    }

    with open(os.path.join(output_dir, f"sae_{layer}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    ap = argparse.ArgumentParser(description="Train 3 SAEs and save both best and last checkpoints.")
    ap.add_argument("--labels", nargs=3, required=True, help="Names like scgpt c2sscale geneformer")
    ap.add_argument("--store-roots", nargs=3, required=True, help="ActivationStore roots for the 3 models")
    ap.add_argument("--layers", nargs=3, required=True, help="Selected layer for each model")
    ap.add_argument("--out-dirs", nargs=3, required=True, help="Output SAE dirs for the 3 models")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--latent-multiplier", type=int, default=8, help="Set n_latents = latent_multiplier * d_in")
    ap.add_argument("--n-latents", nargs=3, type=int, default=None, help="Optional explicit n_latents for the 3 models; overrides multiplier")
    ap.add_argument("--l1", type=float, default=1e-3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--warmup-steps", type=int, default=1000)
    ap.add_argument("--resample-steps", type=int, default=2000)
    ap.add_argument("--no-resample", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=0, help="Optional periodic checkpoint interval")
    ap.add_argument("--best-metric", choices=["loss", "recon", "sparsity"], default="loss")
    args = ap.parse_args()

    all_summaries = []

    for idx, (label, store_root, layer, out_dir) in enumerate(zip(args.labels, args.store_roots, args.layers, args.out_dirs)):
        print("=" * 90)
        print(f"Preparing SAE for {label} | layer={layer}")
        print(f"store_root={store_root}")
        print(f"out_dir={out_dir}")
        print("=" * 90)

        store = ActivationStore(ActivationStoreSpec(root=store_root))
        d_in = infer_d_in(store, layer, args.batch_size)
        if args.n_latents is not None:
            n_latents = int(args.n_latents[idx])
        else:
            n_latents = int(d_in * args.latent_multiplier)

        print(f"[{label}:{layer}] d_in={d_in} -> n_latents={n_latents}")

        spec = build_spec(args, n_latents=n_latents)

        summary = fit_sae_for_layer_bestlast(
            store=store,
            layer=layer,
            spec=spec,
            output_dir=out_dir,
            label=label,
            device=args.device,
            batch_size=args.batch_size,
            save_every=args.save_every,
            best_metric=args.best_metric,
        )
        all_summaries.append(summary)

    summary_df = pd.DataFrame(all_summaries)
    summary_path = Path("runs/sae_train_3models_summary.csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)

    json_path = Path("runs/sae_train_3models_summary.json")
    with open(json_path, "w") as f:
        json.dump(all_summaries, f, indent=2)

    print("\nSaved summary:")
    print(" ", summary_path)
    print(" ", json_path)
    print("\nPreview:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()



# python test_train_3saes.py \
#   --labels scgpt c2sscale geneformer \
#   --store-roots \
#     runs/full_scgpt_cosmx \
#     runs/full_c2sscale_cosmx \
#     runs/full_geneformer_cosmx \
#   --layers layer_4.norm2 layer_17 layer_4 \
#   --out-dirs \
#     runs/full_scgpt_cosmx/sae/layer_4.norm2 \
#     runs/full_c2sscale_cosmx/sae/layer_17 \
#     runs/full_geneformer_cosmx/sae/layer_4 \
#   --device cuda \
#   --batch-size 1024 \
#   --latent-multiplier 8 \
#   --l1 1e-3 \
#   --lr 1e-4 \
#   --steps 8000 \
#   --warmup-steps 1000 \
#   --resample-steps 2000 \
#   --best-metric loss