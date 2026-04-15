from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch

from interp_pipeline.io.activation_store import ActivationStore
from interp_pipeline.sae.sae_base import SAESpec, SAEResult, AutoEncoder


@dataclass
class TrainState:
    step: int = 0
    running_loss: float = 0.0
    running_recon: float = 0.0
    running_sparsity: float = 0.0


class ConstrainedAdam(torch.optim.Adam):
    """
    Adam variant that projects gradients away from the decoder-column direction
    and re-normalizes decoder columns after each step.
    """

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


def _warmup_lr_multiplier(step: int, warmup_steps: int, resample_steps: Optional[int]) -> float:
    if warmup_steps <= 0:
        return 1.0
    if resample_steps is None or resample_steps <= 0:
        return min((step + 1) / warmup_steps, 1.0)
    local_step = (step + 1) % resample_steps
    return min(max(local_step, 1) / warmup_steps, 1.0)


@torch.no_grad()
def _resample_neurons(ae: AutoEncoder, optimizer: ConstrainedAdam, deads: torch.Tensor, activations: torch.Tensor):
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


def fit_sae_for_layer(
    store: ActivationStore,
    layer: str,
    spec: SAESpec,
    output_dir: str,
    device: str = "cuda",
    batch_size: int = 2048,
) -> SAEResult:
    """
    Train SAE on token-level activations in ActivationStore for a layer.

    Restores the older SAE recipe more faithfully than the current minimal trainer:
      - activation-space bias
      - encoder bias
      - constrained decoder columns
      - LR warmup
      - dead-neuron resampling
    """
    os.makedirs(output_dir, exist_ok=True)
    dev = torch.device(device)

    first_batch, _ = next(store.iter_token_batches(layer=layer, batch_size=min(batch_size, 256)))
    d_in = int(first_batch.shape[1])

    torch.manual_seed(spec.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(spec.seed)

    ae = AutoEncoder(d_in=d_in, n_latents=spec.n_latents).to(dev)

    opt = ConstrainedAdam(
        params=ae.parameters(),
        constrained_params=ae.decoder.parameters(),
        lr=spec.lr,
    )

    state = TrainState()
    data_iter = store.iter_token_batches(layer=layer, batch_size=batch_size, shuffle_shards=True)

    steps_since_active = torch.zeros(spec.n_latents, dtype=torch.long, device=dev) if spec.resample_steps else None
    resampled_total = 0

    for step in range(spec.steps):
        try:
            x, _ = next(data_iter)
        except StopIteration:
            data_iter = store.iter_token_batches(layer=layer, batch_size=batch_size, shuffle_shards=True)
            x, _ = next(data_iter)

        x = x.to(dev).float()

        lr_mult = _warmup_lr_multiplier(step, spec.warmup_steps, spec.resample_steps)
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
                n = _resample_neurons(ae, opt, to_resample, x)
                resampled_total += n
                if n > 0:
                    steps_since_active[to_resample] = 0

        state.step = step + 1
        state.running_loss = float(loss.item())
        state.running_recon = float(recon.item())
        state.running_sparsity = float(sparsity.item())

        if (step + 1) % 500 == 0:
            l0 = float((z > 0).float().sum(dim=-1).mean().item())
            l0_pct = 100.0 * l0 / spec.n_latents
            print(
                f"[{layer}] step={step+1} "
                f"loss={loss.item():.6f} recon={recon.item():.6f} "
                f"sparsity={sparsity.item():.6f} l0={l0:.2f} l0_pct={l0_pct:.2f}% "
                f"lr={opt.param_groups[0]['lr']:.2e}"
            )

    model_path = os.path.join(output_dir, f"sae_{layer}.pt")
    torch.save(
        {
            "state_dict": ae.state_dict(),
            "d_in": d_in,
            "n_latents": spec.n_latents,
            "spec": {
                "n_latents": spec.n_latents,
                "l1": spec.l1,
                "lr": spec.lr,
                "steps": spec.steps,
                "warmup_steps": spec.warmup_steps,
                "resample_steps": spec.resample_steps,
                "seed": spec.seed,
            },
            "summary": {
                "final_loss": state.running_loss,
                "final_recon": state.running_recon,
                "final_sparsity": state.running_sparsity,
                "d_in": d_in,
                "n_latents": spec.n_latents,
                "resampled_total": resampled_total,
            },
        },
        model_path,
    )

    return SAEResult(
        layer=layer,
        model_path=model_path,
        summary={
            "final_loss": state.running_loss,
            "final_recon": state.running_recon,
            "final_sparsity": state.running_sparsity,
            "d_in": d_in,
            "n_latents": spec.n_latents,
            "resampled_total": resampled_total,
        },
    )
