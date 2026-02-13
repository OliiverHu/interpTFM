from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from io.activation_store import ActivationStore
from sae.sae_base import SAESpec, SAEResult


class AutoEncoder(nn.Module):
    """
    Minimal SAE: x -> enc -> relu -> dec
    Keep it simple now; you can swap in your exact implementation later.
    """
    def __init__(self, d_in: int, n_latents: int):
        super().__init__()
        self.encoder = nn.Linear(d_in, n_latents, bias=False)
        self.decoder = nn.Linear(n_latents, d_in, bias=False)

    def forward(self, x: torch.Tensor):
        z = F.relu(self.encoder(x))
        x_hat = self.decoder(z)
        return x_hat, z


@dataclass
class TrainState:
    step: int = 0
    running_loss: float = 0.0


def fit_sae_for_layer(
    store: ActivationStore,
    layer: str,
    spec: SAESpec,
    output_dir: str,
    device: str = "cuda",
    batch_size: int = 2048,
) -> SAEResult:
    """
    Trains SAE on token-level activations stored in ActivationStore for `layer`.
    """
    os.makedirs(output_dir, exist_ok=True)
    dev = torch.device(device)

    # Peek one batch to get hidden dim
    first_batch, _ = next(store.iter_token_batches(layer=layer, batch_size=min(batch_size, 256)))
    d_in = int(first_batch.shape[1])

    torch.manual_seed(spec.seed)

    ae = AutoEncoder(d_in=d_in, n_latents=spec.n_latents).to(dev)
    opt = torch.optim.Adam(ae.parameters(), lr=spec.lr)

    state = TrainState()

    data_iter = store.iter_token_batches(layer=layer, batch_size=batch_size, shuffle_shards=True)

    for step in range(spec.steps):
        try:
            x, _ = next(data_iter)
        except StopIteration:
            data_iter = store.iter_token_batches(layer=layer, batch_size=batch_size, shuffle_shards=True)
            x, _ = next(data_iter)

        x = x.to(dev).float()

        opt.zero_grad(set_to_none=True)
        x_hat, z = ae(x)
        recon = F.mse_loss(x_hat, x)
        sparsity = z.abs().mean()
        loss = recon + spec.l1 * sparsity
        loss.backward()
        opt.step()

        state.step += 1
        state.running_loss = float(loss.item())

        if (step + 1) % 500 == 0:
            print(f"[{layer}] step={step+1} loss={loss.item():.6f} recon={recon.item():.6f} sparsity={sparsity.item():.6f}")

    model_path = os.path.join(output_dir, f"sae_{layer}.pt")
    torch.save({"state_dict": ae.state_dict(), "d_in": d_in, "n_latents": spec.n_latents}, model_path)

    return SAEResult(
        layer=layer,
        model_path=model_path,
        summary={"final_loss": state.running_loss, "d_in": d_in, "n_latents": spec.n_latents},
    )
