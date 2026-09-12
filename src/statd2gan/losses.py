"""Loss functions and the evolutionary discriminator-weighting scheme."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .config import C


class EvolutionaryWeights:
    """Fitness-proportionate adaptation of the per-discriminator loss weights.

    Each discriminator's weight tracks how far above (or below) average its
    own fitness (critic loss magnitude) is, bounded so no single head can
    dominate or be starved out (`max_ratio`), and renormalized each step so
    the weights always sum back to `n`.
    """

    def __init__(self, n: int = 3, eta: float = 0.03):
        self.weights = np.ones(n, dtype=np.float32)
        self.eta = eta
        self.sum0 = float(n)
        self.bounds = (0.5, 6.0)
        self.max_ratio = 3.0

    def update(self, fitness: np.ndarray) -> np.ndarray:
        f = np.clip(fitness / (fitness.sum() + 1e-8), 0.01, 1.0)
        avg = np.dot(self.weights, f) / self.weights.sum()
        self.weights *= (1 + self.eta * (f - avg))
        self.weights = np.clip(self.weights, *self.bounds)
        if self.weights.max() / (self.weights.min() + 1e-8) > self.max_ratio:
            self.weights = np.minimum(self.weights, self.weights.min() * self.max_ratio)
        self.weights *= self.sum0 / (self.weights.sum() + 1e-8)
        return self.weights.copy()


def make_scaler_tensors(scaler, device: str):
    """Build per-feature (scale, min) tensors from the fitted MinMaxScaler,
    so `physics_loss` can undo the [-1, 1] scaling without leaving the GPU."""
    sc = torch.ones(C.n_features, device=device)
    mn = torch.zeros(C.n_features, device=device)
    for i, idx in enumerate(C.scalable_idx):
        sc[idx] = float(scaler.scale_[i])
        mn[idx] = float(scaler.min_[i])
    return sc, mn


def physics_loss(fake: torch.Tensor, sc: torch.Tensor, mn: torch.Tensor) -> torch.Tensor:
    """Soft penalty for physically inconsistent generated sequences.

    Thresholds are read from `config.C` — the exact same constants
    `metrics.compute_metrics` uses to flag violations at evaluation time
    (spec C2), so training pressure and evaluation criteria can never
    silently diverge. Unit-circle penalties for the (sin, cos) pairs are
    intentionally absent: `models.Generator.forward` already enforces that
    constraint via `F.normalize`, so an additional penalty would be
    identically zero (spec C1 — behaviour is unchanged, just not redundant).
    """
    ph = (fake - mn) / sc
    t, dew, pres, hum = ph[..., 0], ph[..., 1], ph[..., 2], ph[..., 6]

    loss = 50.0 * (F.softplus(dew - t + C.dew_tol, beta=50) ** 2).mean()
    loss += 5.0 * (F.relu(pres - C.pres_hi) + F.relu(C.pres_lo - pres)).mean()
    loss += 5.0 * (F.relu(hum - C.hum_hi) + F.relu(C.hum_lo - hum)).mean()
    return loss


def cdf_loss(real: torch.Tensor, fake: torch.Tensor, w: float = 30.0) -> torch.Tensor:
    """MSE between real and fake empirical marginal CDFs (one sort over all
    features at once, rather than a Python loop per feature)."""
    r = torch.sort(real.reshape(-1, C.n_features), dim=0)[0]
    s = torch.sort(fake.reshape(-1, C.n_features), dim=0)[0]
    n = min(r.shape[0], s.shape[0])
    return w * F.mse_loss(r[:n], s[:n])


def quantile_loss(real: torch.Tensor, fake: torch.Tensor, w: float = 8.0) -> torch.Tensor:
    """MSE between real and fake values at a fixed set of quantiles."""
    qs = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=real.device)
    r = torch.sort(real.reshape(-1, C.n_features), dim=0)[0]
    s = torch.sort(fake.reshape(-1, C.n_features), dim=0)[0]
    idx_r = (qs * r.shape[0]).long().clamp(0, r.shape[0] - 1)
    idx_s = (qs * s.shape[0]).long().clamp(0, s.shape[0] - 1)
    return w * F.mse_loss(r[idx_r], s[idx_s])


def sample_z(bsz: int, noise_dim: int, epoch: int, device: str) -> torch.Tensor:
    """Progressive tail-heavy latent sampling: standard Gaussian early in
    training, growing to a mixture with a 3x-scaled tail (up to 30% of the
    batch) as training progresses, to keep the generator exposed to extreme
    latent draws once the easy density has been fit."""
    if epoch < 30:
        return torch.randn(bsz, noise_dim, device=device)
    frac = min(0.3, 0.1 + 0.2 * (epoch - 30) / 70)
    n_tail = int(bsz * frac)
    return torch.cat([
        torch.randn(bsz - n_tail, noise_dim, device=device),
        torch.randn(n_tail, noise_dim, device=device) * 3.0,
    ])
