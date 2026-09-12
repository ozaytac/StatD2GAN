"""Generator and discriminator architectures.

Three discriminator heads inspect the same generated sequence from different
angles: `TemporalDiscriminator` judges raw sequence realism, `StatDiscriminator`
(or its `MomentDiscriminator` variant, spec B2) judges distributional fidelity,
and `SortedDiscriminator` (or its `RankSortedDiscriminator` variant, spec F3)
judges marginal / copula structure via a sorted or soft-ranked representation.
Which subset is active, and how their outputs are weighted, is what the
ablation arms in `ablation.py` vary.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import C


class CircularConv1d(nn.Module):
    """1D conv with circular padding, so the 168h sequence is treated as a
    ring rather than having artificial edges at hour 0 / hour 167."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int = 1):
        super().__init__()
        self.pad = (kernel - 1) // 2 * dilation
        self.conv = nn.utils.spectral_norm(
            nn.Conv1d(in_ch, out_ch, kernel, padding=0, dilation=dilation))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.pad, self.pad), "circular"))


class Generator(nn.Module):
    """LSTM generator mapping latent noise to an (T, n_features) sequence.

    The two (sin, cos) feature pairs (wind direction, hour-of-day) are
    re-normalized onto the unit circle after the linear head, so the
    generator can never emit a direction/time encoding off the circle.
    """

    def __init__(self, noise_dim: int = C.latent_dim):
        super().__init__()
        self.noise_dim = noise_dim
        self.lstm = nn.LSTM(noise_dim, C.hidden_dim, 3, batch_first=True)
        self.fc = nn.utils.spectral_norm(nn.Linear(C.hidden_dim, C.n_features))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 2:
            z = z.unsqueeze(1).expand(-1, C.seq_len, -1)
        out, _ = self.lstm(z)
        x = self.fc(out)
        # torch.cat avoids in-place setitem on x, which AMP/autograd needs.
        norm_wind = F.normalize(x[..., 4:6], dim=-1)
        norm_hour = F.normalize(x[..., 7:9], dim=-1)
        return torch.cat([
            x[..., 0:4],
            norm_wind,
            x[..., 6:7],
            norm_hour,
            x[..., 9:11],
        ], dim=-1)


class TemporalDiscriminator(nn.Module):
    """D_temporal: sequence-level realism via an LSTM reading the whole window."""

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(C.n_features, 256, 2, batch_first=True)
        self.fc = nn.utils.spectral_norm(nn.Linear(256, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])


class StatDiscriminator(nn.Module):
    """D_stat (LSTM variant): distributional fidelity via a deeper LSTM."""

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(C.n_features, 512, 3, batch_first=True)
        self.fc = nn.utils.spectral_norm(nn.Linear(512, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])


class SortedDiscriminator(nn.Module):
    """D_sorted: marginal CDF structure via a per-feature sorted representation.

    Sorting each feature along the time axis discards temporal order and
    exposes only the empirical marginal distribution, which is what this
    head is judging.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = CircularConv1d(C.n_features, 128, 5)
        self.conv2 = CircularConv1d(128, 64, 3)
        self.fc = nn.utils.spectral_norm(nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sx = torch.sort(x, dim=1)[0].transpose(1, 2)
        sx = F.leaky_relu(self.conv1(sx), 0.2)
        sx = F.leaky_relu(self.conv2(sx), 0.2)
        return self.fc(torch.max(sx, dim=2)[0])


class MomentDiscriminator(nn.Module):
    """D_stat (moment variant, spec B2): a fully-connected net over
    s(x) = [mean, var, skew, excess kurtosis] per feature (paper Eq. 9-10).

    Moments are computed in fp32 even under autocast: z**4 loses precision
    in fp16, and z is clamped to [-5, 5] because both the gradient penalty
    and the tail-heavy sampling in `losses.sample_z` push it into a regime
    where d(kurtosis)/dx ~ z**3 amplifies by ~125x at |z|=5.
    """

    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(4 * C.n_features)
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(4 * C.n_features, 256)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Linear(256, 128)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Linear(128, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, F)
        with torch.amp.autocast("cuda", enabled=False):
            x = x.float()
            mu = x.mean(1)
            var = x.var(1, unbiased=False)
            sd = var.clamp_min(1e-8).sqrt()
            z = ((x - mu.unsqueeze(1)) / sd.unsqueeze(1)).clamp(-5, 5)
            skew = (z ** 3).mean(1)
            kurt = (z ** 4).mean(1) - 3.0
            return self.net(self.norm(torch.cat([mu, var, skew, kurt], dim=-1)))


def soft_rank(x: torch.Tensor, anchors: int = 32, tau: float = 0.05) -> torch.Tensor:
    """Differentiable rank transform (empirical-copula approximation, spec F3).

    A naive `argsort().float()` has no gradient and leaves that branch dead;
    rank is instead the sigmoid-softened count of comparisons against a set
    of reference anchors. Anchors sit on a deterministic quantile grid
    rather than a random subsample: `randperm` would inject fresh noise into
    the CDF estimate on every forward pass (SD ~ 0.09 at K=32), whereas a
    fixed quantile grid gets the same cost with zero anchor variance, which
    matters when sweeping tau. Anchors are detached: they are a reference
    frame, not a quantity to optimize against. Row alignment is preserved,
    so (unlike full sorting) the output still carries copula information.

    anchors=32 keeps memory at B*T*K*F ~ 30M elements instead of the ~159M a
    full T*T comparison would need.
    """
    grid = torch.linspace(0, 1, anchors, device=x.device, dtype=x.dtype)
    ref = torch.quantile(x.float(), grid.float(), dim=1).permute(1, 0, 2)
    ref = ref.to(x.dtype).detach()               # (B, K, F)
    d = x.unsqueeze(2) - ref.unsqueeze(1)        # (B, T, K, F)
    return torch.sigmoid(d / tau).mean(2)        # (B, T, F) in [0, 1]


class RankSortedDiscriminator(nn.Module):
    """D_sorted (rank variant, spec F3): the copula-aware counterpart of
    `SortedDiscriminator`. Same conv body, but the input is a row-aligned
    soft rank rather than a column-independent hard sort. This does not
    replace the sorted-input head — it runs alongside it as a separate
    ablation arm (`rank_D_sort`), because the comparison between the two is
    itself the experiment: if `no_D_sorted` hurts but the rank variant does
    not degrade the same way, the mechanism is the copula channel rather
    than the marginal channel.
    """

    def __init__(self, anchors: int = 32, tau: float = 0.05):
        super().__init__()
        self.anchors, self.tau = anchors, tau
        self.conv1 = CircularConv1d(C.n_features, 128, 5)
        self.conv2 = CircularConv1d(128, 64, 3)
        self.fc = nn.utils.spectral_norm(nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = 2.0 * soft_rank(x, self.anchors, self.tau) - 1.0  # symmetric [-1, 1]
        sx = r.transpose(1, 2)
        sx = F.leaky_relu(self.conv1(sx), 0.2)
        sx = F.leaky_relu(self.conv2(sx), 0.2)
        return self.fc(torch.max(sx, dim=2)[0])


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
