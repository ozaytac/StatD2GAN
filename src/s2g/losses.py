import numpy as np, torch, torch.nn.functional as F
from .config import P, PHY

M, NF = P["model"], len(P["data"]["features"])


class EvolutionaryWeights:
    """Fitness-proportional adaptation of the critic weights (sum fixed at n)."""
    def __init__(self, n):
        e = M["evo"]
        self.w, self.eta, self.sum0, self.bounds, self.max_ratio = np.ones(n, np.float32), e["eta"], float(n), e["bounds"], e["max_ratio"]

    def update(self, fit):
        f = np.clip(fit / (fit.sum() + 1e-8), 0.01, 1.0)
        self.w = self.w * (1 + self.eta * (f - np.dot(self.w, f) / self.w.sum()))
        self.w = np.clip(self.w, *self.bounds)
        if self.w.max() / (self.w.min() + 1e-8) > self.max_ratio:
            self.w = np.minimum(self.w, self.w.min() * self.max_ratio)
        self.w = self.w * self.sum0 / (self.w.sum() + 1e-8)
        return self.w.copy()


def physics_loss(fake, sc, mn):
    """Same thresholds as the evaluation metric (protocol.physics). sc/mn: scaler tensors."""
    ph = (fake - mn) / sc
    t, dew, pres, hum = ph[..., 0], ph[..., 1], ph[..., 2], ph[..., 6]
    return (50 * (F.softplus(dew - t + PHY["dew_tol"], beta=50) ** 2).mean()
            + 5 * (F.relu(pres - PHY["pres_hi"]) + F.relu(PHY["pres_lo"] - pres)).mean()
            + 5 * (F.relu(hum - PHY["hum_hi"]) + F.relu(PHY["hum_lo"] - hum)).mean())


def _sorted(x):
    return torch.sort(x.reshape(-1, NF), dim=0)[0]


def cdf_loss(real, fake, w=30.0):
    r, s = _sorted(real), _sorted(fake); n = min(len(r), len(s))
    return w * F.mse_loss(r[:n], s[:n])


def quantile_loss(real, fake, w=8.0):
    r, s = _sorted(real), _sorted(fake)
    qs = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=real.device)
    ir, is_ = (qs * len(r)).long().clamp(0, len(r) - 1), (qs * len(s)).long().clamp(0, len(s) - 1)
    return w * F.mse_loss(r[ir], s[is_])


def sample_z(bsz, dim, epoch, dev, gen=None):
    """Progressive heavy-tail mixture after epoch 30."""
    z = torch.randn(bsz, dim, device=dev, generator=gen)
    if epoch >= 30:
        n_tail = int(bsz * min(0.3, 0.1 + 0.2 * (epoch - 30) / 70))
        z[bsz - n_tail:] *= 3.0
    return z
