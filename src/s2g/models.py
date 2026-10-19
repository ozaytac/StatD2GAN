"""StatD2GAN generator and discriminators (design: original notebook, v2)."""
import torch, torch.nn as nn, torch.nn.functional as F
from .config import P

M, NF, L = P["model"], len(P["data"]["features"]), P["data"]["seq_len"]
SN = nn.utils.spectral_norm


class CircularConv1d(nn.Module):
    def __init__(self, i, o, k):
        super().__init__()
        self.pad, self.conv = (k - 1) // 2, SN(nn.Conv1d(i, o, k))

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, self.pad), "circular"))


class Generator(nn.Module):
    def __init__(self, noise_dim=M["latent_dim"]):
        super().__init__()
        self.lstm = nn.LSTM(noise_dim, M["hidden_dim"], 3, batch_first=True)
        self.fc = SN(nn.Linear(M["hidden_dim"], NF))

    def forward(self, z):
        x = self.fc(self.lstm(z.unsqueeze(1).expand(-1, L, -1))[0])
        return torch.cat([x[..., :4], F.normalize(x[..., 4:6], dim=-1), x[..., 6:7],
                          F.normalize(x[..., 7:9], dim=-1), x[..., 9:]], -1)   # (sin, cos) on unit circle


class LSTMCritic(nn.Module):
    """D_temporal (256 x 2) and D_stat (512 x 3): LSTM critic on the raw sequence."""
    def __init__(self, hidden, layers):
        super().__init__()
        self.lstm, self.fc = nn.LSTM(NF, hidden, layers, batch_first=True), SN(nn.Linear(hidden, 1))

    def forward(self, x):
        return self.fc(self.lstm(x)[0][:, -1])


class TemporalDiscriminator(LSTMCritic):
    def __init__(self): super().__init__(256, 2)


class StatDiscriminator(LSTMCritic):
    def __init__(self): super().__init__(512, 3)


class ConvHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1, self.c2, self.fc = CircularConv1d(NF, 128, 5), CircularConv1d(128, 64, 3), SN(nn.Linear(64, 1))

    def head(self, s):                                     # s: (B, F, T)
        s = F.leaky_relu(self.c2(F.leaky_relu(self.c1(s), 0.2)), 0.2)
        return self.fc(s.max(2)[0])


class SortedDiscriminator(ConvHead):
    """D_sort: each feature sorted independently over time (marginal/quantile channel only)."""
    def forward(self, x):
        return self.head(torch.sort(x, dim=1)[0].transpose(1, 2))


def soft_rank(x, anchors=32, tau=0.05):
    """Differentiable rank against a fixed (detached) quantile grid; keeps the row pairing."""
    lv = torch.linspace(0, 1, anchors, device=x.device)
    ref = torch.quantile(x.float(), lv, dim=1).permute(1, 0, 2).to(x.dtype).detach()   # (B,K,F)
    return torch.sigmoid((x.unsqueeze(2) - ref.unsqueeze(1)) / tau).mean(2)           # (B,T,F)


class RankSortedDiscriminator(ConvHead):
    """Copula-preserving variant of D_sort: soft-rank input instead of independent sort."""
    def forward(self, x):
        return self.head((2 * soft_rank(x) - 1).transpose(1, 2))


class MomentDiscriminator(nn.Module):
    """D_stat on [mean, var, skew, excess kurtosis] per feature (fp32)."""
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(4 * NF)
        self.net = nn.Sequential(SN(nn.Linear(4 * NF, 256)), nn.LeakyReLU(0.2),
                                 SN(nn.Linear(256, 128)), nn.LeakyReLU(0.2), SN(nn.Linear(128, 1)))

    def forward(self, x):
        with torch.autocast(x.device.type, enabled=False):
            x = x.float(); mu = x.mean(1); var = x.var(1, unbiased=False)
            z = ((x - mu.unsqueeze(1)) / var.clamp_min(1e-8).sqrt().unsqueeze(1)).clamp(-5, 5)
            return self.net(self.norm(torch.cat([mu, var, (z ** 3).mean(1), (z ** 4).mean(1) - 3], -1)))


def n_params(m):
    return sum(p.numel() for p in m.parameters())
