"""RCGAN (Esteban et al. 2017) and TimeGAN (Yoon et al. 2019), re-implemented compactly.
Both output scaled sequences in [-1, 1] and go through the same calibration/projection/metrics
pipeline as StatD2GAN (train.finish)."""
import time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from . import data as D
from .config import P, ROOT
from .train import DEV, set_seed, finish

NF, L, B = len(P["data"]["features"]), P["data"]["seq_len"], P["baselines"]


def _batch(data, bs, g):
    return data[torch.randint(len(data), (bs,), device=DEV, generator=g)]


class RCGen(nn.Module):
    def __init__(self, c):
        super().__init__(); self.c = c
        self.rnn, self.out = nn.LSTM(c["latent"], c["hidden"], c["layers"], batch_first=True), nn.Linear(c["hidden"], NF)

    def noise(self, b, g):                                   # per-step noise, as in RCGAN
        return torch.randn(b, L, self.c["latent"], device=DEV, generator=g)

    def forward(self, z):
        return torch.tanh(self.out(self.rnn(z)[0]))


def train_rcgan(data, seed):
    c = B["rcgan"]; set_seed(seed); g = torch.Generator(device=DEV).manual_seed(seed)
    G = RCGen(c).to(DEV)
    Dn = nn.ModuleDict(dict(r=nn.LSTM(NF, c["hidden"], c["layers"], batch_first=True), o=nn.Linear(c["hidden"], 1))).to(DEV)
    dis = lambda x: Dn["o"](Dn["r"](x)[0])                   # per-timestep logits
    og, od = torch.optim.Adam(G.parameters(), c["lr"], betas=(.5, .999)), torch.optim.Adam(Dn.parameters(), c["lr"], betas=(.5, .999))
    bce, t0 = F.binary_cross_entropy_with_logits, time.time()
    for ep in range(c["epochs"]):
        for _ in range(max(1, len(data) // c["batch_size"])):
            real = _batch(data, c["batch_size"], g)
            z = torch.randn(c["batch_size"], L, c["latent"], device=DEV, generator=g)   # per-step noise, as in RCGAN
            fake = G(z)
            od.zero_grad(); lr_, lf = dis(real), dis(fake.detach())
            (bce(lr_, torch.ones_like(lr_)) + bce(lf, torch.zeros_like(lf))).backward(); od.step()
            og.zero_grad(); lf = dis(fake); bce(lf, torch.ones_like(lf)).backward(); og.step()
    return G, dict(train_seconds=round(time.time() - t0, 1), params=dict(RCGen=sum(p.numel() for p in G.parameters()),
                   Disc=sum(p.numel() for p in Dn.parameters())))


class TimeGAN(nn.Module):
    def __init__(self, c):
        super().__init__(); h, n, z = c["hidden"], c["layers"], c["latent"]
        gru = lambda i, o, k=n: nn.GRU(i, o, k, batch_first=True)
        self.E, self.R, self.G, self.S, self.D = gru(NF, h), gru(h, h), gru(z, h), gru(h, h, n - 1), gru(h, h)
        self.eo, self.ro, self.go, self.so, self.do = (nn.Linear(h, h), nn.Linear(h, NF), nn.Linear(h, h),
                                                       nn.Linear(h, h), nn.Linear(h, 1))
        self.zdim = z

    def emb(self, x): return torch.sigmoid(self.eo(self.E(x)[0]))
    def rec(self, h): return torch.sigmoid(self.ro(self.R(h)[0]))
    def gen(self, z): return torch.sigmoid(self.go(self.G(z)[0]))
    def sup(self, h): return torch.sigmoid(self.so(self.S(h)[0]))
    def dis(self, h): return self.do(self.D(h)[0])
    def noise(self, b, g):
        return torch.randn(b, L, self.zdim, device=DEV, generator=g)

    def forward(self, z):                                     # per-step noise -> scaled [-1,1] sequence
        return 2 * self.rec(self.sup(self.gen(z))) - 1


def train_timegan(data, seed):
    c = B["timegan"]; set_seed(seed); g = torch.Generator(device=DEV).manual_seed(seed)
    x01 = (data + 1) / 2; m = TimeGAN(c).to(DEV); bs = c["batch_size"]
    opt = lambda *ms: torch.optim.Adam([p for k in ms for p in k.parameters()], c["lr"])
    o_er, o_s, o_g, o_d = opt(m.E, m.R, m.eo, m.ro), opt(m.S, m.so), opt(m.G, m.S, m.go, m.so), opt(m.D, m.do)
    o_e = opt(m.E, m.R, m.eo, m.ro)
    bce, mse, t0 = F.binary_cross_entropy_with_logits, F.mse_loss, time.time()
    zs = lambda: torch.randn(bs, L, c["latent"], device=DEV, generator=g)
    for _ in range(c["iters_ae"]):                            # 1) autoencoder
        x = _batch(x01, bs, g); o_er.zero_grad(); (10 * mse(m.rec(m.emb(x)), x).sqrt()).backward(); o_er.step()
    for _ in range(c["iters_sup"]):                           # 2) supervisor on real latents
        x = _batch(x01, bs, g)
        with torch.no_grad(): h = m.emb(x)
        o_s.zero_grad(); mse(m.sup(h)[:, :-1], h[:, 1:]).backward(); o_s.step()
    for _ in range(c["iters_joint"]):                         # 3) joint
        for _ in range(2):
            x = _batch(x01, bs, g); h = m.emb(x); eh = m.gen(zs()); hh = m.sup(eh); xh = m.rec(hh)
            yf, yfe = m.dis(hh), m.dis(eh)
            gu = bce(yf, torch.ones_like(yf)) + c["gamma"] * bce(yfe, torch.ones_like(yfe))
            gs = mse(m.sup(h)[:, :-1], h[:, 1:])
            gv = (torch.sqrt(xh.var(0, unbiased=False) + 1e-6) - torch.sqrt(x.var(0, unbiased=False) + 1e-6)).abs().mean() \
                + (xh.mean(0) - x.mean(0)).abs().mean()
            o_g.zero_grad(); (gu + 100 * gs.sqrt() + 100 * gv).backward(); o_g.step()
            h = m.emb(x); e_loss = 10 * mse(m.rec(h), x).sqrt() + 0.1 * mse(m.sup(h)[:, :-1], h[:, 1:])
            o_e.zero_grad(); e_loss.backward(); o_e.step()
        x = _batch(x01, bs, g)
        with torch.no_grad(): h = m.emb(x); eh = m.gen(zs()); hh = m.sup(eh)
        yr, yf, yfe = m.dis(h), m.dis(hh), m.dis(eh)
        dl = bce(yr, torch.ones_like(yr)) + bce(yf, torch.zeros_like(yf)) + c["gamma"] * bce(yfe, torch.zeros_like(yfe))
        if dl.item() > 0.15:                                  # as in the reference implementation
            o_d.zero_grad(); dl.backward(); o_d.step()
    return m, dict(train_seconds=round(time.time() - t0, 1), params=dict(TimeGAN=sum(p.numel() for p in m.parameters())))


def run_baseline(name, loc, seed, out_dir=None):
    out = Path(out_dir or ROOT / "runs") / loc / f"{loc}_{name}_seed{seed}"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.with_suffix(".json").exists():
        return None
    d = D.load(ROOT / f"data/processed/{loc}.npz")
    x = torch.from_numpy(d["scaler"].fwd(d["train_phys"])).float().to(DEV)
    G, info = (train_rcgan if name == "rcgan" else train_timegan)(x, seed)
    return finish(G, info, name, loc, seed, d, True, out)
