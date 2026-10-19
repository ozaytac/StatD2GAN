"""One StatD2GAN run: train -> save generator -> evaluate on independent draws -> write JSON once."""
import json, os, time, random
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np, torch
from torch.optim import Adam
from . import data as D, metrics as M_
from .config import P, ROOT
from .losses import EvolutionaryWeights, physics_loss, cdf_loss, quantile_loss, sample_z
from .models import (Generator, TemporalDiscriminator, StatDiscriminator, SortedDiscriminator,
                     MomentDiscriminator, RankSortedDiscriminator, n_params)

M, E = P["model"], P["evaluation"]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = M["amp"] and DEV.type == "cuda"


@dataclass
class Arm:
    name: str
    use_D_temporal: bool = True
    use_D_stat: bool = True
    use_D_sorted: bool = True
    use_calibration: bool = True
    use_evolutionary: bool = True
    fixed_lambda: list = None
    stat_variant: str = "lstm"
    sorted_variant: str = "sorted"
    use_aux_losses: bool = True
    warmup_epochs: int = M["warmup_epochs"]
    total_epochs: int = M["total_epochs"]


def arm(name, **override):
    return Arm(name, **{**P["arms"][name], **override})


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True


def draw(G, n, seed):
    """n synthetic sequences from a dedicated RNG stream (independent of training RNG).
    Baselines expose G.noise(b, gen) (per-step noise); the StatD2GAN generator takes z (B, latent)."""
    G.eval(); g = torch.Generator(device=DEV).manual_seed(seed); out = []
    amp = AMP and not hasattr(G, "noise")
    with torch.no_grad():
        for i in range(0, n, 1000):
            b = min(1000, n - i)
            z = G.noise(b, g) if hasattr(G, "noise") else torch.randn(b, M["latent_dim"], device=DEV, generator=g)
            with torch.autocast(DEV.type, enabled=amp):
                out.append(G(z).float().cpu().numpy())
    return np.concatenate(out)


def scaler_tensors(s):
    sc, mn = torch.ones(len(P["data"]["features"]), device=DEV), torch.zeros(len(P["data"]["features"]), device=DEV)
    for i, k in enumerate(P["data"]["scalable_idx"]):
        sc[k], mn[k] = float(s.scale[i]), float(s.min[i])
    return sc, mn


def train_generator(cfg: Arm, seed, train_scaled, scaler, log=None):
    """Returns (G, info). Loop is the notebook's: CDF warmup, then WGAN-GP with lazy GP."""
    set_seed(seed)
    data = torch.from_numpy(train_scaled).float().to(DEV).contiguous()
    n, bsz = len(data), min(M["batch_size"], len(data))
    n_batch = max(1, n // bsz)
    perm_gen = torch.Generator(device=DEV).manual_seed(seed)
    G = Generator().to(DEV)
    stat_cls = MomentDiscriminator if cfg.stat_variant == "moment" else StatDiscriminator
    sort_cls = RankSortedDiscriminator if cfg.sorted_variant == "rank" else SortedDiscriminator
    Ds = [c() for u, c in [(cfg.use_D_temporal, TemporalDiscriminator), (cfg.use_D_stat, stat_cls),
                           (cfg.use_D_sorted, sort_cls)] if u]
    Ds = [d.to(DEV) for d in Ds]
    opts = [Adam(d.parameters(), lr=M["lr_d"], betas=(0., .9)) for d in Ds]
    opt_g = Adam(G.parameters(), lr=M["lr_g"], betas=(0., .9))
    sc, mn = scaler_tensors(scaler)
    gs_g, gs_d = torch.amp.GradScaler(DEV.type, enabled=AMP), torch.amp.GradScaler(DEV.type, enabled=AMP)
    evo = EvolutionaryWeights(len(Ds)) if cfg.use_evolutionary else None
    lam = evo.w.copy() if evo else np.array((cfg.fixed_lambda or [1.] * len(Ds))[:len(Ds)], np.float32)
    ac = lambda: torch.autocast(DEV.type, enabled=AMP)
    gp_w = M["gp_lambda"] * M["gp_freq"]                 # lazy GP: same expected pressure
    epoch_s, fit_log, lam_log, k = [], [], [], 0

    def batches():
        p = torch.randperm(n, device=DEV, generator=perm_gen)
        for b in range(n_batch):
            yield data[p[b * bsz:(b + 1) * bsz]]

    def sync():
        if DEV.type == "cuda": torch.cuda.synchronize()

    for ep in range(cfg.total_epochs):
        t0 = time.time(); G.train(); [d.train() for d in Ds]
        if ep < cfg.warmup_epochs:                                       # phase 1: CDF warmup
            for real in batches():
                opt_g.zero_grad(set_to_none=True)
                with ac():
                    fake = G(torch.randn(len(real), M["latent_dim"], device=DEV))
                    loss = physics_loss(fake, sc, mn)
                    if cfg.use_aux_losses:
                        loss = loss + cdf_loss(real, fake) + quantile_loss(real, fake)
                gs_g.scale(loss).backward(); gs_g.unscale_(opt_g)
                torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0); gs_g.step(opt_g); gs_g.update()
        else:                                                            # phase 2: adversarial
            if ep == 30:
                for g in opt_g.param_groups: g["lr"] = M["lr_g"] / 10
            n_critic, ep_fit = (2 if ep > 50 else 3), []
            for real in batches():
                do_gp, k = (k % M["gp_freq"] == 0), k + 1
                fits = []
                for _ in range(n_critic):
                    with ac(): fake = G(sample_z(len(real), M["latent_dim"], ep, DEV)).detach()
                    bf = []
                    for d, opt in zip(Ds, opts):
                        opt.zero_grad(set_to_none=True)
                        with ac(): loss_d = -(d(real).mean() - d(fake).mean())
                        tot = loss_d
                        if do_gp:
                            a = torch.rand(len(real), 1, 1, device=DEV)
                            x = (a * real + (1 - a) * fake.float()).requires_grad_(True)
                            with torch.autocast(DEV.type, enabled=False), torch.backends.cudnn.flags(enabled=False):
                                g = torch.autograd.grad(d(x).sum(), x, create_graph=True)[0]
                            tot = loss_d + gp_w * ((g.norm(2, dim=[1, 2]) - 1) ** 2).mean()
                        gs_d.scale(tot).backward(); gs_d.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(d.parameters(), 1.0); gs_d.step(opt)
                        bf.append((-loss_d).detach())
                    gs_d.update(); fits.append(torch.stack(bf))
                f = torch.stack(fits).float().mean(0).abs().cpu().numpy(); ep_fit.append(f)
                if evo: lam = evo.update(f)
                opt_g.zero_grad(set_to_none=True)
                with ac():
                    fake = G(sample_z(len(real), M["latent_dim"], ep, DEV))
                    loss = sum(-float(w) * d(fake).mean() for w, d in zip(lam, Ds)) + 0.1 * physics_loss(fake, sc, mn)
                    if ep >= 35 and cfg.use_aux_losses:
                        loss = loss + min(0.01, 0.01 * (ep - 35) / 15) * (cdf_loss(real, fake) + quantile_loss(real, fake))
                gs_g.scale(loss).backward(); gs_g.unscale_(opt_g)
                torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0); gs_g.step(opt_g); gs_g.update()
            fit_log.append(np.mean(ep_fit, 0).round(5).tolist()); lam_log.append(np.asarray(lam, float).round(4).tolist())
        sync(); epoch_s.append(round(time.time() - t0, 3))
        if log and ep % 10 == 0: log(f"ep {ep} {epoch_s[-1]}s")
    info = dict(epoch_seconds=epoch_s, train_seconds=round(sum(epoch_s), 1), fit_per_epoch=fit_log,
                lambda_per_epoch=lam_log, params={type(m).__name__: n_params(m) for m in [G] + Ds})
    return G, info


def finish(G, info, cfg_name, loc, seed, d, calibrate, out: Path):
    """Shared by StatD2GAN and baselines: save weights, draw twice, evaluate, write JSON once."""
    torch.save(G.state_dict(), out.with_suffix(".pt"))
    t0 = time.time()
    s = d["scaler"]
    ev = s.inv(draw(G, E["n_eval"], E["eval_draw_seed"] + seed))
    ca = s.inv(draw(G, E["n_eval"], E["calib_draw_seed"] + seed))
    res = M_.evaluate(d["train_phys"], d["val_phys"], ev, ca, calibrate=calibrate)
    rec = dict(config=cfg_name, loc=loc, seed=seed, **info, eval_seconds=round(time.time() - t0, 1),
               gpu=torch.cuda.get_device_name(0) if DEV.type == "cuda" else "cpu", **res)
    tmp = out.with_suffix(".json.tmp"); tmp.write_text(json.dumps(rec)); os.replace(tmp, out.with_suffix(".json"))
    return rec


def run_one(name, loc, seed, out_dir=None, epochs=None, log=print):
    out = Path(out_dir or ROOT / "runs") / loc / f"{loc}_{name}_seed{seed}"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.with_suffix(".json").exists():
        return None                                              # runs are written once
    cfg = arm(name)
    if epochs:                                                   # smoke probe only: short run
        cfg.total_epochs, cfg.warmup_epochs = epochs, min(cfg.warmup_epochs, 1)
    d = D.load(ROOT / f"data/processed/{loc}.npz")
    G, info = train_generator(cfg, seed, d["scaler"].fwd(d["train_phys"]), d["scaler"], log)
    return finish(G, info, name, loc, seed, d, cfg.use_calibration, out)
