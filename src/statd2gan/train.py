"""Core training loop (AMP + GPU-resident batches + lazy gradient penalty),
sample generation, and post-hoc rescoring from saved checkpoints.

Training has two phases: a CDF/quantile-matching warmup (no adversarial
signal yet), then adversarial training with WGAN-GP critics, evolutionary
loss-weight adaptation, and a lazily-computed gradient penalty (evaluated
every `gp_freq` critic steps, weight rescaled so the expected regularization
pressure is unchanged — roughly a 5x speedup at `gp_freq=5`).
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch
from torch.optim import Adam

from .config import C, ExperimentConfig, LocationPaths, set_seed
from .data import denorm
from .losses import (EvolutionaryWeights, cdf_loss, make_scaler_tensors,
                      physics_loss, quantile_loss, sample_z)
from .metrics import compute_metrics, fit_calibration, phys_viol_rate, project_physics
from .models import (Generator, MomentDiscriminator, RankSortedDiscriminator,
                      SortedDiscriminator, StatDiscriminator, TemporalDiscriminator,
                      count_params)


def _build_discriminators(cfg: ExperimentConfig, device: str):
    stat_cls = MomentDiscriminator if cfg.stat_variant == "moment" else StatDiscriminator
    sorted_cls = RankSortedDiscriminator if cfg.sorted_variant == "rank" else SortedDiscriminator

    heads = []
    if cfg.use_D_temporal:
        heads.append(TemporalDiscriminator())
    if cfg.use_D_stat:
        heads.append(stat_cls())
    if cfg.use_D_sorted:
        heads.append(sorted_cls())
    return [h.to(device) for h in heads]


def _z_training_mixture(bsz: int, noise_dim: int, device: str) -> torch.Tensor:
    """Fixed end-of-training latent mixture (frac=0.3, tail scale x3),
    used for evaluation draws that should match the late-training regime."""
    n_tail = int(bsz * 0.3)
    return torch.cat([torch.randn(bsz - n_tail, noise_dim, device=device),
                      torch.randn(n_tail, noise_dim, device=device) * 3.0])


def _generate(G: Generator, noise_dim: int, device: str, n: int = 5000,
              eval_z: str = "standard") -> np.ndarray:
    G.eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, n, 1000):
            b = min(1000, n - i)
            z = (_z_training_mixture(b, noise_dim, device) if eval_z == "training_mixture"
                 else torch.randn(b, noise_dim, device=device))
            with torch.amp.autocast("cuda", enabled=C.use_amp):
                out = G(z)
            chunks.append(out.float().cpu().numpy())
    return np.concatenate(chunks)


def train_experiment(cfg: ExperimentConfig, seed: int, device: str, paths: LocationPaths,
                      train_data: torch.Tensor, real_train_phys: np.ndarray,
                      real_val_phys: np.ndarray, scaler, verbose: bool = True) -> dict:
    """Train one (config, seed) arm end to end and return its held-out metrics.

    Saves the generator checkpoint and a companion fitlog (per-epoch fitness,
    gradient-penalty, and lambda trajectories) to `paths.gen_root` before
    computing any metric, so a crash during evaluation never loses a
    completed training run (see the `finally` block for GPU cleanup).
    """
    set_seed(seed)
    gp_freq = cfg.gp_freq if cfg.gp_freq > 0 else C.gp_freq

    # GPU-resident data, bypassing DataLoader: worker overhead exceeds the
    # benefit at this dataset size (tens of MB).
    data = train_data[:cfg.n_samples] if cfg.n_samples else train_data
    data_gpu = data.to(device, non_blocking=True).contiguous()
    n_train = data_gpu.size(0)
    # n_train < batch_size used to make n_batch=0 (no training at all) — this
    # broke the n_samples=500 sensitivity arm silently until fixed here.
    bsz_eff = min(C.batch_size, n_train)
    n_batch = max(1, n_train // bsz_eff)
    perm_gen = torch.Generator(device=device).manual_seed(seed)

    def epoch_batches():
        perm = torch.randperm(n_train, device=device, generator=perm_gen)
        for b in range(n_batch):
            yield data_gpu[perm[b * bsz_eff:(b + 1) * bsz_eff]]

    G = Generator(cfg.noise_dim).to(device)
    if C.use_compile:
        try:
            G = torch.compile(G, mode="reduce-overhead")
        except Exception as e:
            if verbose:
                print(f"  compile(G) skipped: {e}")

    Ds = _build_discriminators(cfg, device)
    opts_d = [Adam(d.parameters(), lr=C.lr_d, betas=(0., 0.9)) for d in Ds]
    opt_g = Adam(G.parameters(), lr=C.lr_g, betas=(0., 0.9))
    sc, mn = make_scaler_tensors(scaler, device)

    scaler_g = torch.amp.GradScaler("cuda", enabled=C.use_amp)
    scaler_d = torch.amp.GradScaler("cuda", enabled=C.use_amp)

    t0 = time.time()
    fit_log, gp_log, lam_log = [], [], []
    evo = EvolutionaryWeights(n=len(Ds)) if cfg.use_evolutionary else None
    weights = (evo.weights.copy() if evo
               else np.array((cfg.fixed_lambda or [1.0] * len(Ds))[:len(Ds)], dtype=np.float32))

    try:
        # ---- Phase 1: CDF/quantile warmup (no adversarial signal yet) ----
        if verbose:
            print("  Warmup...", end=" ", flush=True)
        for ep in range(cfg.warmup_epochs):
            G.train()
            for real in epoch_batches():
                z = torch.randn(real.size(0), cfg.noise_dim, device=device)
                opt_g.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=C.use_amp):
                    fake = G(z)
                    loss = physics_loss(fake, sc, mn)
                    if cfg.use_aux_losses:
                        loss = loss + cdf_loss(real, fake) + quantile_loss(real, fake)
                scaler_g.scale(loss).backward()
                scaler_g.unscale_(opt_g)
                torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
                scaler_g.step(opt_g)
                scaler_g.update()
        if verbose:
            print("done.")

        # ---- Phase 2: adversarial training (WGAN-GP + evolutionary weights, lazy GP) ----
        if verbose:
            print(f"  Adversarial (gp_freq={gp_freq})...", end=" ", flush=True)
        gp_lambda_lazy = 10.0 * gp_freq  # rescaled so expected reg pressure is unchanged
        batch_idx = 0

        for ep in range(cfg.warmup_epochs, cfg.total_epochs):
            G.train()
            for d in Ds:
                d.train()
            if ep == 30:
                for pg in opt_g.param_groups:
                    pg["lr"] = C.lr_g / 10
            n_critic = 2 if ep > 50 else 3

            ep_fits = []
            ep_gps = [[] for _ in Ds]
            for real in epoch_batches():
                bsz = real.size(0)
                do_gp = (batch_idx % gp_freq == 0)
                batch_idx += 1

                d_fits = []
                for _ in range(n_critic):
                    z = sample_z(bsz, cfg.noise_dim, ep, device)
                    with torch.amp.autocast("cuda", enabled=C.use_amp):
                        fake = G(z).detach()
                    fake_fp32 = fake.float() if do_gp else None
                    batch_fits = []

                    for k, (d, opt) in enumerate(zip(Ds, opts_d)):
                        opt.zero_grad(set_to_none=True)
                        with torch.amp.autocast("cuda", enabled=C.use_amp):
                            loss_d = -(d(real).mean() - d(fake).mean())

                        # spec B2c escape hatch: gp_on_moment=False lets
                        # MomentDiscriminator rely on spectral norm alone.
                        skip_gp = (not cfg.gp_on_moment) and isinstance(d, MomentDiscriminator)
                        if do_gp and not skip_gp:
                            alpha = torch.rand(bsz, 1, 1, device=device)
                            interp = (alpha * real + (1 - alpha) * fake_fp32).requires_grad_(True)
                            with torch.amp.autocast("cuda", enabled=False):
                                with torch.backends.cudnn.flags(enabled=False):
                                    di = d(interp)
                                grads = torch.autograd.grad(di.sum(), interp, create_graph=True)[0]
                                gp = ((grads.norm(2, dim=[1, 2]) - 1) ** 2).mean()
                            total_d = loss_d + gp_lambda_lazy * gp
                            ep_gps[k].append(gp.detach())
                        else:
                            total_d = loss_d

                        scaler_d.scale(total_d).backward()
                        scaler_d.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(d.parameters(), 1.0)
                        scaler_d.step(opt)
                        batch_fits.append((-loss_d).detach())  # accumulate tensor, one sync/batch
                    scaler_d.update()  # one update after all discriminator steps (multi-opt AMP pattern)
                    d_fits.append(batch_fits)

                fits = torch.stack([torch.stack(b) for b in d_fits]).float().cpu().numpy()
                ep_fits.append(np.abs(fits.mean(axis=0)))
                if evo:
                    weights = evo.update(ep_fits[-1])

                # ---- Generator update ----
                z = sample_z(bsz, cfg.noise_dim, ep, device)
                opt_g.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=C.use_amp):
                    fake = G(z)
                    loss_adv = sum(-w * d(fake).mean() for w, d in zip(weights, Ds))
                    loss_phys = 0.1 * physics_loss(fake, sc, mn)
                    loss_aux = 0.0
                    # warmup_epochs=0 alone is not enough to disable aux
                    # losses: they return at ep>=35 unless use_aux_losses=False.
                    if ep >= 35 and cfg.use_aux_losses:
                        aux_w = min(0.01, 0.01 * (ep - 35) / 15)
                        loss_aux = aux_w * (cdf_loss(real, fake) + quantile_loss(real, fake))
                    # WGAN objective: minimize -sum_k lambda_k E[D_k(G(z))].
                    loss_G = loss_adv + loss_phys + loss_aux
                scaler_g.scale(loss_G).backward()
                scaler_g.unscale_(opt_g)
                torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
                scaler_g.step(opt_g)
                scaler_g.update()

            fit_log.append(np.mean(ep_fits, axis=0).tolist())
            gp_log.append([float(torch.stack(g).mean()) if g else None for g in ep_gps])
            lam_log.append(np.asarray(weights, dtype=float).tolist())
        if verbose:
            print("done.")

        # ---- Persist checkpoint + fitlog before computing any metric ----
        gen_path = f"{paths.gen_root}/{cfg.name}_{seed}_{cfg.total_epochs}.pt"
        torch.save({k: v.half() for k, v in G.state_dict().items()}, gen_path)
        with open(gen_path.replace(".pt", "_fitlog.json"), "w") as f:
            json.dump({"d_names": [type(d).__name__ for d in Ds],
                      "fit_per_epoch": fit_log,
                      "gp_per_epoch": gp_log,
                      "lambda_per_epoch": lam_log,
                      "train_seconds": round(time.time() - t0, 1),
                      "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                      "param_counts": {type(m).__name__: count_params(m) for m in [G] + Ds}}, f)

        # ---- Held-out evaluation (spec A1): calibration fit on train, metrics on val ----
        syn_norm = _generate(G, cfg.noise_dim, device)
        syn_phys = denorm(syn_norm, scaler, C.scalable_idx)
        # Raw (pre-calibration) violation rate; the amount calibration
        # injects is `phys_viol_rate - phys_viol_raw`, cheap enough (one
        # mask) to compute on every run.
        extra = {"phys_viol_raw": phys_viol_rate(syn_phys)}
        if cfg.use_calibration:
            phi, tie_frac = fit_calibration(real_train_phys, syn_phys)
            extra["tie_frac_mean"] = float(np.mean(list(tie_frac().values())))
            syn_phys = phi(syn_phys)
        m = compute_metrics(real_val_phys, syn_phys)
        return {**m, **extra, "train_seconds": round(time.time() - t0, 1)}

    finally:
        # An exception mid-run leaves IPython traceback references holding
        # GPU memory alive; explicit cleanup prevents the next run from OOMing.
        del data_gpu, G, Ds
        torch.cuda.empty_cache()


def rescore(cfg: ExperimentConfig, seed: int, device: str, paths: LocationPaths,
            real_train_phys: np.ndarray, real_val_phys: np.ndarray, scaler,
            n: int = 5000, calibration_fit: str = "train", eval_z: str = "standard",
            projection: bool = False, metric_fn=None) -> dict:
    """Recompute metrics from a saved checkpoint with no retraining.

    `calibration_fit='train'` is the honest, paper-reported protocol;
    `'val'` is an oracle upper bound (spec A4a), diagnostic only and never
    reported as a headline number. The gap between the two is the transfer
    cost of calibration.
    """
    gen_path = f"{paths.gen_root}/{cfg.name}_{seed}_{cfg.total_epochs}.pt"
    G = Generator(cfg.noise_dim).to(device)
    sd = torch.load(gen_path, map_location=device)
    G.load_state_dict({k: v.float() for k, v in sd.items()})

    set_seed(seed)  # same z-stream as training -> reproducible synthetic draw
    syn_phys = denorm(_generate(G, cfg.noise_dim, device, n=n, eval_z=eval_z), scaler, C.scalable_idx)
    extra = {"phys_viol_raw": phys_viol_rate(syn_phys)}
    if cfg.use_calibration:
        fit_ref = real_train_phys if calibration_fit == "train" else real_val_phys
        phi, tie_frac = fit_calibration(fit_ref, syn_phys)
        extra["tie_frac_mean"] = float(np.mean(list(tie_frac().values())))
        extra["tie_frac_per_feat"] = tie_frac()
        syn_phys = phi(syn_phys)
    if projection:
        syn_phys = project_physics(syn_phys)
    fn = metric_fn or compute_metrics
    del G
    torch.cuda.empty_cache()
    return {**fn(real_val_phys, syn_phys), **extra, "calibration_fit": calibration_fit, "eval_z": eval_z}


def rescore_independent(cfg: ExperimentConfig, seed: int, device: str, paths: LocationPaths,
                         real_train_phys: np.ndarray, real_val_phys: np.ndarray, scaler,
                         n: int = 5000, calibration_fit: str = "train", eval_z: str = "standard",
                         projection: bool = False, metric_fn=None,
                         calib_seed_offset: int = 1_000_000) -> dict:
    """Independent-draw variant of `rescore`, used as a robustness check.

    `rescore` draws a single synthetic sample and reuses it both to fit the
    calibration map's synthetic-side quantiles and as the final evaluation
    set — the real side is properly split (train for fit, val for eval) but
    the synthetic side is not. This variant draws two independent samples
    from the same saved generator (different seed stream, same n and eval_z
    convention): one dedicated to fitting calibration, one to evaluation.
    Same interface and return keys as `rescore`, plus `independent_draws: True`.
    Requires no retraining, only the saved checkpoint.
    """
    gen_path = f"{paths.gen_root}/{cfg.name}_{seed}_{cfg.total_epochs}.pt"
    G = Generator(cfg.noise_dim).to(device)
    sd = torch.load(gen_path, map_location=device)
    G.load_state_dict({k: v.float() for k, v in sd.items()})

    set_seed(seed)
    syn_phys_eval = denorm(_generate(G, cfg.noise_dim, device, n=n, eval_z=eval_z), scaler, C.scalable_idx)
    extra = {"phys_viol_raw": phys_viol_rate(syn_phys_eval), "independent_draws": True}

    if cfg.use_calibration:
        set_seed(seed + calib_seed_offset)
        syn_phys_calib = denorm(_generate(G, cfg.noise_dim, device, n=n, eval_z=eval_z), scaler, C.scalable_idx)
        fit_ref = real_train_phys if calibration_fit == "train" else real_val_phys
        phi, tie_frac = fit_calibration(fit_ref, syn_phys_calib)
        extra["tie_frac_mean"] = float(np.mean(list(tie_frac().values())))
        extra["tie_frac_per_feat"] = tie_frac()
        syn_phys_eval = phi(syn_phys_eval)  # map fit on draw 2, applied to draw 1

    if projection:
        syn_phys_eval = project_physics(syn_phys_eval)
    fn = metric_fn or compute_metrics
    del G
    torch.cuda.empty_cache()
    return {**fn(real_val_phys, syn_phys_eval), **extra, "calibration_fit": calibration_fit, "eval_z": eval_z}
