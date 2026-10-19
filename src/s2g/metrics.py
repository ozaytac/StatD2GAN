"""Evaluation metrics, calibration, projection (numpy/scipy only; no torch)."""
import numpy as np
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from .config import P, PHY, SC_IDX

E = P["evaluation"]


def phys_viol_rate(x):
    """Fraction of (sequence, time) cells violating at least one constraint. x: physical units."""
    t, dew, pres, h = x[..., 0], x[..., 1], x[..., 2], x[..., 6]
    bad = (dew > t + PHY["dew_tol"]) | (pres > PHY["pres_hi"]) | (pres < PHY["pres_lo"]) \
        | (h > PHY["hum_hi"]) | (h < PHY["hum_lo"])
    return float(bad.mean())


def project_physics(x):
    o = x.copy()
    o[..., 1] = np.minimum(o[..., 1], o[..., 0] + PHY["dew_tol"])
    o[..., 6] = np.clip(o[..., 6], PHY["hum_lo"], PHY["hum_hi"])
    o[..., 2] = np.clip(o[..., 2], PHY["pres_lo"], PHY["pres_hi"])
    return o


def acf(x, nlags):
    """Biased sample ACF along the last axis for a batch (same estimator as statsmodels acf,
    adjusted=False): r_k = sum (x_t-m)(x_{t+k}-m) / sum (x_t-m)^2."""
    x = x - x.mean(-1, keepdims=True)
    n = x.shape[-1]
    f = np.fft.rfft(x, 2 * n)
    c = np.fft.irfft(f * f.conj())[..., :nlags + 1]
    return c / np.where(c[..., :1] == 0, 1, c[..., :1])


def ks_per_feature(real, syn):
    return {f: float(stats.ks_2samp(real[..., f].ravel(), syn[..., f].ravel()).statistic) for f in SC_IDX}


def ks_floor(train_phys, val_phys):
    return ks_per_feature(train_phys, val_phys)


def compute_metrics(real_val, syn):
    """All metrics against the HELD-OUT real sequences; stochastic subsamples use metric_seed."""
    rng = np.random.default_rng(E["metric_seed"])
    ks = float(np.mean(list(ks_per_feature(real_val, syn).values())))
    rf, sf = real_val.reshape(-1, real_val.shape[-1]), syn.reshape(-1, syn.shape[-1])
    n = min(E["tau_subsample"], len(rf), len(sf))
    rs, ss = rf[rng.choice(len(rf), n, replace=False)], sf[rng.choice(len(sf), n, replace=False)]
    tau = float(np.mean([abs(stats.kendalltau(rs[:, i], rs[:, j])[0] - stats.kendalltau(ss[:, i], ss[:, j])[0])
                         for i, j in E["tau_pairs"]]))
    k = min(E["acf_n_seq"], len(real_val), len(syn))
    ri, si = rng.choice(len(real_val), k, replace=False), rng.choice(len(syn), k, replace=False)
    a = float(np.mean([np.abs(acf(real_val[ri][..., f], E["acf_nlags"]).mean(0)
                              - acf(syn[si][..., f], E["acf_nlags"]).mean(0)).mean()
                       for f in E["acf_features"]]))
    return dict(ks_stat=ks, kendall_mae=tau, acf_mae=a, phys_viol_rate=phys_viol_rate(syn))


def seq_stats(x):
    """Sequence-level statistics on temperature (feature 0). x: (N, T, F) physical units."""
    t = x[..., 0]
    return dict(seq_mean_sd=float(t.mean(1).std()), within_sd=float(t.std(1).mean()),
                step_diff=float(np.abs(np.diff(t, axis=1)).mean()))


def fit_calibration(real_train, syn, n_q=1000):
    """Isotonic quantile map per scalable feature, fit real TRAIN quantiles vs synthetic quantiles.
    Returns phi(x) and tie_frac (share of quantile mass on flat segments of the map)."""
    q = np.linspace(0, 1, n_q)
    maps = {}
    for f in SC_IDX:
        r_q, s_q = np.quantile(real_train[..., f].ravel(), q), np.quantile(syn[..., f].ravel(), q)
        maps[f] = (IsotonicRegression(out_of_bounds="clip").fit(s_q, r_q), s_q)

    def phi(x):
        o = x.copy()
        for f, (ir, _) in maps.items():
            o[..., f] = ir.predict(x[..., f].ravel()).reshape(x[..., f].shape)
        return o

    tie = float(np.mean([(np.diff(ir.predict(s_q)) == 0).mean() for ir, s_q in maps.values()]))
    return phi, tie


def evaluate(real_train, real_val, syn_eval, syn_calib, calibrate=True):
    """Pipeline on two independent synthetic draws. Returns stages raw / cal / proj + extras."""
    out = {"raw": compute_metrics(real_val, syn_eval)}
    if calibrate:
        phi, tie = fit_calibration(real_train, syn_calib)
        cal = phi(syn_eval)
        out["tie_frac_mean"] = tie
    else:
        cal = syn_eval
    out["cal"] = compute_metrics(real_val, cal)
    out["proj"] = compute_metrics(real_val, project_physics(cal))
    out["seq"] = seq_stats(cal)
    return out
