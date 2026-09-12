"""Held-out evaluation metrics, isotonic calibration, and diagnostics (v2 protocol).

The primary KS metric is the sup-distance (`ks_stat`), a sample-size-independent
effect size; KS pass rate is retained only as a secondary, fixed-subsample
diagnostic (spec D3) and is superseded by `ks_stat` for any headline claim
(see the paper's evidence rules — pass-rate values from the pre-v2 campaign
must not be cited).
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.stats import kendalltau
from statsmodels.tsa.stattools import acf as compute_acf
from sklearn.isotonic import IsotonicRegression

from .config import C

METRIC_COLS = ["ks_stat", "ks_pass_rate", "kendall_mae", "acf_mae", "phys_viol_rate"]
COL_LABELS = ["KS sup-dist (down)", "KS pass rate (up)", "Kendall tau MAE (down)",
              "ACF MAE (down)", "Phys. violation rate (down)"]

_ACF_FEATURES = [0, 1, 2, 3, 6]
_TAU_PAIRS = [(0, 1), (0, 6), (2, 9), (6, 9), (0, 2), (1, 6)]


def phys_viol_rate(syn_phys: np.ndarray) -> float:
    """Fraction of (sequence, timestep) cells violating at least one physics
    constraint. Single source of truth shared by training diagnostics and
    the raw/calibrated violation-rate comparison in the analysis pipeline."""
    t, dew = syn_phys[..., 0], syn_phys[..., 1]
    pres, hum = syn_phys[..., 2], syn_phys[..., 6]
    mask = ((dew > t + C.dew_tol) | (pres > C.pres_hi) | (pres < C.pres_lo)
            | (hum > C.hum_hi) | (hum < C.hum_lo))
    return float(mask.mean())


def project_physics(syn_phys: np.ndarray) -> np.ndarray:
    """Post-calibration constraint projection.

    Pilot finding: physics violations are injected by the marginal-independent
    isotonic calibration map, not by the generator itself. This hard
    projection is applied for the paper's "projection" pipeline decision
    (see `analysis.projection_analysis`) and eliminates violations at
    negligible KS cost.
    """
    out = syn_phys.copy()
    out[..., 1] = np.minimum(out[..., 1], out[..., 0] + C.dew_tol)
    out[..., 6] = np.clip(out[..., 6], C.hum_lo, C.hum_hi)
    out[..., 2] = np.clip(out[..., 2], C.pres_lo, C.pres_hi)
    return out


def compute_metrics(real_phys: np.ndarray, syn_phys: np.ndarray) -> dict:
    """Compute the full metric suite for one synthetic sample.

    `real_phys` must be the held-out validation split (spec A1). All
    stochastic subsampling uses `C.metric_seed`, so metric noise never mixes
    with seed-to-seed training variance (spec D1).
    """
    rng = np.random.default_rng(C.metric_seed)
    n_real, t_len, n_feat = real_phys.shape

    # 1. KS: sup-distance (primary) + pass rate at a fixed subsample (secondary, spec D3)
    ks_stats, ks_pass = [], 0
    for f in C.scalable_idx:
        r, s = real_phys[..., f].ravel(), syn_phys[..., f].ravel()
        ks_stats.append(stats.ks_2samp(r, s).statistic)
        n = min(C.ks_subsample, len(r), len(s))
        r_i = rng.choice(len(r), n, replace=False)
        s_i = rng.choice(len(s), n, replace=False)
        ks_pass += stats.ks_2samp(r[r_i], s[s_i]).pvalue > 0.05
    ks_stat = float(np.mean(ks_stats))
    ks_rate = ks_pass / len(C.scalable_idx)

    # 2. Kendall tau MAE — fixed-seed subsample of size C.tau_subsample (spec D1)
    r_flat = real_phys.reshape(-1, n_feat)
    s_flat = syn_phys.reshape(-1, n_feat)
    r_sub = r_flat[rng.choice(len(r_flat), min(C.tau_subsample, len(r_flat)), replace=False)]
    s_sub = s_flat[rng.choice(len(s_flat), min(C.tau_subsample, len(s_flat)), replace=False)]
    kendall_mae = float(np.mean([
        abs(kendalltau(r_sub[:, i], r_sub[:, j])[0] - kendalltau(s_sub[:, i], s_sub[:, j])[0])
        for i, j in _TAU_PAIRS
    ]))

    # 3. ACF MAE — computed per sequence, then averaged (spec D2: never on a raveled array)
    def mean_acf(arr: np.ndarray, feat: int, n_seq: int) -> np.ndarray:
        per_seq = np.stack([compute_acf(arr[k, :, feat], nlags=24, fft=True) for k in range(n_seq)])
        return per_seq.mean(0)

    n_seq = min(200, n_real, syn_phys.shape[0])
    acf_mae = float(np.mean([
        np.mean(np.abs(mean_acf(real_phys, f, n_seq) - mean_acf(syn_phys, f, n_seq)))
        for f in _ACF_FEATURES
    ]))

    # 4. Physics violation rate — indicator function, same constants as training (spec C2)
    phys_viol = phys_viol_rate(syn_phys)

    return dict(ks_stat=ks_stat, ks_pass_rate=ks_rate, kendall_mae=kendall_mae,
                acf_mae=acf_mae, phys_viol_rate=phys_viol)


def fit_calibration(real_fit_phys: np.ndarray, syn_phys: np.ndarray, n_q: int = 1000):
    """Fit an isotonic quantile map from synthetic to real marginals.

    Held-out protocol (spec A1): the fit reference is the *train* split;
    metrics are always measured on val. Returns `(phi, tie_frac)`:

    - `phi(x)`: applies the fitted map to any array of the same shape.
    - `tie_frac()`: the fraction of quantile mass where the isotonic map is
      flat (spec A4b) — the mechanism by which calibration can alter Kendall
      tau via tie injection. Flat segments come from mass concentration in
      the *target* (train) distribution (e.g. RH saturating at 100%); this
      is distinct from clipping, which is zero by construction here and
      carries no diagnostic value.
    """
    q_grid = np.linspace(0, 1, n_q)
    maps = {}
    for f in C.scalable_idx:
        r_q = np.quantile(real_fit_phys[..., f].ravel(), q_grid)
        s_q = np.quantile(syn_phys[..., f].ravel(), q_grid)
        ir = IsotonicRegression(out_of_bounds="clip").fit(s_q, r_q)
        maps[f] = (ir, s_q)

    def phi(x: np.ndarray) -> np.ndarray:
        out = x.copy()
        for f, (ir, _) in maps.items():
            out[..., f] = ir.predict(out[..., f].ravel()).reshape(out[..., f].shape)
        return out

    def tie_frac() -> dict:
        out = {}
        for f, (ir, s_q) in maps.items():
            y_q = ir.predict(s_q)
            out[f] = float((np.diff(y_q) == 0).mean())
        return out

    return phi, tie_frac


def ks_noise_floor(real_train_phys: np.ndarray, real_val_phys: np.ndarray) -> dict:
    """Real-vs-real noise floor (spec D3b): the KS sup-distance no generator
    can beat, since it reflects train/val distribution drift rather than
    any generator shortcoming. Computed once per location."""
    return {f: float(stats.ks_2samp(real_train_phys[..., f].ravel(),
                                    real_val_phys[..., f].ravel()).statistic)
            for f in C.scalable_idx}


def bootstrap_ci(arr: np.ndarray, n_boot: int = 2000, alpha: float = 0.05):
    """Percentile bootstrap CI for the mean of `arr`."""
    boot = [np.mean(np.random.choice(arr, len(arr), replace=True)) for _ in range(n_boot)]
    return np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
