import numpy as np, pytest
from s2g import metrics as M
from s2g.config import PHY


def toy(n=300, seed=0):
    r = np.random.default_rng(seed)
    x = np.zeros((n, 168, 11), np.float32)
    x[..., 0] = r.normal(10, 5, (n, 168))
    x[..., 1] = x[..., 0] - np.abs(r.normal(2, 1, (n, 168)))
    x[..., 2] = r.normal(1013, 8, (n, 168)).clip(985, 1040)
    x[..., 3] = np.abs(r.normal(3, 1, (n, 168)))
    x[..., 6] = r.uniform(20, 100, (n, 168))
    x[..., 9] = np.abs(r.normal(0.5, 0.2, (n, 168)))
    x[..., 10] = r.normal(0, 1, (n, 168))
    return x


def test_acf_matches_direct_formula():
    x = np.random.default_rng(1).normal(size=(4, 168)).cumsum(1)
    a = M.acf(x, 24)
    for i in range(4):
        d = x[i] - x[i].mean()
        ref = [np.sum(d[: len(d) - k] * d[k:]) / np.sum(d * d) for k in range(25)]
        assert np.allclose(a[i], ref, atol=1e-10)


def test_phys_viol_and_projection():
    x = toy()
    x[0, 0, 1] = x[0, 0, 0] + 5; x[1, 0, 2] = 900; x[2, 0, 6] = 120
    assert M.phys_viol_rate(x) > 0
    p = M.project_physics(x)
    assert M.phys_viol_rate(p) == 0
    assert np.array_equal(M.project_physics(p), p)                   # idempotent
    ok = toy(); ok[..., 1] = ok[..., 0] - 1
    assert np.array_equal(M.project_physics(ok), ok)                 # no-op on valid data


def test_metrics_identity():
    x = toy()
    m = M.compute_metrics(x, x)
    assert m["ks_stat"] == 0 and m["phys_viol_rate"] == 0
    assert m["kendall_mae"] < 0.02 and m["acf_mae"] < 0.03


def test_calibration_fixes_marginals_and_reports_ties():
    real, syn = toy(seed=1), toy(seed=2)
    syn[..., 0] = syn[..., 0] * 0.5 + 8                              # wrong scale/location
    phi, tie = M.fit_calibration(real, syn)
    before = M.compute_metrics(real, syn)["ks_stat"]
    after = M.compute_metrics(real, phi(syn))["ks_stat"]
    assert after < before / 2 and 0 <= tie <= 1


def test_evaluate_stages_and_no_calibration_arm():
    real, s1, s2 = toy(seed=1), toy(seed=2), toy(seed=3)
    o = M.evaluate(real, real, s1, s2, calibrate=True)
    assert set(o) >= {"raw", "cal", "proj", "seq", "tie_frac_mean"} and o["proj"]["phys_viol_rate"] == 0
    n = M.evaluate(real, real, s1, s2, calibrate=False)
    assert n["cal"] == n["raw"] and "tie_frac_mean" not in n


def test_seq_stats():
    x = toy(); x[..., 0] = np.arange(300)[:, None]                   # constant within sequence
    s = M.seq_stats(x)
    assert s["within_sd"] == 0 and s["step_diff"] == 0 and s["seq_mean_sd"] > 80


def test_floor_real_vs_real_split_is_small():
    x = toy(600)
    f = M.ks_floor(x[:300], x[300:])
    assert max(f.values()) < 0.06
