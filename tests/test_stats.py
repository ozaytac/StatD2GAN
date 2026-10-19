import numpy as np, pandas as pd, pytest
pytest.importorskip("statsmodels")
from s2g import stats as S


def holm_ref(p):
    """Independent reference: sort, multiply by (m-i), running maximum, clip, unsort."""
    p = np.asarray(p, float); m = len(p); o = np.argsort(p)
    adj = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * p[o]))
    out = np.empty(m); out[o] = adj
    return out


def test_holm_known_nonmonotone_case():
    p = [0.005, 0.01, 0.03, 0.04]                # naive step-down: .02 .03 .06 .04 (not monotone)
    assert np.allclose(sorted(S.holm(p)), [0.02, 0.03, 0.06, 0.06])


def test_holm_is_monotone_and_matches_reference():
    r = np.random.default_rng(0)
    for _ in range(200):
        p = r.uniform(0, 0.2, r.integers(2, 9))
        a = S.holm(p)
        assert np.allclose(a, holm_ref(p))
        assert (np.diff(a[np.argsort(p)]) >= -1e-12).all()
        assert (a >= p - 1e-12).all() and (a <= 1).all()


def _frame(effect=0.1, n_loc=5, seeds=(1, 2, 3, 4, 5), arms=("a", "b"), seed=0):
    r = np.random.default_rng(seed); rows = []
    for l in range(n_loc):
        for s in seeds:
            base = r.normal(0.2, 0.02)
            rows.append(dict(loc=f"L{l}", seed=s, config="full_model", m=base))
            rows.append(dict(loc=f"L{l}", seed=s, config="a", m=base + effect + r.normal(0, 0.01)))
            rows.append(dict(loc=f"L{l}", seed=s, config="b", m=base + r.normal(0, 0.01)))
    return pd.DataFrame(rows)


def test_contrast_table_detects_effect_only_in_a():
    t = S.contrast_table(_frame(), ["a", "b"], metrics=["m"])
    a, b = t.set_index("contrast").loc["a"], t.set_index("contrast").loc["b"]
    assert a.n == 25 and a.p_holm < 0.001 and a.n_loc_median_pos == 5 and b.p_holm > 0.05


def test_bootstrap_statistic_and_reproducibility():
    d = S.paired_diffs(_frame(), "m", "a")
    pt, lo, hi = S.clustered_bootstrap(d, n_boot=2000, seed=1)
    assert np.isclose(pt, d.groupby(level=0).median().mean())
    assert lo <= pt <= hi and lo > 0
    assert S.clustered_bootstrap(d, 2000, 1) == (pt, lo, hi)
