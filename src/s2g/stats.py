"""Paired inference: Wilcoxon signed-rank, Holm (monotone) via statsmodels, clustered bootstrap."""
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
from .config import P

I = P["inference"]


def holm(p):
    """Holm step-down adjusted p-values; statsmodels enforces monotonicity (running maximum)."""
    return multipletests(np.asarray(p, float), method="holm")[1]


def paired_diffs(df, metric, arm, ref="full_model", key=("loc", "seed"), col="config"):
    w = df.pivot_table(index=list(key), columns=col, values=metric)
    return (w[arm] - w[ref]).dropna()


def wilcoxon_p(d):
    try:
        return float(wilcoxon(d, alternative="two-sided", method="auto").pvalue)
    except ValueError:                                       # all differences zero
        return 1.0


def contrast_table(df, arms, metrics=None, ref="full_model", col="config", loc_col="loc"):
    """One row per (metric, arm): median diff, p, Holm p (family = arms within metric),
    per-location positive counts, and location medians (for decision rule D2)."""
    rows = []
    for m in metrics or I["metrics"]:
        r = []
        for a in arms:
            d = paired_diffs(df, m, a, ref, col=col)
            per = d.groupby(level=0)
            r.append(dict(metric=m, contrast=a, n=len(d), median_diff=float(d.median()),
                          p_wilcoxon=wilcoxon_p(d),
                          pos_by_loc=" ".join(f"{l[:2].upper()}:{int((g > 0).sum())}/{len(g)}" for l, g in per),
                          n_loc_median_pos=int((per.median() > 0).sum())))
        ph = holm([x["p_wilcoxon"] for x in r])
        for x, p in zip(r, ph):
            x["p_holm"] = float(p)
        rows += r
    return pd.DataFrame(rows)


def clustered_bootstrap(d, n_boot=None, seed=None):
    """d: paired differences indexed by (loc, seed). Statistic = mean over locations of the
    per-location median. Resample LOCATIONS with replacement. Returns (point, lo, hi)."""
    n_boot, seed = n_boot or I["bootstrap"]["n_boot"], seed or I["bootstrap"]["seed"]
    med = d.groupby(level=0).median().to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(med), (n_boot, len(med)))
    boot = med[idx].mean(1)
    return float(med.mean()), *np.percentile(boot, [2.5, 97.5]).tolist()
