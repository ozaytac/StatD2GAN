"""figures/fig{1..4}.{pdf,png} from results/*.csv only."""
import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from s2g.config import ROOT, LOCS, read_csv, assert_frozen

assert_frozen()
R, F = ROOT / "results", ROOT / "figures"; F.mkdir(exist_ok=True)
NAMES = [l.capitalize() for l in LOCS]; x = np.arange(len(LOCS))
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
save = lambda fig, n: [fig.savefig(F / f"{n}.{e}", dpi=200, bbox_inches="tight") for e in ("pdf", "png")]

# Fig 1: KS / real train-held-out reference
kr = read_csv(R / "ks_ratio.csv"); m = kr.groupby(["loc", "config"]).ks_ratio.mean().unstack()
fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.bar(x, [m.loc[l, "no_calibration"] for l in LOCS], 0.5, color="#dd5555", label="No calibration")
for i, l in enumerate(LOCS):
    ax.text(i, m.loc[l, "no_calibration"] + .05, f"{m.loc[l, 'no_calibration']:.1f}x", ha="center")
cal = [c for c in m.columns if c != "no_calibration"]
for c in cal:
    ax.plot(x, [m.loc[l, c] for l in LOCS], "o", color="#1f77b4", ms=5, alpha=.6, label=f"Calibrated configs ({len(cal)})" if c == cal[0] else None)
ax.axhline(1, color="k", ls="--", lw=1, label="real train vs held-out reference")
ax.set_xticks(x, NAMES); ax.set_ylabel("KS sup-distance / reference"); ax.legend(frameon=False, loc="upper right")
save(fig, "fig1_ks_floor")

# Fig 2: D_sort effect (a) per location, (b) paired differences with clustered-bootstrap CI
pc, ab, bs, rn = (read_csv(R / f) for f in ("pct_change.csv", "abl_perseed.csv", "bootstrap.csv", "run_noise.csv"))
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.8))
a.bar(x, [pc[(pc["loc"] == l) & (pc.config == "no_D_sorted")].kendall_mae.iloc[0] for l in LOCS], .55, color="#4a90c4")
w = ab.pivot_table(index=["loc", "seed"], columns="config", values="kendall_mae")
per = 100 * (w["no_D_sorted"] - w["full_model"]) / w["full_model"]
for i, l in enumerate(LOCS):
    a.plot([i] * len(per[l]), per[l], "o", color="k", alpha=.5, ms=3)
band = float(rn.band_pct.iloc[0]); a.axhspan(-band, band, color="0.9", zorder=0); a.axhline(0, color="k", lw=.8)
a.text(.98, .03, f"run-to-run band (+/-{band:.0f}%)", transform=a.transAxes, ha="right", va="bottom", color="0.4", fontsize=8)
a.set_xticks(x, NAMES); a.set_ylabel(r"$\tau$-MAE change on removing $D_{sort}$ (%)")
labels = [("kendall_mae", r"$\tau$-MAE"), ("acf_mae", "ACF MAE"), ("phys_viol_rate", "PhysViol")]
for j, (c, col, lab) in enumerate([("no_D_sorted", "#4a90c4", "remove $D_{sort}$"), ("rank_D_sort", "#ff9933", "rank (copula) variant")]):
    r = [bs[(bs.metric == k) & (bs.contrast == c)].iloc[0] for k, _ in labels]
    xx = np.arange(3) + (j - .5) * .38
    b.bar(xx, [q.point for q in r], .36, color=col, label=lab)
    b.errorbar(xx, [q.point for q in r], [[q.point - q.lo for q in r], [q.hi - q.point for q in r]], fmt="none", ecolor="k", capsize=3)
b.axhline(0, color="k", lw=.8); b.set_xticks(range(3), [l for _, l in labels]); b.legend(frameon=False)
b.set_ylabel("paired difference vs full model\n(mean of location medians, 95% clustered CI)")
a.set_title("(a) Regime-dependent effect"); b.set_title("(b) Quantile channel vs copula variant")
save(fig, "fig2_dsort_mechanism")

# Fig 3: violations injected by calibration, removed by projection
pr = read_csv(R / "projection.csv").groupby("loc")[["raw_phys_viol_rate", "cal_phys_viol_rate", "proj_phys_viol_rate"]].mean().loc[LOCS]
fig, ax = plt.subplots(figsize=(6.4, 3.8))
for j, (c, col, lab) in enumerate([("raw_phys_viol_rate", "#4caf50", "generator output (raw)"),
                                   ("cal_phys_viol_rate", "#d94848", "after calibration"),
                                   ("proj_phys_viol_rate", "#4a90c4", "after projection")]):
    ax.bar(x + (j - 1) * .27, pr[c], .26, color=col, label=lab)
ax.set_xticks(x, NAMES); ax.set_ylabel("physics violation rate"); ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(.5, 1.0))
save(fig, "fig3_injection_projection")

# Fig 4: variance decomposition
sq = read_csv(R / "seq_level.csv"); fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
for ax, met, t in zip(axs, ("seq_mean_sd", "within_sd"), ("(a) Seasonal diversity: SD of per-sequence means", "(b) Within-sequence SD")):
    for j, (mod, col, lab) in enumerate([("statd2gan", "#4a90c4", "StatD2GAN"), ("timegan", "#ff9933", "TimeGAN"), ("rcgan", "#4caf50", "RCGAN")]):
        v = sq[sq.model == mod].groupby("loc")[met].median().loc[LOCS]
        ax.bar(x + (j - 1) * .27, v, .26, color=col, label=lab)
    rv = sq[sq.model == "real_val"].set_index("loc")[met].loc[LOCS]
    ax.hlines(rv, x - .42, x + .42, color="k", lw=2.5, label="real (held-out)")
    ax.set_xticks(x, NAMES); ax.set_ylabel("temperature SD (C)"); ax.set_title(t)
axs[0].legend(frameon=False, fontsize=8)
save(fig, "fig4_variance_decomposition")
