#!/usr/bin/env python3
"""Post-freeze plot for Figure 1 (location-clustered robustness view).

No frozen figure shows the location-clustered bootstrap, so this script plots it from frozen
results only: results/bootstrap.csv (point = mean over locations of the per-location median
paired difference, 95% percentile interval from resampling locations) for every contrast that
survives Holm correction in results/wilcoxon.csv. Per-location medians are recomputed from
results/abl_perseed.csv and asserted to reproduce the frozen bootstrap point estimate.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from post_freeze import register, verify_frozen  # noqa: E402

LOCS = ["ankara", "bergen", "dubai", "lhasa", "yakutsk"]
METRIC = {"ks_stat": "KS", "kendall_mae": r"Kendall $\tau$ MAE", "acf_mae": "ACF MAE",
          "phys_viol_rate": "PhysViol"}
ARM = {"no_calibration": "no calibration", "no_D_sorted": r"remove $D_{\mathrm{sort}}$",
       "rank_D_sort": "rank (diagnostic)"}


def main(root: Path) -> None:
    names = ("bootstrap", "wilcoxon", "abl_perseed")
    verify_frozen(root, [root / "results" / f"{n}.csv" for n in names] + [root / "protocol.yaml"])
    rd = lambda n: pd.read_csv(root / "results" / f"{n}.csv", comment="#")  # noqa: E731
    boot, wil, abl = (rd(n) for n in names)
    alpha = yaml.safe_load((root / "protocol.yaml").read_text())["inference"]["alpha"]

    sig = wil[(wil.family_size > 2) & (wil.p_holm < alpha)][["metric", "contrast", "p_holm"]]
    sig = sig.merge(boot, on=["metric", "contrast"], validate="one_to_one")
    assert len(sig) and ((sig.lo > 0) | (sig.hi < 0)).all(), "a Holm-significant interval spans 0"

    full = abl[abl.config == "full_model"].set_index(["loc", "seed"])
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    n = len(sig)
    fig, axs = plt.subplots(n, 1, figsize=(4.6, 0.62 * n + 0.5), squeeze=False)
    for ax, r in zip(axs[:, 0], sig.sort_values(["contrast", "metric"]).itertuples()):
        arm = abl[abl.config == r.contrast].set_index(["loc", "seed"])[r.metric]
        loc_med = (arm - full.loc[arm.index, r.metric]).groupby(level="loc").median()
        assert abs(loc_med.mean() - r.point) <= 1e-9 * max(1.0, abs(r.point)), r
        ax.axvline(0, color="0.3", lw=0.8)
        ax.scatter(loc_med.reindex(LOCS), [0] * len(LOCS), s=12, color="0.65", zorder=2,
                   label="location median" if ax is axs[0, 0] else None)
        ax.errorbar(r.point, 0, xerr=[[r.point - r.lo], [r.hi - r.point]], fmt="o", ms=4,
                    color="#0072B2", capsize=3, lw=1.2, zorder=3,
                    label="mean of location medians, 95% interval" if ax is axs[0, 0] else None)
        ax.set_yticks([0], [f"{ARM[r.contrast]}: {METRIC[r.metric]}"])
        ax.set_ylim(-0.6, 0.6)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 3))
    axs[-1, 0].set_xlabel(r"paired difference (configuration $-$ full model)")
    fig.legend(frameon=False, fontsize=7, loc="upper center", ncol=2, bbox_to_anchor=(0.55, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out = root / "figures" / "fig1_location_robustness_pf"
    files = []
    for ext in ("pdf", "png"):
        f = out.with_suffix(f".{ext}")
        fig.savefig(f, dpi=300, metadata={"CreationDate": None} if ext == "pdf" else None)
        files.append(f)
    register(root, [root / "scripts" / "fig1_extra.py", *files])


if __name__ == "__main__":
    main(Path(sys.argv[sys.argv.index("--root") + 1] if "--root" in sys.argv else "."))
