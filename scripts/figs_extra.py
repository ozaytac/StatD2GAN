#!/usr/bin/env python3
"""Post-freeze redraw of frozen Figures 2 and 4 with corrected labels.

The frozen figures (figures/fig2_dsort_mechanism.*, figures/fig4_variance_decomposition.*) carry
two panel titles that the paper no longer supports: "(a) Regime-dependent effect" (a retracted
claim) and "(a) Seasonal diversity: ..." (the statistic is between-sequence spread, not calendar
seasonality). Frozen files are never edited. This script redraws both figures from frozen
results/*.csv only, with plotting code copied verbatim from the frozen scripts/figures.py
(sha-verified against MANIFEST.json). Every departure from the frozen code is a parameter in
FROZEN / PF below and nothing else changes:
  fig2 (a) title   "(a) Regime-dependent effect" -> "(a) Per-location effect"
  fig2 (b) title   "(b) Quantile channel vs copula variant" -> "(b) Removal vs rank-transform variant"
  fig2 (b) legend  "rank (copula) variant" -> "rank-transform variant"   (D4 is "consistent with" only)
  fig2 wspace      matplotlib default -> 0.35 (the (b) y-label overlapped the (a) band label)
  fig4 (a) title   "(a) Seasonal diversity: SD of per-sequence means" -> "(a) Between-sequence spread: SD of per-sequence means"
  fig4 y-labels    "temperature SD (C)" -> "temperature SD (K)" (the paper text uses K; SDs are equal in C and K)
No new inference and no new numbers. `--check` first redraws with the FROZEN parameters and compares
the result with the frozen PNGs pixel by pixel; this needs the frozen plotting environment.

Outputs: figures/fig2_dsort_mechanism_pf.{pdf,png}, figures/fig4_variance_decomposition_pf.{pdf,png},
and figures/figs_extra_record.json (sha256 of script, inputs and outputs). post_freeze.py, which
registers outputs in MANIFEST_post_freeze.json, was not available; the record file carries the same
information until it is merged. MANIFEST.json is never modified.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["sans-serif"]   # stock matplotlib default (DejaVu Sans); a site matplotlibrc may override it
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

FROZEN = {"fig2_a": "(a) Regime-dependent effect", "fig2_b": "(b) Quantile channel vs copula variant",
          "fig2_rank": "rank (copula) variant", "fig2_wspace": None,
          "fig4_a": "(a) Seasonal diversity: SD of per-sequence means", "fig4_unit": "C"}
PF = {"fig2_a": "(a) Per-location effect", "fig2_b": "(b) Removal vs rank-transform variant",
      "fig2_rank": "rank-transform variant", "fig2_wspace": 0.35,
      "fig4_a": "(a) Between-sequence spread: SD of per-sequence means", "fig4_unit": "K"}
INPUTS = ["results/pct_change.csv", "results/abl_perseed.csv", "results/bootstrap.csv", "results/run_noise.csv",
          "results/seq_level.csv", "protocol.yaml", "scripts/figures.py"]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731


def verify(root: Path) -> dict:
    man = json.loads((root / "MANIFEST.json").read_text())
    assert man["protocol_sha256"] == sha(root / "protocol.yaml"), "protocol.yaml is not the frozen one"
    out = {}
    for rel in INPUTS:
        h = sha(root / rel)
        assert man["files"][rel] == h, f"{rel}: sha differs from MANIFEST.json"
        out[rel] = h
    return out


def draw(root: Path, titles: dict, suffix: str) -> list[Path]:
    # --- verbatim from frozen scripts/figures.py (setup, Fig 2, Fig 4); departures are the FROZEN/PF parameters
    read_csv = lambda p: pd.read_csv(p, comment="#")  # noqa: E731  (= s2g.config.read_csv)
    LOCS = list(yaml.safe_load((root / "protocol.yaml").read_text())["data"]["locations"])
    R, F = root / "results", root / "figures"; F.mkdir(exist_ok=True)
    NAMES = [l.capitalize() for l in LOCS]; x = np.arange(len(LOCS))
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    files: list[Path] = []
    save = lambda fig, n: [(fig.savefig(F / f"{n}{suffix}.{e}", dpi=200, bbox_inches="tight"),  # noqa: E731
                            files.append(F / f"{n}{suffix}.{e}")) for e in ("pdf", "png")]

    # Fig 2: D_sort effect (a) per location, (b) paired differences with clustered-bootstrap CI
    pc, ab, bs, rn = (read_csv(R / f) for f in ("pct_change.csv", "abl_perseed.csv", "bootstrap.csv", "run_noise.csv"))
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw=None if titles["fig2_wspace"] is None else {"wspace": titles["fig2_wspace"]})
    a.bar(x, [pc[(pc["loc"] == l) & (pc.config == "no_D_sorted")].kendall_mae.iloc[0] for l in LOCS], .55, color="#4a90c4")
    w = ab.pivot_table(index=["loc", "seed"], columns="config", values="kendall_mae")
    per = 100 * (w["no_D_sorted"] - w["full_model"]) / w["full_model"]
    for i, l in enumerate(LOCS):
        a.plot([i] * len(per[l]), per[l], "o", color="k", alpha=.5, ms=3)
    band = float(rn.band_pct.iloc[0]); a.axhspan(-band, band, color="0.9", zorder=0); a.axhline(0, color="k", lw=.8)
    a.text(.98, .03, f"run-to-run band (+/-{band:.0f}%)", transform=a.transAxes, ha="right", va="bottom", color="0.4", fontsize=8)
    a.set_xticks(x, NAMES); a.set_ylabel(r"$\tau$-MAE change on removing $D_{sort}$ (%)")
    labels = [("kendall_mae", r"$\tau$-MAE"), ("acf_mae", "ACF MAE"), ("phys_viol_rate", "PhysViol")]
    for j, (c, col, lab) in enumerate([("no_D_sorted", "#4a90c4", "remove $D_{sort}$"), ("rank_D_sort", "#ff9933", titles["fig2_rank"])]):
        r = [bs[(bs.metric == k) & (bs.contrast == c)].iloc[0] for k, _ in labels]
        xx = np.arange(3) + (j - .5) * .38
        b.bar(xx, [q.point for q in r], .36, color=col, label=lab)
        b.errorbar(xx, [q.point for q in r], [[q.point - q.lo for q in r], [q.hi - q.point for q in r]], fmt="none", ecolor="k", capsize=3)
    b.axhline(0, color="k", lw=.8); b.set_xticks(range(3), [l for _, l in labels]); b.legend(frameon=False)
    b.set_ylabel("paired difference vs full model\n(mean of location medians, 95% clustered CI)")
    a.set_title(titles["fig2_a"]); b.set_title(titles["fig2_b"])
    save(fig, "fig2_dsort_mechanism")

    # Fig 4: variance decomposition
    sq = read_csv(R / "seq_level.csv"); fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, met, t in zip(axs, ("seq_mean_sd", "within_sd"), (titles["fig4_a"], "(b) Within-sequence SD")):
        for j, (mod, col, lab) in enumerate([("statd2gan", "#4a90c4", "StatD2GAN"), ("timegan", "#ff9933", "TimeGAN"), ("rcgan", "#4caf50", "RCGAN")]):
            v = sq[sq.model == mod].groupby("loc")[met].median().loc[LOCS]
            ax.bar(x + (j - 1) * .27, v, .26, color=col, label=lab)
        rv = sq[sq.model == "real_val"].set_index("loc")[met].loc[LOCS]
        ax.hlines(rv, x - .42, x + .42, color="k", lw=2.5, label="real (held-out)")
        ax.set_xticks(x, NAMES); ax.set_ylabel(f"temperature SD ({titles['fig4_unit']})"); ax.set_title(t)
    axs[0].legend(frameon=False, fontsize=8)
    save(fig, "fig4_variance_decomposition")
    plt.close("all")
    return files


def check(root: Path, frozen_dir: Path) -> dict:
    """Redraw with the frozen titles and compare with the frozen PNGs (same size, pixel difference)."""
    files = draw(root, FROZEN, "_check")
    res = {}
    for f in (p for p in files if p.suffix == ".png"):
        a = plt.imread(f); b = plt.imread(frozen_dir / f.name.replace("_check", ""))
        same = a.shape == b.shape
        res[f.name] = {"same_shape": same, "max_abs_diff": float(np.abs(a - b).max()) if same else None,
                       "frac_pixels_diff": float((np.abs(a - b).max(-1) > 1e-6).mean()) if same else None}
        f.unlink(); f.with_suffix(".pdf").unlink()
    return res


def main(root: Path, frozen_dir: Path | None) -> None:
    ins = verify(root)
    print(f"verified {len(ins)} frozen inputs against MANIFEST.json")
    rec = {"script": "scripts/figs_extra.py", "script_sha256": sha(root / "scripts" / "figs_extra.py"),
           "inputs_sha256": ins, "matplotlib": matplotlib.__version__, "changes": {k: [FROZEN[k], PF[k]] for k in PF}}
    if frozen_dir:
        rec["check_vs_frozen_png"] = check(root, frozen_dir)
        print(json.dumps(rec["check_vs_frozen_png"], indent=1))
    out = draw(root, PF, "_pf")
    rec["outputs_sha256"] = {str(p.relative_to(root)): sha(p) for p in out}
    (root / "figures" / "figs_extra_record.json").write_text(json.dumps(rec, indent=1))
    print("wrote", *[str(p.relative_to(root)) for p in out], "figures/figs_extra_record.json")


if __name__ == "__main__":
    a = sys.argv
    main(Path(a[a.index("--root") + 1]) if "--root" in a else Path("."),
         Path(a[a.index("--check") + 1]) if "--check" in a else None)
