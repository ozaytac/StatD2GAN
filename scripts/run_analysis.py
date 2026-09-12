#!/usr/bin/env python
"""Cross-location analysis: backfill diagnostics, build the summary/percent-
change/Wilcoxon tables, export LaTeX, and (optionally) the constraint
projection decision. Requires every location's ablation run to be complete
(see scripts/run_ablation.py) — this script does no training.

Usage:
    python scripts/run_analysis.py --data-root ./data --out-dir ./results
    python scripts/run_analysis.py --data-root ./data --out-dir ./results --projection
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from statd2gan.ablation import ABLATION_CONFIGS
from statd2gan.analysis import (ALL_LOCATIONS, backfill_diagnostics, lambda_insurance,
                                load_all, make_summary, pct_table, projection_analysis,
                                run_noise, wilcoxon_table, write_latex)
from statd2gan.config import C, DEVICE


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=C.base_root)
    ap.add_argument("--out-dir", default="./results")
    ap.add_argument("--locations", nargs="+", default=ALL_LOCATIONS)
    ap.add_argument("--projection", action="store_true",
                    help="Also run the (slower, ~20 min) constraint-projection comparison.")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cfg_by_name = {c.name: c for c in ABLATION_CONFIGS}
    contrasts = [n for n in cfg_by_name if n != "full_model"]

    backfill_diagnostics(DEVICE, cfg_by_name, args.data_root, args.locations)
    df, floors = load_all(args.data_root, args.locations, args.out_dir)
    summary = make_summary(df, floors, args.out_dir)
    pct = pct_table(df, contrasts, args.out_dir)
    wtab = wilcoxon_table(df, contrasts, args.out_dir)
    write_latex(summary, wtab, floors, contrasts, args.out_dir)

    pd.set_option("display.width", 200)
    print("\n=== KS / NOISE FLOOR RATIO (1.00 = at floor) ===")
    print(summary[summary.config == "full_model"][["loc", "ks_stat_mean", "ks_floor", "ks_over_floor"]]
          .round(3).to_string(index=False))

    print("\n=== CALIBRATION VIOLATION INJECTION (full_model, phys - phys_raw) ===")
    fm = summary[summary.config == "full_model"]
    print(fm.assign(injected=(fm.phys_viol_rate_mean - fm.phys_viol_raw_mean))[
          ["loc", "phys_viol_raw_mean", "phys_viol_rate_mean", "injected", "tie_frac_mean_mean"]]
          .round(4).to_string(index=False))

    print("\n=== PERCENT CHANGE (from raw per-seed data; tau) ===")
    print(pct.pivot(index="config", columns="loc", values="kendall_mae").to_string())

    print("\n=== WILCOXON (25 pairs) ===")
    print(wtab.to_string(index=False))

    print("\n=== WITHIN-RUN NOISE (tau, full vs no_calibration) ===")
    print(run_noise(df).to_string())

    print("\n=== LAMBDA INSURANCE (retracted claim, reported for completeness) ===")
    print(lambda_insurance(df).to_string())

    print(f"\nFiles written to: {args.out_dir}/ "
          "(abl_perseed_v2, all_locations_summary_v2, pct_change_v2, wilcoxon_v2, tables_generated_v2.tex)")

    if args.projection:
        projection_analysis(DEVICE, cfg_by_name, args.data_root, args.out_dir)


if __name__ == "__main__":
    main()
