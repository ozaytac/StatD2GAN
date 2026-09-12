#!/usr/bin/env python
"""Run preprocessing -> ablation -> sensitivity -> tables & plots for one or
more locations. Resumable: interrupting and re-running skips completed work.

Usage:
    python scripts/run_ablation.py --locations ankara dubai --data-root ./data
    python scripts/run_ablation.py --locations bergen --fast   # smoke test
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import shutil
import time

from statd2gan.ablation import (ABLATION_CONFIGS, calibration_ablation,
                                run_ablation, run_sensitivity)
from statd2gan.config import C, DEVICE, LOCATIONS, make_paths
from statd2gan.data import load_location
from statd2gan.plotting import plot_ablation, plot_sensitivity
from statd2gan.preprocessing import preprocess_location

EXPORT_FILES = ["ablation_raw.csv", "ablation_table.csv", "bootstrap_ci.csv",
                "sensitivity_multiseed.csv", "calibration_ablation.csv",
                "ablation_chart.png", "sensitivity_chart.png"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--locations", nargs="+", default=LOCATIONS,
                    help=f"Locations to process (default: all {LOCATIONS}).")
    ap.add_argument("--data-root", default=C.base_root,
                    help="Directory containing <location>.csv and where outputs are written.")
    ap.add_argument("--fast", action="store_true",
                    help="Smoke-test mode: short training schedule, for pipeline validation only.")
    ap.add_argument("--skip-sensitivity", action="store_true")
    args = ap.parse_args()

    # Fast mode overrides each ablation config's epoch counts individually
    # (dataclasses.replace) rather than mutating the frozen global C, since
    # ExperimentConfig defaults are already bound at import time and would
    # not see a later change to C.
    configs = ABLATION_CONFIGS
    if args.fast:
        configs = [dataclasses.replace(c, warmup_epochs=5, total_epochs=30) for c in ABLATION_CONFIGS]
        print("[fast mode] warmup_epochs=5, total_epochs=30 -- results are NOT paper-comparable.")

    for loc in args.locations:
        t0 = time.time()
        print("\n" + "#" * 70)
        print(f"#  LOCATION: {loc.upper()}")
        print("#" * 70)

        paths = make_paths(loc, args.data_root)
        if not preprocess_location(loc, paths, args.data_root, C):
            continue
        loc_data = load_location(paths, C)

        df_abl = run_ablation(DEVICE, paths, loc_data.train_tensor, loc_data.real_train_phys,
                              loc_data.real_val_phys, loc_data.scaler, configs=configs)
        plot_ablation(paths, df_abl, n_seeds=len(C.seeds))

        if not args.skip_sensitivity:
            df_sens = run_sensitivity(DEVICE, paths, loc_data.train_tensor, loc_data.real_train_phys,
                                      loc_data.real_val_phys, loc_data.scaler)
            plot_sensitivity(paths, df_sens)

        calibration_ablation(paths, df_abl)

        for fn in EXPORT_FILES:
            src = f"{paths.out_root}/{fn}"
            if os.path.exists(src):
                shutil.copy(src, f"{args.data_root}/{loc}_{fn}")

        print(f"\n[{loc}] COMPLETE in {(time.time() - t0) / 3600:.2f} h")

    print("\nALL LOCATIONS DONE.")


if __name__ == "__main__":
    main()
