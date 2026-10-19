#!/usr/bin/env python3
"""make distance: recompute results_extra/distance_cal_train.csv from the stored generator weights.

Needs the weights (`make fetch-weights`), the raw ERA5 files (data/raw/*.csv) and torch. GPU
recommended; CPU works but the gate will most likely stop it (see below). Writes only to
build/distance/; the committed CSV is never touched.

Gate (scripts/distance_extra.py, unchanged): every run's recomputed raw and calibrated KS must equal
the frozen run record within 1e-4, and the triangle inequality must hold per feature. A single
failure stops the script and no CSV is written. On the original device class (NVIDIA RTX PRO 6000
Blackwell) the recomputation was exact (maximum difference 0 over 250 runs). On other devices the
noise draws can differ, so agreement is approximate at best and the gate may refuse; bitwise
agreement on other devices is not claimed.

After a successful run the new CSV is compared with the committed one: byte-identical, or identical
in every numeric column (the environment columns device/torch/cuda/cudnn record where it ran).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, ensure_dirs, load, run, sha  # noqa: E402

ENV_COLS = {"device", "torch", "cuda", "cudnn", "json_gpu"}


def main(build: Path, locs: list[str] | None, cpu: bool) -> int:
    pts = list(ROOT.glob("runs/*/*.pt"))
    if not pts:
        sys.exit("no runs/*/*.pt: run `make fetch-weights` first")
    try:
        import torch  # noqa: F401
    except ImportError:
        sys.exit("torch is not installed (see README: make env WITH_TORCH=1)")
    build.mkdir(parents=True, exist_ok=True)
    ensure_dirs(ROOT)
    base = [sys.executable, "scripts/distance_extra.py", "--runs-dir", "runs", "--data-dir", "data",
            "--manifest", "MANIFEST.json", "--floor-csv", "results/ks_floor.csv", "--out", build]
    run(base + (["--locs", *locs] if locs else []) + (["--cpu"] if cpu else []), ROOT, build / "distance.log")
    if locs:
        print("partial run (LOC given): no CSV is finalised; run `make distance` without LOC to finish")
        return 0
    run(base + ["--finalize"], ROOT, build / "distance.log")
    new, ref = build / "distance_cal_train.csv", ROOT / "results_extra" / "distance_cal_train.csv"
    reg = load("MANIFEST_post_freeze.json")["files"]["results_extra/distance_cal_train.csv"]
    if sha(new) == reg:
        print("distance: PASSED, CSV byte-identical to the registered one (sha256 %s...)" % reg[:16])
        return 0
    import pandas as pd
    a, b = (pd.read_csv(p, comment="#") for p in (ref, new))
    cols = [c for c in a.columns if c not in ENV_COLS]
    same = list(a.columns) == list(b.columns) and a[cols].equals(b[cols])
    print("distance:", "PASSED, numerically identical; only the environment columns differ" if same
          else "DIFFERS from the registered CSV: inspect build/distance (the gate passed, so every run is within 1e-4)")
    return 0 if same else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=Path, default=ROOT / "build" / "distance")
    ap.add_argument("--loc", nargs="*")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    sys.exit(main(a.build.resolve(), a.loc, a.cpu))
