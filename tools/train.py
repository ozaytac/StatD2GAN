#!/usr/bin/env python3
"""make train: re-run the training campaign with the frozen code. GPU-HOURS: the full campaign is
275 runs (9 arms x 5 locations x 5 seeds + 2 baselines x 25); the frozen run records sum to 34 hours
of training (MANIFEST.json gpu_hours) on an NVIDIA RTX PRO 6000 Blackwell, evaluation not included. Not part of `make all` or `make all-heavy`.

The release already contains all 275 run records, and the frozen campaign script never overwrites a
run (written once), so training happens in a scratch copy, build/train/, holding src/, scripts/,
protocol.yaml, protocol.lock, data/raw/, runs/budget.json (the frozen scope decision; active arms) and,
if present, runs/pilot_gate.json. New run records land in build/train/runs/ and can be compared with the frozen records in
runs/. GPU nondeterminism means a retrained campaign is a replication, not a byte-identical rebuild.

Without CONFIRM=yes it only prepares build/train (copy, raw-data check, preprocessing) and stops.
Usage: make train LOC=ankara [SEED=42] CONFIRM=yes   (one location, one stage at a time is advisable)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, copy, ensure_dirs, run  # noqa: E402

PY = sys.executable


def main(build: Path, loc: str | None, seed: int | None, stage: str, confirm: str) -> int:
    for need in ["protocol.lock", "runs/budget.json"]:
        if not (ROOT / need).exists():
            sys.exit(f"missing {need}: train needs the complete frozen release")
    raws = sorted(ROOT.glob("data/raw/*.csv"))
    if not raws:
        sys.exit("no data/raw/*.csv: see `make data`")
    if not build.exists():
        for rel in ["src", "scripts", "protocol.yaml", "protocol.lock", "data/raw", "data/raw.sha256",
                    "runs/budget.json", "runs/pilot_gate.json", "Makefile"]:
            if (ROOT / rel).exists():
                copy(rel, build)
    ensure_dirs(build)
    run([PY, "scripts/verify_raw.py", *([loc] if loc else [])], build)
    run([PY, "scripts/preprocess.py", *([loc] if loc else [])], build)
    if confirm != "yes":                     # preparation (CPU) done; the GPU campaign needs consent
        print(f"train: {build} prepared (raw data verified, preprocessed). The next step runs the GPU campaign "
              "(the frozen 275 runs took 34 GPU-hours of training, MANIFEST.json). Re-run with CONFIRM=yes, preferably with LOC=<location>.")
        return 2
    stages = ["full", "baselines"] if stage == "all" else [stage]
    for s in stages:
        cmd = [PY, "scripts/run_campaign.py", "--stage", s] + (["--loc", loc] if loc else []) + (["--seed", str(seed)] if seed else [])
        run(cmd, build, build / f"train_{s}.log")
    print(f"train: run records in {build / 'runs'}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=Path, default=ROOT / "build" / "train")
    ap.add_argument("--loc")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--stage", choices=["full", "baselines", "all"], default="all")
    ap.add_argument("--confirm", default="no")
    a = ap.parse_args()
    sys.exit(main(a.build.resolve(), a.loc, a.seed, a.stage, a.confirm))
