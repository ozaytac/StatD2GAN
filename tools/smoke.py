#!/usr/bin/env python3
"""make smoke: does the training/evaluation pipeline run end to end? CPU or GPU, minutes.

Nothing here is a result. Everything is written to build/smoke/ (a scratch copy of src/, scripts/,
protocol.yaml, protocol.lock and one raw ERA5 file, with the output directories of the frozen code
created by tools/_common.ensure_dirs); runs/ and data/ of the release are untouched.
  1. frozen scripts/preprocess.py for LOC (no torch needed); the .npz is checked for its fields,
     window shape and held-out years
  2. frozen unit tests that need torch (tests/test_torch.py, incl. a 2-epoch run on synthetic
     windows and the exact-redraw check) and tests/test_distance_extra.py
  3. one frozen `run_one(full_model, LOC, SEED)` with 2 epochs instead of the protocol's, including
     calibration, projection and the full evaluation; the run record and the weight file are checked
Without torch, steps 2 and 3 are reported as skipped and the exit status is non-zero.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, copy, ensure_dirs, fresh, run  # noqa: E402

PY = sys.executable
KEYS = {"config", "loc", "seed", "raw", "cal", "proj", "seq", "train_seconds", "eval_seconds", "gpu"}


def check_npz(p: Path) -> str:
    import numpy as np
    import yaml
    P = yaml.safe_load((p.parents[2] / "protocol.yaml").read_text())
    D = P["data"]
    z = np.load(p)
    need = {"train_phys", "val_phys", "lo", "hi", "val_years"}
    assert need <= set(z.files), f"{p.name}: missing {need - set(z.files)}"
    nf = len(D["features"])
    for k in ("train_phys", "val_phys"):
        assert z[k].ndim == 3 and z[k].shape[1:] == (D["seq_len"], nf), (k, z[k].shape)
        assert np.isfinite(z[k]).all(), f"{k} not finite"
    ys = z["val_years"].tolist()
    assert ys == list(range(D["years"][1] - D["val_years"] + 1, D["years"][1] + 1)), ys
    return f"train {z['train_phys'].shape} val {z['val_phys'].shape} val_years {ys}"


def check_run(js: Path, epochs: int) -> str:
    rec = json.loads(js.read_text())
    missing = KEYS - set(rec)
    assert not missing, f"run record lacks {sorted(missing)}"
    assert len(rec.get("epoch_seconds", [])) == epochs, "epoch count"
    ks = rec["cal"]["ks_stat"]
    assert math.isfinite(ks) and 0 <= ks <= 1, ks
    pt = js.with_suffix(".pt")
    assert pt.exists() and pt.stat().st_size > 0, "weights not written"
    return f"calibrated KS {ks:.4f} (meaningless at {epochs} epochs), weights {pt.stat().st_size} B"


def main(build: Path, loc: str, seed: int, epochs: int) -> int:
    raw, lock = ROOT / f"data/raw/{loc}.csv", ROOT / "protocol.lock"
    missing = [str(p.relative_to(ROOT)) for p in (raw, lock) if not p.exists()]
    if missing:
        sys.exit(f"smoke: missing {missing} (see `make data`)")
    B = fresh(build)
    for rel in ["src", "scripts", "tests", "protocol.yaml", "protocol.lock", f"data/raw/{loc}.csv"]:
        copy(rel, B)
    ensure_dirs(B)
    log, t = B / "smoke.log", {}
    t["preprocess"] = run([PY, "scripts/preprocess.py", loc], B, log)
    print("npz check:", check_npz(B / f"data/processed/{loc}.npz"))
    try:
        import torch
    except ImportError:
        print("smoke: INCOMPLETE. preprocess passed; unit tests and the 2-epoch run skipped: torch not installed "
              "(`make env WITH_TORCH=1`)")
        return 2
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__} on {dev}")
    t["unit_tests"] = run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_torch.py",
                           "tests/test_distance_extra.py"], B, log)
    code = (f"import sys; sys.path.insert(0,'src'); from s2g.train import run_one; "
            f"run_one('full_model','{loc}',{seed},out_dir='runs_smoke',epochs={epochs})")
    t["train_eval"] = run([PY, "-c", code], B, log)
    print("run check:", check_run(B / "runs_smoke" / loc / f"{loc}_full_model_seed{seed}.json", epochs))
    print(f"smoke: PASSED on {dev}: {loc} full_model seed {seed}, {epochs} epochs; "
          f"timings {json.dumps({k: round(v) for k, v in t.items()})} s")
    (B / "report.json").write_text(json.dumps(dict(device=dev, torch=torch.__version__, seconds=t), indent=1))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=Path, default=ROOT / "build" / "smoke")
    ap.add_argument("--loc", default="ankara")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=2)
    a = ap.parse_args()
    sys.exit(main(a.build.resolve(), a.loc, a.seed, a.epochs))
