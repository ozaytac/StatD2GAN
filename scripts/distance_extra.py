#!/usr/bin/env python3
"""Post-freeze: distance of the calibrated sample to the TRAIN block, d(cal, train), per run.

For every calibrated run (8 calibrated arms x 25 + RCGAN 25 + TimeGAN 25 = 250; no_calibration
is out of scope) this script reloads the saved generator weights (.pt), redraws the evaluation
and calibration samples with the frozen seeds, refits the calibration map, and computes with the
frozen KS function (s2g.metrics.ks_per_feature: scipy ks_2samp sup-distance on the pooled ECDFs,
the seven scaled features, the train/held-out windows the floor uses):

    ks_raw_val_f   = KS_f(raw eval draw, held-out)     (gate against JSON raw.ks_stat)
    ks_cal_val_f   = KS_f(calibrated draw, held-out)   (gate against JSON cal.ks_stat)
    ks_floor_f     = KS_f(train, held-out)             (gate against frozen results/ks_floor.csv)
    d_cal_train_f  = KS_f(calibrated draw, train)
    d_raw_train_f  = KS_f(raw eval draw, train)        (descriptive)

No logic is new: draw = s2g.train.draw, scaler = s2g.data, calibration = s2g.metrics.fit_calibration
(refit on the independent calibration draw, calib_draw_seed + seed; the .pt holds generator weights
only, no calibration map), KS = s2g.metrics.ks_per_feature. No training. The frozen MANIFEST.json,
numbers.tex and tables.tex are only read.

Gates (any failure stops the run; no output table is written):
  G1  |mean_f ks_raw_val_f - JSON raw.ks_stat| <= 1e-4 and the same for cal.ks_stat.
  G2  triangle inequality |ks_cal_val_f - ks_floor_f| <= d_cal_train_f + 1e-12 for every feature,
      and for the feature means. It holds for any two samples, so a violation means a bug.
  G0  (with --manifest/--floor-csv) protocol.yaml, src/s2g/*.py, every .pt/.json sha256 == MANIFEST;
      per-feature floor == frozen ks_floor.csv within 1e-12.

Everything this script writes goes to ONE dedicated output folder (--out, required, created if
missing); the repo, the runs and the data folders are only read:
  <out>/partial/<loc>.jsonl          one line per finished run (resume point)
  <out>/gate_failures.jsonl          only if a gate fails
  <out>/run_log.txt                  copy of everything printed
  <out>/distance_cal_train.csv       --finalize, written once
  <out>/MANIFEST_post_freeze.json    --finalize: the repo's MANIFEST_post_freeze.json (read only)
                                     plus sha256 of this script and of the CSV

Usage (repo root = statd2gan_repro with the FROZEN protocol.yaml):
  python scripts/distance_extra.py --runs-dir R --data-dir D --out OUT \
      --manifest MANIFEST.json --floor-csv results/ks_floor.csv [--locs ankara] [--probe] [--cpu]
  python scripts/distance_extra.py --runs-dir R --data-dir D --out OUT --finalize
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PROTOCOL_SHA = "2d9a3b9d210df5b2b6104f6563eff583dd5642130ac863ff882b5f380b5aabd7"
GATE_TOL = 1e-4          # G1: recomputed vs recorded KS (absolute)
TRI_TOL = 1e-12          # G2: triangle inequality slack (float rounding only)
OUT_CSV = "distance_cal_train.csv"


class GateError(RuntimeError):
    pass


LOG = None                                           # set to <out>/run_log.txt by main()


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    if LOG:
        with open(LOG, "a") as f:
            f.write(s + "\n")


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---------------------------------------------------------------- torch-free core
def distances(train, val, ev, ca, floor_f: dict | None = None) -> dict:
    """All per-feature quantities for one run from the two physical-unit draws.
    train/val: real windows (N, T, F); ev: evaluation draw; ca: independent calibration draw."""
    from s2g import metrics as M
    phi, tie = M.fit_calibration(train, ca)
    cal = phi(ev)
    floor_f = floor_f or M.ks_floor(train, val)
    q = dict(ks_raw_val=M.ks_per_feature(val, ev), ks_cal_val=M.ks_per_feature(val, cal),
             ks_floor=dict(floor_f), d_cal_train=M.ks_per_feature(train, cal),
             d_raw_train=M.ks_per_feature(train, ev))
    q["tie_frac"] = tie
    return q


def mean(d: dict) -> float:
    return float(np.mean(list(d.values())))


def triangle(q: dict) -> dict:
    """G2 slack: d_f - |ks_cal_f - floor_f| per feature and for the means (must be >= -TRI_TOL)."""
    sl = {f: q["d_cal_train"][f] - abs(q["ks_cal_val"][f] - q["ks_floor"][f]) for f in q["d_cal_train"]}
    sm = mean(q["d_cal_train"]) - abs(mean(q["ks_cal_val"]) - mean(q["ks_floor"]))
    return dict(min_slack_feature=float(min(sl.values())), slack_mean=float(sm),
                ok=bool(min(sl.values()) >= -TRI_TOL and sm >= -TRI_TOL))


def gate(q: dict, rec: dict) -> dict:
    """G1 diffs against the frozen run JSON."""
    g = dict(gate_raw_diff=abs(mean(q["ks_raw_val"]) - rec["raw"]["ks_stat"]),
             gate_cal_diff=abs(mean(q["ks_cal_val"]) - rec["cal"]["ks_stat"]),
             tie_frac_diff=abs(q["tie_frac"] - rec["tie_frac_mean"]) if "tie_frac_mean" in rec else float("nan"))
    g["ok"] = bool(g["gate_raw_diff"] <= GATE_TOL and g["gate_cal_diff"] <= GATE_TOL)
    return g


def row(rec: dict, q: dict, g: dict, t: dict, env: dict) -> dict:
    fl = mean(q["ks_floor"]); dc = mean(q["d_cal_train"]); kc = mean(q["ks_cal_val"])
    r = dict(loc=rec["loc"], config=rec["config"], seed=rec["seed"],
             d_cal_train=dc, d_raw_train=mean(q["d_raw_train"]), ks_cal_val=kc,
             ks_raw_val=mean(q["ks_raw_val"]), ks_floor=fl, ratio_ks_cal_floor=kc / fl,
             ratio_d_cal_floor=dc / fl, gap_over_floor=abs(kc - fl) / fl,
             below_floor=bool(kc < fl), slack_mean=t["slack_mean"], min_slack_feature=t["min_slack_feature"],
             gate_raw_diff=g["gate_raw_diff"], gate_cal_diff=g["gate_cal_diff"], tie_frac_diff=g["tie_frac_diff"],
             gate_ok=g["ok"], triangle_ok=t["ok"], **env, json_gpu=rec.get("gpu", ""))
    for k in ("d_cal_train", "d_raw_train", "ks_cal_val", "ks_floor"):
        r.update({f"{k}_f{f}": v for f, v in q[k].items()})
    return r


# ---------------------------------------------------------------- torch layer (lazy)
def environment() -> dict:
    import torch
    from s2g.train import DEV
    return dict(device=torch.cuda.get_device_name(0) if DEV.type == "cuda" else f"cpu ({platform.processor() or platform.machine()})",
                torch=torch.__version__, cuda=str(torch.version.cuda), cudnn=str(torch.backends.cudnn.version()))


def build_generator(config: str):
    from s2g.config import P
    if config in ("rcgan", "timegan"):
        from s2g import baselines as B
        return (B.RCGen if config == "rcgan" else B.TimeGAN)(P["baselines"][config])
    if config not in P["arms"] or config == "no_calibration":
        raise ValueError(f"not a calibrated config: {config}")
    from s2g.models import Generator
    return Generator()


def draws(pt: Path, rec: dict, scaler, n_eval: int | None = None):
    """Evaluation and calibration draws exactly as train.finish makes them (physical units)."""
    import torch
    from s2g import train as T
    from s2g.config import P
    E = P["evaluation"]; n = n_eval or E["n_eval"]; s = rec["seed"]
    G = build_generator(rec["config"]).to(T.DEV)
    G.load_state_dict(torch.load(pt, map_location=T.DEV, weights_only=True))
    T.set_seed(s)                                   # same backend flags (tf32, cudnn) as after training
    ev = scaler.inv(T.draw(G, n, E["eval_draw_seed"] + s))
    ca = scaler.inv(T.draw(G, n, E["calib_draw_seed"] + s))
    return ev, ca


# ---------------------------------------------------------------- data, discovery, integrity
def load_data(data_dir: Path, loc: str):
    from s2g import data as D
    for p in (data_dir / "processed" / f"{loc}.npz", data_dir / f"{loc}.npz"):
        if p.exists():
            return D.load(p), str(p)
    for p in (data_dir / "raw" / f"{loc}.csv", data_dir / f"{loc}.csv"):
        if p.exists():                               # same path as scripts/preprocess.py
            import pandas as pd
            return D.build(pd.read_csv(p, parse_dates=["valid_time"])), str(p)
    raise SystemExit(f"no processed/{loc}.npz or raw/{loc}.csv under {data_dir}")


def find_runs(runs_dir: Path, locs, configs) -> list[tuple[Path, Path, dict]]:
    out = []
    for js in sorted(runs_dir.rglob("*_seed*.json")):
        try:
            rec = json.loads(js.read_text())
        except Exception:
            continue
        if not isinstance(rec, dict) or rec.get("loc") not in locs or rec.get("config") not in configs:
            continue
        pt = js.with_suffix(".pt")
        if not pt.exists():
            raise SystemExit(f"weights missing for {js}")
        out.append((js, pt, rec))
    return out


def check_manifest(manifest: Path, runs) -> int:
    files = json.loads(manifest.read_text())["files"]
    bad = []
    if sha256(ROOT / "protocol.yaml") != PROTOCOL_SHA:
        bad.append("protocol.yaml (use the FROZEN protocol.yaml, sha256 2d9a3b9d...)")
    for k, v in files.items():
        if k.startswith("src/s2g/") and (ROOT / k).exists() and sha256(ROOT / k) != v:
            bad.append(k)
    n = 0
    for js, pt, rec in runs:
        for p in (js, pt):
            key = f"runs/{rec['loc']}/{p.name}"
            if key not in files:
                bad.append(f"{key} (not in MANIFEST)")
            elif sha256(p) != files[key]:
                bad.append(key)
            n += 1
    if bad:
        raise SystemExit("G0 integrity failed: " + "; ".join(bad[:20]))
    return n


def frozen_floor(floor_csv: Path, loc: str) -> dict:
    import pandas as pd
    f = pd.read_csv(floor_csv, comment="#")
    f = f[f["loc"] == loc]
    return {int(r.feature): float(r.ks_floor) for r in f.itertuples()}


# ---------------------------------------------------------------- driver
def calibrated_configs() -> list[str]:
    from s2g.config import P
    return [a for a in P["arms"] if a != "no_calibration"] + list(P["baselines"]["models"])


def run(a):
    from s2g.config import LOCS
    locs = a.locs or LOCS
    configs = a.configs or calibrated_configs()
    for k, d in (("--runs-dir", a.runs_dir), ("--data-dir", a.data_dir)):
        if not d.is_dir():
            raise SystemExit(f"{k} {d} is not a directory")
    runs = find_runs(a.runs_dir, locs, configs)
    if not runs:
        raise SystemExit(f"no run JSON with a matching .pt found under {a.runs_dir}")
    if a.probe:                                      # one run per model family, first location
        fam = {}
        for x in runs:
            k = x[2]["config"] if x[2]["config"] in ("rcgan", "timegan") else "arm"
            if x[2]["loc"] == locs[0] and k not in fam:
                fam[k] = x
        runs = list(fam.values())
    log(f"{len(runs)} run(s) selected ({len(locs)} location(s))")
    if a.manifest:
        log(f"G0: {check_manifest(a.manifest, runs)} .pt/.json sha256 == MANIFEST; protocol and src/s2g OK")
    env = environment()
    log("environment:", env)
    (a.out / "partial").mkdir(parents=True, exist_ok=True)
    fails = a.out / "gate_failures.jsonl"
    worst = dict(raw=0.0, cal=0.0, slack=np.inf)
    for loc in locs:
        sel = [x for x in runs if x[2]["loc"] == loc]
        if not sel:
            continue
        part = a.out / "partial" / f"{loc}.jsonl"
        done = {(r["config"], r["seed"]) for r in map(json.loads, part.read_text().splitlines())} if part.exists() else set()
        todo = [x for x in sel if (x[2]["config"], x[2]["seed"]) not in done]
        log(f"[{loc}] {len(sel) - len(todo)} done, {len(todo)} to go")
        if not todo:
            continue
        d, src = load_data(a.data_dir, loc)
        if a.manifest and src.endswith(".csv"):         # G0: raw input == frozen data/raw/<loc>.csv
            key = f"data/raw/{loc}.csv"
            if sha256(Path(src)) != json.loads(a.manifest.read_text())["files"].get(key):
                raise SystemExit(f"G0: {src} sha256 != MANIFEST {key}")
            log(f"[{loc}] {src} sha256 == MANIFEST {key}")
        from s2g import metrics as M
        fl = M.ks_floor(d["train_phys"], d["val_phys"])
        if a.floor_csv:
            ff = frozen_floor(a.floor_csv, loc)
            df = max(abs(fl[f] - ff[f]) for f in ff)
            if set(ff) != set(fl) or df > TRI_TOL:
                raise SystemExit(f"G0 floor mismatch at {loc}: max |diff| {df:.3e} (data != frozen floor input)")
            log(f"[{loc}] floor == frozen ks_floor.csv (max |diff| {df:.1e}); data {src}")
        for i, (js, pt, rec) in enumerate(todo, 1):
            t0 = time.time()
            ev, ca = draws(pt, rec, d["scaler"])
            q = distances(d["train_phys"], d["val_phys"], ev, ca, fl)
            g, t = gate(q, rec), triangle(q)
            r = row(rec, q, g, t, env)
            worst["raw"] = max(worst["raw"], g["gate_raw_diff"]); worst["cal"] = max(worst["cal"], g["gate_cal_diff"])
            worst["slack"] = min(worst["slack"], t["min_slack_feature"], t["slack_mean"])
            log(f"[{loc}] {i}/{len(todo)} {rec['config']} s{rec['seed']}: gate raw {g['gate_raw_diff']:.2e} "
                  f"cal {g['gate_cal_diff']:.2e} | d_cal {r['d_cal_train']:.4f} floor {r['ks_floor']:.4f} "
                  f"slack {t['min_slack_feature']:.2e} | {time.time() - t0:.0f}s")
            if not (g["ok"] and t["ok"]):
                with open(fails, "a") as f:
                    f.write(json.dumps(r) + "\n")
                what = "G1 (recomputed KS != JSON)" if not g["ok"] else "G2 (triangle inequality violated: BUG)"
                msg = (f"{what} at {loc} {rec['config']} seed {rec['seed']}: raw diff {g['gate_raw_diff']:.3e}, "
                       f"cal diff {g['gate_cal_diff']:.3e}, min slack {t['min_slack_feature']:.3e}; "
                       f"this device {env}, original {rec.get('gpu')}")
                if not a.diagnose:
                    raise GateError(msg + ". Stopped; no table is produced.")
                log("  DIAGNOSE:", msg)
                continue
            with open(part, "a") as f:
                f.write(json.dumps(r) + "\n"); f.flush(); os.fsync(f.fileno())
    log(f"gate summary: max raw diff {worst['raw']:.3e}, max cal diff {worst['cal']:.3e} (tol {GATE_TOL}); "
          f"min triangle slack {worst['slack']:.3e} (tol -{TRI_TOL})")
    if fails.exists():
        log(f"gate failures recorded in {fails}: no table may be produced")


def finalize(a):
    import pandas as pd
    from s2g.config import LOCS, SEEDS
    out = a.out / OUT_CSV
    if out.exists():
        raise SystemExit(f"{out} exists (written once)")
    base = a.base_manifest_pf                        # preflight: never write an unregistrable table
    if not base.exists():
        raise SystemExit(f"{base} not found (the repo's MANIFEST_post_freeze.json, read only)")
    if (a.out / "gate_failures.jsonl").exists():
        raise SystemExit("gate failures recorded: refusing to write the table")
    rows = [json.loads(l) for p in sorted((a.out / "partial").glob("*.jsonl")) for l in p.read_text().splitlines()]
    df = pd.DataFrame(rows).drop_duplicates(["loc", "config", "seed"])
    want = {(l, c, s) for l in (a.locs or LOCS) for c in (a.configs or calibrated_configs()) for s in SEEDS}
    have = set(zip(df["loc"], df["config"], df["seed"]))
    if want - have:
        raise SystemExit(f"{len(want - have)} run(s) missing, e.g. {sorted(want - have)[:5]}")
    df = df[[k in want for k in zip(df["loc"], df["config"], df["seed"])]]
    if not (df.gate_ok.all() and df.triangle_ok.all()):
        raise SystemExit("a stored row failed a gate: refusing to write the table")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write(f"# produced by scripts/distance_extra.py (post-freeze, recomputed from runs/*/*.pt with frozen seeds); "
                f"gates: |KS - JSON| <= {GATE_TOL}, triangle slack >= -{TRI_TOL}; protocol sha256 {PROTOCOL_SHA[:12]}\n")
        df.sort_values(["loc", "config", "seed"]).to_csv(f, index=False)
    log(f"wrote {out}: {len(df)} runs; max gate diff raw {df.gate_raw_diff.max():.3e} cal {df.gate_cal_diff.max():.3e}; "
          f"min slack {min(df.min_slack_feature.min(), df.slack_mean.min()):.3e}")
    from datetime import datetime, timezone
    m = json.loads(base.read_text())                 # repo copy is read only; the updated copy goes to <out>
    m["files"]["scripts/distance_extra.py"] = sha256(Path(__file__))
    m["files"][f"results_extra/{OUT_CSV}"] = sha256(out)
    m["updated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (a.out / "MANIFEST_post_freeze.json").write_text(json.dumps(m, indent=2) + "\n")
    log(f"registered scripts/distance_extra.py and results_extra/{OUT_CSV} in {a.out / 'MANIFEST_post_freeze.json'}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs-dir", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="dedicated output folder (created if missing); nothing is written elsewhere")
    ap.add_argument("--locs", nargs="*")
    ap.add_argument("--configs", nargs="*", help="default: the 8 calibrated arms + rcgan + timegan")
    ap.add_argument("--manifest", type=Path, help="frozen MANIFEST.json (G0 sha256 checks)")
    ap.add_argument("--floor-csv", type=Path, help="frozen results/ks_floor.csv (G0 data check)")
    ap.add_argument("--probe", action="store_true", help="one arm, one RCGAN, one TimeGAN run on the first location")
    ap.add_argument("--diagnose", action="store_true", help="record gate failures and continue (never yields a table)")
    ap.add_argument("--cpu", action="store_true", help="force CPU (original draws were on CUDA: G1 will likely fail)")
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--base-manifest-pf", type=Path, default=ROOT / "MANIFEST_post_freeze.json",
                    help="current MANIFEST_post_freeze.json (read only); the updated copy is written to --out")
    a = ap.parse_args(argv)
    if a.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    global LOG
    a.out.mkdir(parents=True, exist_ok=True)
    LOG = a.out / "run_log.txt"
    log(f"--- {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(sys.argv[1:] if argv is None else argv)}")
    finalize(a) if a.finalize else run(a)


if __name__ == "__main__":
    try:
        main()
    except GateError as e:
        sys.exit(f"STOP: {e}")
