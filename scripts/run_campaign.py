"""Resumable campaign. Runs are written once; existing run JSONs are skipped.
   --stage pilot | full | baselines"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from s2g.baselines import run_baseline
from s2g.config import P, ROOT, LOCS, SEEDS, assert_frozen
from s2g.train import run_one

ap = argparse.ArgumentParser(); ap.add_argument("--stage", required=True); ap.add_argument("--loc", default=None, help="run one location only"); ap.add_argument("--seed", type=int, default=None, help="run one seed only")
a = ap.parse_args()
assert_frozen()
active = json.loads((ROOT / "runs/budget.json").read_text())["active_arms"]
if a.stage == "pilot":
    jobs = [(x, l, s) for l in P["pilot"]["locations"] for x in P["pilot"]["arms"] for s in P["pilot"]["seeds"]]
elif a.stage == "full":
    # pilot arms first inside a location, then the rest; the gate only decides the paper claim, never the scope
    order = lambda x: (x not in P["pilot"]["arms"], x)
    jobs = [(x, l, s) for l in LOCS for x in sorted(active, key=order) for s in SEEDS]
else:
    jobs = [(m, l, s) for l in LOCS for m in P["baselines"]["models"] for s in SEEDS]
jobs = [j for j in jobs if a.loc in (None, j[1]) and a.seed in (None, j[2])]
assert jobs, f"no jobs for stage={a.stage} loc={a.loc} seed={a.seed}"
for i, (x, l, s) in enumerate(jobs, 1):
    t = time.time()
    r = (run_baseline if a.stage == "baselines" else run_one)(x, l, s)
    print(f"[{i}/{len(jobs)}] {l} {x} {s}: {'skip' if r is None else f'{time.time() - t:.0f}s'}", flush=True)
(ROOT / f"runs/.{a.stage}{'_' + a.loc if a.loc else ''}{'_' + str(a.seed) if a.seed else ''}_done").write_text(time.ctime())
