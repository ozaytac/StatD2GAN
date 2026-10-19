"""Feasibility: (1) short probe of every arm on one location (throwaway), (2) one REAL
baseline run each (ankara, seed 42; kept as campaign runs), (3) GPU-hour projection + ladder.
Reads timings only. Writes runs/budget.json."""
import json, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from s2g import budget
from s2g.baselines import run_baseline
from s2g.config import P, ROOT, ARMS, SEEDS, LOCS, assert_frozen
from s2g.train import run_one

assert_frozen()
LOC = next((a for a in sys.argv[1:] if a in LOCS), "ankara")      # timing location (any; seed = first protocol seed)
out = ROOT / "runs/budget.json"
if out.exists():
    sys.exit("budget.json exists (written once)")
probe_dir, per_arm = Path(tempfile.mkdtemp()), {}
for a in ARMS:
    r = run_one(a, LOC, SEEDS[0], out_dir=probe_dir, epochs=4)
    w = P["arms"][a].get("warmup_epochs", P["model"]["warmup_epochs"])
    per_arm[a] = budget.run_seconds(r["epoch_seconds"], P["model"]["total_epochs"], w, r["eval_seconds"])
    print(f"{a}: projected {per_arm[a]:.0f}s/run")
base = {}
for m in P["baselines"]["models"]:
    r = run_baseline(m, LOC, SEEDS[0]) or json.loads((ROOT / f"runs/{LOC}/{LOC}_{m}_seed{SEEDS[0]}.json").read_text())
    base[m] = r["train_seconds"] + r["eval_seconds"]
b = P["budget"]
res = budget.project(per_arm, base, len(LOCS) * len(SEEDS), b["safety_factor"], b["gpu_hours"], b["ladder"])
res.update(per_arm_seconds=per_arm, baseline_seconds=base, gpu_hours_available=b["gpu_hours"])
out.write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
if not res["fits"]:
    sys.exit("does not fit even after the ladder: scope decision needed (no metric was read)")
