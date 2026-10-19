"""results/abl_perseed.csv: one row per (loc, arm, seed); calibrated-stage metrics + diagnostics."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from s2g import runs
from s2g.config import ROOT, ARMS, write_csv, assert_frozen

assert_frozen()
r = runs.load_runs(names=ARMS)
o = r[["config", "seed", "loc"]].copy()
for m in ("ks_stat", "kendall_mae", "acf_mae", "phys_viol_rate"):
    o[m] = r[f"cal_{m}"]
o["ks_stat_raw"], o["phys_viol_raw"], o["tie_frac_mean"], o["train_seconds"] = r.raw_ks_stat, r.raw_phys_viol_rate, r.tie_frac_mean, r.train_seconds
write_csv(o.sort_values(["loc", "config", "seed"]), ROOT / "results/abl_perseed.csv", "make_abl_perseed.py", ["runs/*/*.json"])
