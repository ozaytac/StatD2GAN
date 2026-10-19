"""results/baselines.csv: one row per (loc, baseline, seed); same pipeline and columns as abl_perseed."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from s2g import runs
from s2g.config import P, ROOT, write_csv, assert_frozen

assert_frozen()
r = runs.load_runs(names=P["baselines"]["models"])
o = r[["config", "seed", "loc"]].copy()
for m in ("ks_stat", "kendall_mae", "acf_mae", "phys_viol_rate"):
    o[m] = r[f"cal_{m}"]
o["ks_stat_raw"], o["phys_viol_raw"], o["tie_frac_mean"], o["train_seconds"] = r.raw_ks_stat, r.raw_phys_viol_rate, r.tie_frac_mean, r.train_seconds
write_csv(o.sort_values(["loc", "config", "seed"]), ROOT / "results/baselines.csv", "make_baselines.py", ["runs/*/*.json"])
