"""results/projection.csv: full_model runs; metrics raw / after calibration / after projection."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from s2g import runs
from s2g.config import ROOT, write_csv, assert_frozen

assert_frozen()
r = runs.load_runs(names=["full_model"])
cols = ["loc", "seed"] + [f"{s}_{m}" for m in ("ks_stat", "kendall_mae", "acf_mae", "phys_viol_rate") for s in ("raw", "cal", "proj")]
write_csv(r[cols].sort_values(["loc", "seed"]), ROOT / "results/projection.csv", "make_projection.py", ["runs/*/full_model_*.json"])
