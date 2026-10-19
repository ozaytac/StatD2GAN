"""results/ks_floor.csv: real TRAIN vs real HELD-OUT KS sup-distance per location and feature."""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from s2g import data, metrics
from s2g.config import ROOT, LOCS, write_csv, assert_frozen

assert_frozen()
sel = [a for a in sys.argv[1:] if a in LOCS] or LOCS      # optional: locations (per-session partial floor)
rows = []
for l in sel:
    d = data.load(ROOT / f"data/processed/{l}.npz")
    rows += [dict(loc=l, feature=f, ks_floor=v) for f, v in metrics.ks_floor(d["train_phys"], d["val_phys"]).items()]
write_csv(pd.DataFrame(rows), ROOT / "results/ks_floor.csv", "make_floor.py", [f"data/processed/{l}.npz" for l in sel])
