"""raw ERA5 CSV -> data/processed/{loc}.npz (physical-unit train/held-out windows + scaler)."""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from s2g import data
from s2g.config import ROOT, LOCS, assert_frozen

assert_frozen()
for loc in [a for a in sys.argv[1:] if a in LOCS] or LOCS:      # optional: locations to process
    out = ROOT / f"data/processed/{loc}.npz"
    if out.exists():
        continue
    d = data.build(pd.read_csv(ROOT / f"data/raw/{loc}.csv", parse_dates=["valid_time"]))
    data.save(out, d)
    print(f"{loc}: train {d['train_phys'].shape} val {d['val_phys'].shape} val_years {d['val_years']}")
