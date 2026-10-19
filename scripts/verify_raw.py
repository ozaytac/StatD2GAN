"""Raw-input integrity. Default: sha256 of data/raw/*.csv against data/raw.sha256 (used by make).
--redownload: numeric comparison of data/raw_redownload/ against the pinned files."""
import hashlib, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from s2g.config import P, ROOT, LOCS

if "--redownload" in sys.argv:
    tol = P["data"]["raw_tolerance"]
    for l in LOCS:
        a, b = (pd.read_csv(ROOT / f"data/{d}/{l}.csv", parse_dates=["valid_time"]).set_index("valid_time")
                for d in ("raw", "raw_redownload"))
        cols = ["u10", "v10", "d2m", "t2m", "msl", "sp"]
        err = np.abs(a[cols].astype("float32") - b.reindex(a.index)[cols].astype("float32")).max().max()
        print(f"{l}: max abs diff {err:.3g}"); assert err <= tol * max(1, np.abs(a[cols].to_numpy()).max()), l
    sys.exit(0)
sel = [a for a in sys.argv[1:] if a in LOCS] or LOCS      # optional: locations to check
bad = 0
for line in (ROOT / "data/raw.sha256").read_text().split("\n"):
    if line.strip() and Path(line.split()[1]).stem in sel:
        h, f = line.split(); ok = hashlib.sha256((ROOT / f).read_bytes()).hexdigest() == h
        print(("OK  " if ok else "BAD ") + f); bad += not ok
sys.exit(bad)
