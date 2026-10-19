"""Read per-run JSONs (written once by train/baselines) into flat frames."""
import json
from pathlib import Path
import pandas as pd
from .config import ROOT

STAGES = ("raw", "cal", "proj")


def load_runs(root=None, names=None, locs=None) -> pd.DataFrame:
    rows = []
    for f in sorted(Path(root or ROOT / "runs").glob("*/*.json")):
        if f.parent.name.startswith("_"):
            continue
        r = json.loads(f.read_text())
        if (names and r["config"] not in names) or (locs and r["loc"] not in locs):
            continue
        flat = {k: r[k] for k in ("config", "loc", "seed", "train_seconds", "eval_seconds", "gpu") if k in r}
        flat["tie_frac_mean"] = r.get("tie_frac_mean")
        for s in STAGES:
            flat.update({f"{s}_{k}": v for k, v in r[s].items()})
        flat.update({k: v for k, v in r["seq"].items()})
        rows.append(flat)
    return pd.DataFrame(rows)
