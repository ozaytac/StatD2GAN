"""results/seq_level.csv: sequence-level statistics on temperature (calibrated draws) for StatD2GAN
(full_model), RCGAN, TimeGAN, and the real held-out sequences (model=real_val, seed=-1)."""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from s2g import data, metrics, runs
from s2g.config import P, ROOT, LOCS, write_csv, assert_frozen

assert_frozen()
r = runs.load_runs(names=["full_model", *P["baselines"]["models"]])
r["model"] = r.config.replace({"full_model": "statd2gan"})
rows = r[["loc", "model", "seed", "seq_mean_sd", "within_sd", "step_diff"]]
real = pd.DataFrame([dict(loc=l, model="real_val", seed=-1,
                          **metrics.seq_stats(data.load(ROOT / f"data/processed/{l}.npz")["val_phys"])) for l in LOCS])
write_csv(pd.concat([real, rows]).sort_values(["loc", "model", "seed"]), ROOT / "results/seq_level.csv",
          "make_seq_level.py", ["runs/*/*.json", "data/processed/*.npz"])
