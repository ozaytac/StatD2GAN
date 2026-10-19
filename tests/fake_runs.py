"""Schema-identical fake run JSONs (random metrics with known effects) for pipeline integration tests.
Never used for results: written only into a temporary copy of the package."""
import json
from pathlib import Path
import numpy as np

METS = ("ks_stat", "kendall_mae", "acf_mae", "phys_viol_rate")


def write(root: Path, locs, arms, seeds, floors, baselines=("rcgan", "timegan"), pilot_pass=True):
    r = np.random.default_rng(0)
    for l in locs:
        (root / "runs" / l).mkdir(parents=True, exist_ok=True)
        for name in [*arms, *baselines]:
            for s in seeds:
                tau_base = 0.15 + r.normal(0, .03)
                eff = {"no_D_sorted": .08, "rank_D_sort": .08}.get(name, 0.0) + (.2 if name in baselines else 0)
                raw = dict(ks_stat=.2 + r.normal(0, .01), kendall_mae=tau_base, acf_mae=.3, phys_viol_rate=abs(r.normal(.001, .0005)))
                calk = floors[l] * (5 if name == "no_calibration" else 1.01)
                cal = dict(ks_stat=calk, kendall_mae=tau_base + eff, acf_mae=.3 + r.normal(0, .02), phys_viol_rate=abs(r.normal(.04, .01)))
                if name == "no_calibration": cal = dict(raw)
                proj = dict(cal, phys_viol_rate=0.0)
                seq = dict(seq_mean_sd=0.01 if name == "timegan" else 4 + r.normal(0, .2), within_sd=4.0, step_diff=1.0)
                rec = dict(config=name, loc=l, seed=s, epoch_seconds=[1.0] * 3, train_seconds=650.0, eval_seconds=20.0, gpu="FAKE",
                           raw=raw, cal=cal, proj=proj, seq=seq)
                if name != "no_calibration": rec["tie_frac_mean"] = .05
                (root / "runs" / l / f"{name}_{s}.json").write_text(json.dumps(rec))
