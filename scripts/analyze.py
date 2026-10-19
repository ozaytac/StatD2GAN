"""CSV-only analysis layer. Reads results/{abl_perseed,baselines,projection,seq_level,ks_floor}.csv and writes
ks_ratio, summary, pct_change, wilcoxon, bootstrap, baseline_tests, run_noise (CSV) + decisions.json."""
import itertools, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from s2g import decide, stats
from s2g.config import P, ROOT, LOCS, read_csv, write_csv, assert_frozen

assert_frozen()
R = ROOT / "results"; METRICS = P["inference"]["metrics"]
abl, base, proj, seq, fl = (read_csv(R / f) for f in ("abl_perseed.csv", "baselines.csv", "projection.csv", "seq_level.csv", "ks_floor.csv"))
floor = fl.groupby("loc").ks_floor.mean()
arms = [a for a in abl.config.unique() if a != "full_model"]
src = lambda *f: [f"results/{x}" for x in f]

# 1. KS relative to the real train/held-out floor (a practical reference, not a lower bound)
kr = abl[["loc", "config", "seed"]].assign(ks_ratio=abl.ks_stat / abl["loc"].map(floor), floor=abl["loc"].map(floor))
write_csv(kr, R / "ks_ratio.csv", "analyze.py", src("abl_perseed.csv", "ks_floor.csv"))

# 2. summary (mean, SD over seeds)
cols = METRICS + ["phys_viol_raw", "tie_frac_mean"]
s = abl.groupby(["loc", "config"])[cols].agg(["mean", "std"]); s.columns = ["_".join(c) for c in s.columns]
s = s.reset_index(); s["ks_floor"] = s["loc"].map(floor); s["ks_over_floor"] = s.ks_stat_mean / s.ks_floor
write_csv(s, R / "summary.csv", "analyze.py", src("abl_perseed.csv", "ks_floor.csv"))

# 3. percent change vs full_model, from per-seed data
m = abl.groupby(["loc", "config"])[METRICS].mean()
rows = []
for l, a in itertools.product(LOCS, arms):
    b = m.loc[(l, "full_model")]
    rows.append(dict(loc=l, config=a, **{k: (100 * (m.loc[(l, a), k] - b[k]) / b[k] if b[k] else np.nan) for k in METRICS}))
write_csv(pd.DataFrame(rows).round(2), R / "pct_change.csv", "analyze.py", src("abl_perseed.csv"))

# 4. Wilcoxon + Holm (family = the contrasts within a metric)
w = stats.contrast_table(abl, arms); w["family_size"] = len(arms)
write_csv(w, R / "wilcoxon.csv", "analyze.py", src("abl_perseed.csv"))

# 5. location-clustered bootstrap of the paired difference
rows = []
for mt, a in itertools.product(METRICS, arms):
    pt, lo, hi = stats.clustered_bootstrap(stats.paired_diffs(abl, mt, a))
    rows.append(dict(metric=mt, contrast=a, point=pt, lo=lo, hi=hi))
write_csv(pd.DataFrame(rows), R / "bootstrap.csv", "analyze.py", src("abl_perseed.csv"))

# 6. baselines vs full_model, same pairing and Holm within metric
both = pd.concat([abl[abl.config == "full_model"], base])
bt = stats.contrast_table(both, P["baselines"]["models"]); bt["family_size"] = len(P["baselines"]["models"])
write_csv(bt, R / "baseline_tests.csv", "analyze.py", src("abl_perseed.csv", "baselines.csv"))

# 7. run-to-run noise: median pairwise |d tau| between seeds of the same (loc, arm), full_model
fm = abl[abl.config == "full_model"].pivot(index="seed", columns="loc", values="kendall_mae")
pair = np.median([abs(x - y) for l in fm for x, y in itertools.combinations(fm[l], 2)])
write_csv(pd.DataFrame([dict(median_pairwise_abs_dtau=pair, mean_tau_full=fm.to_numpy().mean(),
                             band_pct=100 * pair / fm.to_numpy().mean())]), R / "run_noise.csv", "analyze.py", src("abl_perseed.csv"))

# 8. pre-declared decision rules
wp = lambda mt, a: w[(w.metric == mt) & (w.contrast == a)].iloc[0]
dec = dict(D1=decide.d1(float(kr[kr.config == "full_model"].ks_ratio.median()), float(wp("ks_stat", "no_calibration").p_holm)),
           D2=decide.d2(w), D4=decide.d4(w) if "rank_D_sort" in arms else None,
           D5=decide.d5(proj, float(floor.mean())), D6=decide.d6(seq))
assert not (R / "decisions.json").exists(), "decisions.json exists (written once)"
(R / "decisions.json").write_text(json.dumps(dec, indent=1)); print(json.dumps(dec, indent=1))
