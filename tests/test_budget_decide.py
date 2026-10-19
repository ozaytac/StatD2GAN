import numpy as np, pandas as pd
from s2g import budget, decide


def test_run_seconds_projection():
    s = budget.run_seconds([10, 2, 2, 2], 150, 15, 20)          # 15*10 + 135*2 + 20
    assert s == 15 * 10 + 135 * 2 + 20
    assert budget.run_seconds([2, 2, 2], 150, 0, 0) == 300      # no warmup arm


def test_ladder_drops_in_order_and_never_touches_others():
    per = {"full_model": 650, "no_calibration": 650, "no_aux_losses": 740, "fixed_lambda": 650, "moment_D_stat": 260}
    base = {"rcgan": 60, "timegan": 200}
    ladder = ["no_aux_losses", "fixed_lambda", "moment_D_stat"]
    full = budget.project(per, base, 25, 1.0, 1e9, ladder)
    assert full["dropped"] == [] and full["fits"]
    h = full["full_scope_hours"]
    p = budget.project(per, base, 25, 1.0, h - 0.5, ladder)
    assert p["dropped"] == ["no_aux_losses"] and p["fits"]
    q = budget.project(per, base, 25, 1.0, 1.0, ladder)
    assert q["dropped"] == ladder and not q["fits"] and "full_model" in q["active_arms"]


def _pilot(pct, pos):
    rows = []
    for s in range(5):
        rows += [dict(loc="dubai", seed=s, config="full_model", cal_kendall_mae=0.10),
                 dict(loc="dubai", seed=s, config="no_D_sorted",
                      cal_kendall_mae=0.10 * (1 + pct / 100) * (1 if s < pos else -0.0 + 0.2))]
    return pd.DataFrame(rows)


def test_pilot_gate_needs_both_conditions():
    assert decide.pilot_gate(_pilot(30, 5))["passed"]
    assert not decide.pilot_gate(_pilot(30, 3))["passed"]        # too few positive seeds
    r = _pilot(10, 5); assert not decide.pilot_gate(r)["passed"]  # effect too small


def test_d1_d4_d5_d6():
    assert decide.d1(1.01, 0.001)["holds"] and not decide.d1(1.2, 0.001)["holds"] and not decide.d1(1.0, 0.2)["holds"]
    w = pd.DataFrame([dict(metric="kendall_mae", contrast="rank_D_sort", median_diff=0.08, p_holm=0.01, n_loc_median_pos=5),
                      dict(metric="kendall_mae", contrast="no_D_stat", median_diff=0.01, p_holm=0.4, n_loc_median_pos=3)])
    assert decide.d4(w)["holds"] and decide.d2(w)["rank_D_sort"]["holds"] and not decide.d2(w)["no_D_stat"]["holds"]
    pr = pd.DataFrame(dict(proj_phys_viol_rate=[0.0, 0.0], proj_ks_stat=[.04, .04], cal_ks_stat=[.04, .0401]))
    assert decide.d5(pr, 0.05)["holds"]
    pr.loc[0, "proj_phys_viol_rate"] = 1e-4; assert not decide.d5(pr, 0.05)["holds"]
    seq = pd.DataFrame([dict(loc=l, model=m, seed=s, seq_mean_sd=v) for l in "abcde"
                        for m, s, v in [("real_val", -1, 5.0), ("timegan", 1, 0.01), ("rcgan", 1, 1.0)]])
    r = decide.d6(seq); assert r["timegan"]["holds"] and not r["rcgan"]["holds"]
