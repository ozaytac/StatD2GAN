"""Pre-declared decision rules (protocol.yaml: pilot, decision_rules) evaluated in code."""
import numpy as np, pandas as pd
from .config import P

A = P["inference"]["alpha"]


def pilot_gate(runs: pd.DataFrame):
    """Dubai: pct increase of mean tau-MAE for no_D_sorted vs full_model >= threshold AND
    per-seed paired difference positive in >= gate_min_positive_seeds seeds."""
    pc = P["pilot"]
    d = runs[runs["loc"] == "dubai"].pivot(index="seed", columns="config", values="cal_kendall_mae")
    pct = 100 * (d["no_D_sorted"].mean() / d["full_model"].mean() - 1)
    pos = int(((d["no_D_sorted"] - d["full_model"]) > 0).sum())
    return dict(pct=float(pct), positive_seeds=pos, n_seeds=len(d),
                passed=bool(pct >= pc["gate_min_pct"] and pos >= pc["gate_min_positive_seeds"]))


def d1(ratio_median_full, holm_p_nocal):
    return dict(holds=bool(ratio_median_full <= 1.05 and holm_p_nocal < A),
                detail=f"median ks/floor={ratio_median_full:.3f}, Holm p(no_calibration)={holm_p_nocal:.4g}")


def d2(w: pd.DataFrame):
    """w: contrast table (s2g.stats.contrast_table). One verdict per arm on kendall_mae."""
    t = w[w.metric == "kendall_mae"]
    return {r.contrast: dict(holds=bool(r.p_holm < A and r.median_diff > 0 and r.n_loc_median_pos >= 4),
                             detail=f"median diff={r.median_diff:.4f}, Holm p={r.p_holm:.4g}, "
                                    f"positive in {r.n_loc_median_pos}/5 location medians")
            for r in t.itertuples()}


def d4(w: pd.DataFrame):
    r = w[(w.metric == "kendall_mae") & (w.contrast == "rank_D_sort")].iloc[0]
    return dict(holds=bool(r.median_diff > 0 and r.p_holm < A),
                detail=f"rank_D_sort median diff={r.median_diff:.4f}, Holm p={r.p_holm:.4g}")


def d5(proj_long: pd.DataFrame, mean_floor: float):
    """proj_long: full_model runs with cal_/proj_ ks_stat and proj_phys_viol_rate."""
    zero = bool((proj_long.proj_phys_viol_rate == 0).all())
    dks = float((proj_long.proj_ks_stat - proj_long.cal_ks_stat).abs().max())
    return dict(holds=bool(zero and dks <= 0.05 * mean_floor), detail=f"all zero={zero}, max|dKS|={dks:.5f}")


def d6(seq: pd.DataFrame, models=("rcgan", "timegan")):
    """seq: rows (loc, model, seed, seq_mean_sd); real rows have model == 'real_val'."""
    out = {}
    real = seq[seq.model == "real_val"].set_index("loc").seq_mean_sd
    for m in models:
        med = seq[seq.model == m].groupby("loc").seq_mean_sd.median()
        n = int((med < 0.1 * real.reindex(med.index)).sum())
        out[m] = dict(holds=bool(n >= 4), detail=f"{n}/{len(med)} locations below 10% of real seq_mean_sd")
    return out
