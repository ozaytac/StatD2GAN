"""Cross-location analysis: summary tables, paired Wilcoxon + Holm tests,
LaTeX table export, and the calibration constraint-projection decision.

This is the "one cell, five locations" analysis that turns the per-location
`ablation_raw.csv` files into the tables reported in the paper. Evidence
rule (see the project's evidence log): every numerical claim is computed
from `abl_perseed_v2.csv`, never from an intermediate summary of a summary.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import mannwhitneyu, wilcoxon

from .config import C, ExperimentConfig, LocationPaths, make_paths, set_seed
from .data import denorm, load_location
from .metrics import compute_metrics, fit_calibration, ks_noise_floor, phys_viol_rate, project_physics
from .models import Generator
from .train import _generate, rescore
from .utils import load_json, atomic_json_dump

ALL_LOCATIONS = ["ankara", "dubai", "bergen", "lhasa", "yakutsk"]
METRICS = ["ks_stat", "kendall_mae", "acf_mae", "phys_viol_rate"]


def _light_diag(cfg: ExperimentConfig, seed: int, device: str, paths: LocationPaths,
                 real_train_phys: np.ndarray, scaler) -> tuple[float, float]:
    """Raw violation rate + calibration tie fraction only — skips the full
    `compute_metrics` call, so backfilling missing diagnostic columns on old
    runs is fast."""
    G = Generator(cfg.noise_dim).to(device)
    sd = torch.load(f"{paths.gen_root}/{cfg.name}_{seed}_{cfg.total_epochs}.pt", map_location=device)
    G.load_state_dict({k: v.float() for k, v in sd.items()})
    set_seed(seed)
    syn = denorm(_generate(G, cfg.noise_dim, device, n=5000), scaler, C.scalable_idx)
    raw = phys_viol_rate(syn)
    tie = float("nan")
    if cfg.use_calibration:
        _, tie_frac = fit_calibration(real_train_phys, syn)
        tie = float(np.mean(list(tie_frac().values())))
    del G
    torch.cuda.empty_cache()
    return raw, tie


def backfill_diagnostics(device: str, configs_by_name: dict, base_root: str = C.base_root,
                          locations=ALL_LOCATIONS) -> None:
    """Fill in `phys_viol_raw` / `tie_frac_mean` for any ablation rows saved
    before those diagnostic columns existed, and generate a missing
    `ks_noise_floor.json` for any location that predates the patch."""
    for loc in locations:
        paths = make_paths(loc, base_root)
        loc_data = load_location(paths, C)
        path = f"{paths.out_root}/ablation_raw.csv"
        df = pd.read_csv(path)
        for col in ["phys_viol_raw", "tie_frac_mean"]:
            if col not in df:
                df[col] = np.nan
        todo = df[df["phys_viol_raw"].isna()]
        for i, r in todo.iterrows():
            raw, tie = _light_diag(configs_by_name[r["config"]], int(r["seed"]), device, paths,
                                   loc_data.real_train_phys, loc_data.scaler)
            df.loc[i, "phys_viol_raw"] = raw
            df.loc[i, "tie_frac_mean"] = tie
        df.to_csv(path, index=False)

        floor_path = f"{paths.out_root}/ks_noise_floor.json"
        if not os.path.exists(floor_path):
            floor = ks_noise_floor(loc_data.real_train_phys, loc_data.real_val_phys)
            atomic_json_dump(floor_path, {"per_feature": {str(k): v for k, v in floor.items()},
                                          "mean": float(np.mean(list(floor.values())))})
        print(f"[{loc}] backfill complete ({len(todo)} rows)")


def load_all(base_root: str = C.base_root, locations=ALL_LOCATIONS, out_dir: str = ".") -> tuple[pd.DataFrame, dict]:
    """Concatenate every location's `ablation_raw.csv` plus its KS noise floor."""
    frames, floors = [], {}
    for loc in locations:
        paths = make_paths(loc, base_root)
        d = pd.read_csv(f"{paths.out_root}/ablation_raw.csv")
        d["loc"] = loc
        frames.append(d)
        floors[loc] = load_json(f"{paths.out_root}/ks_noise_floor.json")["mean"]
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(f"{out_dir}/abl_perseed_v2.csv", index=False)
    return df, floors


def make_summary(df: pd.DataFrame, floors: dict, out_dir: str = ".") -> pd.DataFrame:
    """Per-(location, config) mean/std summary, plus KS-over-noise-floor ratio."""
    g = df.groupby(["loc", "config"])
    out = g[METRICS + ["phys_viol_raw", "tie_frac_mean"]].agg(["mean", "std"])
    out.columns = ["_".join(c) for c in out.columns]
    out = out.reset_index()
    out["ks_floor"] = out["loc"].map(floors)
    out["ks_over_floor"] = out["ks_stat_mean"] / out["ks_floor"]
    out.to_csv(f"{out_dir}/all_locations_summary_v2.csv", index=False)
    return out


def pct_table(df: pd.DataFrame, contrasts: list[str], out_dir: str = ".") -> pd.DataFrame:
    """Percent change of each ablation arm vs. `full_model`, computed
    directly from per-seed raw data (evidence rule E1) rather than from an
    already-averaged summary table."""
    rows = []
    for loc in ALL_LOCATIONS:
        base = df[(df.loc == loc) & (df.config == "full_model")][METRICS].mean()
        for c in contrasts:
            m = df[(df.loc == loc) & (df.config == c)][METRICS].mean()
            rows.append({"loc": loc, "config": c,
                         **{k: 100 * (m[k] - base[k]) / base[k] if base[k] != 0 else np.nan
                            for k in METRICS}})
    p = pd.DataFrame(rows).round(1)
    p.to_csv(f"{out_dir}/pct_change_v2.csv", index=False)
    return p


def wilcoxon_table(df: pd.DataFrame, contrasts: list[str], out_dir: str = ".") -> pd.DataFrame:
    """Paired Wilcoxon signed-rank test over the 25 (location, seed) pairs
    for each contrast-vs-full_model, with Holm correction applied within
    each metric family (spec E2)."""
    piv = {m: df.pivot_table(index=["loc", "seed"], columns="config", values=m) for m in METRICS}
    rows = []
    for m in METRICS:
        P = piv[m]
        res = []
        for c in contrasts:
            d = (P[c] - P["full_model"]).dropna()
            try:
                p = wilcoxon(d, alternative="two-sided", mode="auto").pvalue
            except ValueError:
                p = np.nan  # every difference is exactly zero
            signs = " ".join(f"{loc[:2].upper()}:{int((d.loc[loc] > 0).sum())}/{len(d.loc[loc])}"
                             for loc in ALL_LOCATIONS if loc in d.index.get_level_values(0))
            res.append({"metric": m, "contrast": c, "median_diff": d.median(),
                        "p_wilcoxon": p, "signs(+)": signs})
        res = sorted(res, key=lambda r: (np.isnan(r["p_wilcoxon"]), r["p_wilcoxon"]))
        k = len([r for r in res if not np.isnan(r["p_wilcoxon"])])
        for i, r in enumerate(res):
            r["p_holm"] = min(1.0, r["p_wilcoxon"] * (k - i)) if not np.isnan(r["p_wilcoxon"]) else np.nan
        rows += res
    t = pd.DataFrame(rows)
    t[["median_diff", "p_wilcoxon", "p_holm"]] = t[["median_diff", "p_wilcoxon", "p_holm"]].round(4)
    t.to_csv(f"{out_dir}/wilcoxon_v2.csv", index=False)
    return t


def run_noise(df: pd.DataFrame) -> pd.Series:
    """Within-run noise estimate: |tau_MAE(no_calibration) - tau_MAE(full_model)|
    across matched (loc, seed) pairs. Calibration barely touches Kendall
    tau, so this isolates seed-to-seed noise from any calibration effect."""
    P = df.pivot_table(index=["loc", "seed"], columns="config", values="kendall_mae")
    d = (P["no_calibration"] - P["full_model"]).abs().dropna()
    return pd.Series({"mean": d.mean(), "median": d.median(), "max": d.max(),
                      "n_pairs": len(d)}).round(4)


def lambda_insurance(df: pd.DataFrame) -> pd.DataFrame:
    """One-sided Mann-Whitney test of the (retracted) 'lambda insurance'
    claim: does evolutionary weighting reduce per-location deviation
    relative to a fixed [1,1,1] baseline? Reported for completeness even
    though the claim did not survive (p=0.50, see the paper's null results)."""
    out = {}
    for m in ["kendall_mae", "acf_mae"]:
        P = df.pivot_table(index=["loc", "seed"], columns="config", values=m)
        dev = lambda col: (P[col] - P[col].groupby(level=0).transform("median")).abs().dropna()
        a, b = dev("full_model"), dev("fixed_lambda")
        out[m] = {"MAD_full": a.mean().round(4), "MAD_fixed": b.mean().round(4),
                  "p_MW_one_sided": round(mannwhitneyu(a, b, alternative="less").pvalue, 4)}
    return pd.DataFrame(out).T


def write_latex(summary: pd.DataFrame, wtab: pd.DataFrame, floors: dict,
                contrasts: list[str], out_dir: str = ".") -> None:
    lines = []
    for m, cap in [("kendall_mae", "Kendall tau MAE"), ("acf_mae", "ACF MAE"),
                   ("phys_viol_rate", "Physics violation rate"), ("ks_stat", "KS sup-distance")]:
        w = summary.pivot(index="config", columns="loc", values=f"{m}_mean").round(3)
        s = summary.pivot(index="config", columns="loc", values=f"{m}_std").round(3)
        body = w.astype(str) + " $\\pm$ " + s.astype(str)
        body = body.reindex(["full_model"] + contrasts)
        lines.append(body.to_latex(caption=f"{cap} (mean $\\pm$ SD, N=5 seeds, held-out).",
                                   label=f"tab:{m}_v2", escape=False))
    lines.append(pd.Series(floors, name="KS noise floor").round(4).to_frame()
                 .to_latex(caption="Real train--val KS noise floor per location.",
                           label="tab:floor", escape=False))
    lines.append(wtab.to_latex(index=False, escape=False,
                 caption="Paired Wilcoxon over 25 (location, seed) pairs; Holm within metric.",
                 label="tab:wilcoxon"))
    with open(f"{out_dir}/tables_generated_v2.tex", "w") as f:
        f.write("\n\n".join(lines))


def projection_analysis(device: str, configs_by_name: dict, base_root: str = C.base_root,
                         out_dir: str = ".", seeds=C.seeds) -> pd.DataFrame:
    """Rescore `full_model` with and without the post-calibration physics
    projection, across all locations and seeds. Backs the paper's decision
    to apply projection: violations drop to ~0 at negligible KS cost."""
    rows = []
    for loc in ALL_LOCATIONS:
        paths = make_paths(loc, base_root)
        loc_data = load_location(paths, C)
        for s in seeds:
            for proj in [False, True]:
                m = rescore(configs_by_name["full_model"], s, device, paths,
                           loc_data.real_train_phys, loc_data.real_val_phys, loc_data.scaler,
                           projection=proj)
                rows.append({"loc": loc, "seed": s, "proj": proj, **{k: m[k] for k in METRICS}})
    pr = pd.DataFrame(rows)
    pr.to_csv(f"{out_dir}/projection_v2.csv", index=False)
    print(pr.groupby("proj")[METRICS].mean().round(4))
    return pr
