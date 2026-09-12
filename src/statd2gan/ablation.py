"""Ablation arm definitions and experiment drivers (checkpoint/resume-safe).

Every arm below flips exactly one factor relative to `full_model`, so any
metric difference against `full_model` is attributable to that one factor
(with the caveat, see `train_experiment`, that run-to-run seed noise on
Kendall tau MAE is ~0.085 - single-seed comparisons are not valid; each arm
is run across all seeds in `config.C.seeds` and compared via the paired
Wilcoxon tests in `analysis.py`).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .config import C, ExperimentConfig, LocationPaths
from .metrics import bootstrap_ci, ks_noise_floor, METRIC_COLS
from .train import train_experiment
from .utils import atomic_json_dump, load_json

ABLATION_CONFIGS = [
    ExperimentConfig(name="full_model"),
    ExperimentConfig(name="no_D_temporal", use_D_temporal=False),
    ExperimentConfig(name="no_D_stat", use_D_stat=False),
    ExperimentConfig(name="no_D_sorted", use_D_sorted=False),
    ExperimentConfig(name="no_calibration", use_calibration=False),
    # spec B6: correct control — the evolutionary arm also starts at [1,1,1],
    # so the only attributable difference from full_model is adaptation itself.
    ExperimentConfig(name="fixed_lambda", use_evolutionary=False, fixed_lambda=[1.0, 1.0, 1.0]),
    # spec B2: the moment-based D_stat described in the paper text
    ExperimentConfig(name="moment_D_stat", stat_variant="moment"),
    # spec F3 (mandatory): copula-aware soft-rank D_sorted, run alongside the hard-sort arm
    ExperimentConfig(name="rank_D_sort", sorted_variant="rank"),
    # spec B3/F1 (mandatory, not optional): directly tests whether marginal
    # fit is achieved adversarially; warmup_epochs=0 alone is insufficient,
    # use_aux_losses=False also disables the post-epoch-35 aux-loss return.
    ExperimentConfig(name="no_aux_losses", warmup_epochs=0, use_aux_losses=False),
]

# Optional arms (spec B6 asymmetry question); not run by default.
OPTIONAL_CONFIGS = [
    ExperimentConfig(name="fixed_lambda_asym", use_evolutionary=False, fixed_lambda=[1.0, 5.0, 0.7]),
]

CONFIG_ORDER = [c.name for c in ABLATION_CONFIGS]

# Pre-registered pilot gate criterion (spec: run budget, v3) — written before
# any run, not fitted to the outcome after the fact.
PILOT_GATE = (
    "Pilot: Ankara + Dubai, 5 seeds. Gate question: does the D_sorted effect "
    "survive the held-out split? CRITERION: no_D_sorted degrades tau-MAE in "
    "Dubai by >= 20% AND at least 4 of 5 seeds agree in sign. If it passes, "
    "run the full campaign; if not, revisit the thesis and inspect rank_D_sort "
    "separately (if D_sorted drops but the rank variant holds, the mechanism "
    "is the copula channel)."
)


def run_ablation(device: str, paths: LocationPaths, train_tensor, real_train_phys: np.ndarray,
                  real_val_phys: np.ndarray, scaler, configs=None, seeds=None) -> pd.DataFrame:
    """Run every (config, seed) combination for one location, with
    checkpoint/resume: a run already present in `checkpoint.json` is skipped.
    """
    configs = configs or ABLATION_CONFIGS
    seeds = seeds or C.seeds

    # ---- noise floor (spec D3b): computed once per location, cached to disk ----
    floor_path = f"{paths.out_root}/ks_noise_floor.json"
    if not os.path.exists(floor_path):
        floor = ks_noise_floor(real_train_phys, real_val_phys)
        atomic_json_dump(floor_path, {"per_feature": {str(k): v for k, v in floor.items()},
                                      "mean": float(np.mean(list(floor.values())))})
    floor_info = load_json(floor_path)
    print(f"[{paths.location}] KS noise floor: mean={floor_info['mean']:.4f} | "
          "per-feature=" + str({k: round(v, 3) for k, v in floor_info["per_feature"].items()}))

    ckpt_file = f"{paths.out_root}/checkpoint.json"
    raw_csv = f"{paths.out_root}/ablation_raw.csv"

    done = load_json(ckpt_file) if os.path.exists(ckpt_file) else {}
    records = list(done.values())
    total = len(configs) * len(seeds)
    if done:
        print(f"[{paths.location}] Resuming ablation: {len(done)}/{total} runs complete.")

    for cfg in configs:
        print(f"\n  Config: {cfg.name}")
        for seed in seeds:
            key = f"{cfg.name}_{seed}_{cfg.total_epochs}"
            legacy_key = f"{cfg.name}_{seed}"  # backward-compat with older checkpoints
            if key in done or legacy_key in done:
                print(f"    seed={seed}... cached, skip.")
                continue

            print(f"    seed={seed}...", end=" ", flush=True)
            m = train_experiment(cfg, seed, device, paths, train_tensor, real_train_phys,
                                 real_val_phys, scaler, verbose=True)
            rec = {"config": cfg.name, "seed": seed, **m}
            records.append(rec)
            done[key] = rec
            atomic_json_dump(ckpt_file, done)
            pd.DataFrame(records).to_csv(raw_csv, index=False)
            print(f"    KSstat={m['ks_stat']:.4f}  KSpass={m['ks_pass_rate']:.3f}  "
                  f"tauMAE={m['kendall_mae']:.4f}  ACF={m['acf_mae']:.4f}  "
                  f"Phys={m['phys_viol_rate']:.5f}")

    df = pd.DataFrame(records)
    df.to_csv(raw_csv, index=False)
    print(f"\n[{paths.location}] Ablation saved: {raw_csv}")
    return df


SENS_EPOCHS = 50
SENS_WARMUP = 5
SENS_SEEDS = (42, 123, 456, 789, 2024)


def run_sensitivity(device: str, paths: LocationPaths, train_tensor, real_train_phys: np.ndarray,
                     real_val_phys: np.ndarray, scaler) -> pd.DataFrame:
    """Multi-seed sensitivity sweep over noise dimension, training-set size,
    and epoch count, with resume + de-duplication (the noise_dim=default arm
    is byte-identical to a full-size, SENS_EPOCHS-length run, so its metrics
    are copied rather than recomputed)."""
    sens_csv = f"{paths.out_root}/sensitivity_multiseed.csv"
    df_prev = pd.read_csv(sens_csv) if os.path.exists(sens_csv) else pd.DataFrame()
    all_sens = df_prev.to_dict("records")
    done_keys = {(int(r["seed"]), r["param"], str(r["value"])) for r in all_sens}

    def record(seed, param, value, m):
        all_sens.append({"seed": seed, "param": param, "value": value, **m})
        done_keys.add((seed, param, str(value)))
        pd.DataFrame(all_sens).to_csv(sens_csv, index=False)

    def run(seed, param, value, cfg):
        key = (seed, param, str(value))
        if key in done_keys:
            rec = next(r for r in all_sens if (int(r["seed"]), r["param"], str(r["value"])) == key)
            print(f"    {param}={value}: cached, skip.")
            return {k: rec[k] for k in METRIC_COLS}
        print(f"    {param}={value}...", end=" ", flush=True)
        m = train_experiment(cfg, seed, device, paths, train_tensor, real_train_phys,
                             real_val_phys, scaler, verbose=False)
        record(seed, param, value, m)
        print(f"KS={m['ks_pass_rate']:.3f} tauMAE={m['kendall_mae']:.4f}")
        return m

    full_n = len(train_tensor)
    for seed in SENS_SEEDS:
        print(f"\n  === Seed {seed} ===")

        # noise_dim == C.latent_dim is identical to the full-size/SENS_EPOCHS
        # run below -> train once, copy metrics into the other two sweeps.
        for nd in [32, 64, 128, 256]:
            cfg = ExperimentConfig(name=f"noise_{nd}", noise_dim=nd,
                                   total_epochs=SENS_EPOCHS, warmup_epochs=SENS_WARMUP)
            m = run(seed, "noise_dim", nd, cfg)
            if nd == C.latent_dim and m is not None:
                if (seed, "n_samples", str(full_n)) not in done_keys:
                    record(seed, "n_samples", full_n, m)
                if (seed, "total_epochs", str(SENS_EPOCHS)) not in done_keys:
                    record(seed, "total_epochs", SENS_EPOCHS, m)

        for ns in [500, 1000, 2000]:
            cfg = ExperimentConfig(name=f"nsamples_{ns}", n_samples=ns,
                                   total_epochs=SENS_EPOCHS, warmup_epochs=SENS_WARMUP)
            run(seed, "n_samples", ns, cfg)

        for ep in [e for e in [30, 50, 75, 100, 150] if e != SENS_EPOCHS]:
            wu = max(5, ep // 10)
            cfg = ExperimentConfig(name=f"epochs_{ep}", total_epochs=ep, warmup_epochs=wu)
            run(seed, "total_epochs", ep, cfg)

    df = pd.DataFrame(all_sens)
    print(f"\n[{paths.location}] Sensitivity saved: {sens_csv} ({len(df)} rows, {df.seed.nunique()} seeds)")
    return df


def calibration_ablation(paths: LocationPaths, df_ablation: pd.DataFrame):
    """Compare `full_model` (with calibration) against `no_calibration`,
    using results already in `df_ablation` — no retraining needed."""
    from .metrics import COL_LABELS

    cal_configs = ["full_model", "no_calibration"]
    cal_labels = {"full_model": "with_calibration", "no_calibration": "without_calibration"}

    print(f"\n[{paths.location}] CALIBRATION ABLATION")
    print("=" * 60)
    for cfg_name in cal_configs:
        grp = df_ablation[df_ablation.config == cfg_name]
        print(f"\n{cal_labels[cfg_name]}")
        for col, label in zip(METRIC_COLS, COL_LABELS):
            m, s = grp[col].mean(), grp[col].std()
            lo, hi = bootstrap_ci(grp[col].values)
            print(f"  {label:<28} {m:.4f} +/- {s:.4f}   CI: [{lo:.4f}, {hi:.4f}]")

    print("\nDELTA (without - with calibration):")
    print("=" * 60)
    g_with = df_ablation[df_ablation.config == "full_model"]
    g_without = df_ablation[df_ablation.config == "no_calibration"]
    for col, label in zip(METRIC_COLS, COL_LABELS):
        delta = g_without[col].mean() - g_with[col].mean()
        worse = ((col in ("kendall_mae", "acf_mae", "phys_viol_rate") and delta > 0)
                 or (col == "ks_pass_rate" and delta < 0))
        print(f"  {label:<28} delta={delta:+.4f}  ({'worse' if worse else 'better'})")

    rows = []
    for cfg_name in cal_configs:
        grp = df_ablation[df_ablation.config == cfg_name]
        row = {"config": cal_labels[cfg_name]}
        for col in METRIC_COLS:
            row[f"{col}_mean"] = grp[col].mean()
            row[f"{col}_std"] = grp[col].std()
        rows.append(row)
    out_path = f"{paths.out_root}/calibration_ablation.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
