"""Ablation and sensitivity chart generation (matplotlib, saved to `out_root`)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from .ablation import CONFIG_ORDER
from .config import LocationPaths
from .metrics import COL_LABELS, METRIC_COLS

FULL_COLOR = "#2ca02c"
BASE_COLOR = "#1f77b4"


def plot_ablation(paths: LocationPaths, df_ablation: pd.DataFrame, n_seeds: int):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    present = [n for n in CONFIG_ORDER if n in df_ablation.config.values]

    for ax, col, label in zip(axes.ravel(), METRIC_COLS, COL_LABELS):
        means = [df_ablation[df_ablation.config == n][col].mean() for n in present]
        stds = [df_ablation[df_ablation.config == n][col].std() for n in present]
        colors = [FULL_COLOR if n == "full_model" else BASE_COLOR for n in present]

        ax.bar(range(len(present)), means, yerr=stds, color=colors,
               capsize=5, edgecolor="black", linewidth=0.8, alpha=0.85)
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels([n.replace("_", "\n") for n in present], fontsize=8)
        ax.set_title(label, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.axhline(means[0], color="green", linestyle="--", alpha=0.4, linewidth=1.2)

    plt.suptitle(f"StatD2GAN Ablation \u2014 {paths.location.title()} (mean +/- std, {n_seeds} seeds)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{paths.out_root}/ablation_chart.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_sensitivity(paths: LocationPaths, df_sens: pd.DataFrame):
    n_seeds = df_sens["seed"].nunique()
    params = df_sens["param"].unique()

    fig, axes = plt.subplots(len(params), 2, figsize=(12, 4 * len(params)))
    if len(params) == 1:
        axes = axes[None, :]

    for row_axes, param in zip(axes, params):
        agg = (df_sens[df_sens.param == param]
               .groupby("value")[["ks_pass_rate", "kendall_mae"]]
               .agg(["mean", "std"]).reset_index().sort_values("value"))
        for ax, col, label in zip(row_axes, ["ks_pass_rate", "kendall_mae"],
                                   ["KS Pass Rate", "Kendall tau MAE"]):
            ax.errorbar(agg["value"].astype(float), agg[(col, "mean")],
                        yerr=agg[(col, "std")].fillna(0), fmt="o-", color="#1f77b4",
                        linewidth=2, markersize=7, capsize=4)
            ax.set_xlabel(param, fontweight="bold")
            ax.set_ylabel(label)
            ax.set_title(f"{label} vs {param}", fontweight="bold")
            ax.grid(alpha=0.3)

    plt.suptitle(f"Sensitivity Analysis \u2014 {paths.location.title()} (mean +/- std, N={n_seeds} seeds)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{paths.out_root}/sensitivity_chart.png", dpi=150, bbox_inches="tight")
    plt.show()
