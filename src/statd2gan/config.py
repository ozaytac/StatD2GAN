"""Global constants and experiment configuration for StatD2GAN (v2 protocol).

Single source of truth for hyperparameters and protocol constants. Physics
thresholds and metric subsample sizes are each defined exactly once here and
imported by both the training loss and the evaluation metric, so the two
can never silently drift apart (see `losses.physics_loss` and
`metrics.compute_metrics`).
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch


@dataclass(frozen=True)
class Constants:
    # ---- storage ----
    base_root: str = os.environ.get("STATD2GAN_DATA_ROOT", "./data")
    protocol: str = "v2"

    # ---- sequence / model shape ----
    batch_size: int = 512
    seq_len: int = 168
    n_features: int = 11
    latent_dim: int = 128
    hidden_dim: int = 512
    lr_g: float = 1e-3
    lr_d: float = 1e-3
    scalable_idx: tuple = (0, 1, 2, 3, 6, 9, 10)

    # ---- held-out split protocol (spec A2 / A3) ----
    val_years: int = 2        # last N full calendar years -> validation
    embargo_hours: int = 168  # gap left at the train/val boundary (= seq_len)

    # ---- physics thresholds: single source for loss AND metric (spec C2) ----
    dew_tol: float = 0.1
    pres_lo: float = 980.0
    pres_hi: float = 1045.0
    hum_lo: float = 0.0
    hum_hi: float = 100.0

    # ---- metric constants (spec D1, D3) ----
    tau_subsample: int = 50_000
    ks_subsample: int = 20_000
    metric_seed: int = 1907

    # ---- training schedule ----
    fast_mode: bool = False   # True -> ~30 epochs, for smoke-testing the pipeline
    warmup_epochs: int = 15
    total_epochs: int = 150
    seeds: tuple = (42, 123, 456, 789, 2024)

    # ---- performance flags ----
    use_compile: bool = False
    gp_freq: int = 5          # gradient penalty every N critic steps (lazy GP)

    @property
    def use_amp(self) -> bool:
        return torch.cuda.is_available()

    def __post_init__(self):
        # frozen=True still allows __post_init__ to set fields via
        # object.__setattr__; this keeps fast_mode a single flag to flip
        # instead of two fields the caller has to remember to set together.
        if self.fast_mode:
            object.__setattr__(self, "warmup_epochs", 5)
            object.__setattr__(self, "total_epochs", 30)


C = Constants()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOCATIONS = ["ankara", "yakutsk", "lhasa", "dubai", "bergen"]


@dataclass
class ExperimentConfig:
    """One ablation arm. Defaults reproduce the full model; every other arm
    in `ablation.ABLATION_CONFIGS` flips exactly one of these off."""
    name: str
    use_D_temporal: bool = True
    use_D_stat: bool = True
    use_D_sorted: bool = True
    use_calibration: bool = True
    use_evolutionary: bool = True
    fixed_lambda: Optional[List[float]] = None
    noise_dim: int = C.latent_dim
    n_samples: Optional[int] = None
    warmup_epochs: int = C.warmup_epochs
    total_epochs: int = C.total_epochs
    gp_freq: int = 0                # 0 -> fall back to C.gp_freq
    stat_variant: str = "lstm"      # 'lstm' | 'moment'        (spec B2)
    sorted_variant: str = "sorted"  # 'sorted' | 'rank'        (spec F3)
    use_aux_losses: bool = True     # cdf + quantile losses    (spec B3/F1)
    gp_on_moment: bool = True       # False -> skip GP for MomentDiscriminator (spec B2c)


@dataclass
class LocationPaths:
    location: str
    root: str
    data_path: str
    data_path_val: str
    scaler_path: str
    out_root: str
    gen_root: str


def make_paths(location: str, base_root: str = C.base_root, protocol: str = C.protocol) -> LocationPaths:
    """Build (and create) every filesystem path derived from a location.

    All v2 outputs carry the protocol tag in their filename so they can
    never be silently mixed up with pre-protocol (v1) caches on disk.
    """
    root = f"{base_root}/{location}"
    data_path = f"{root}/weather_sequences_train_{protocol}.npy"
    paths = LocationPaths(
        location=location,
        root=root,
        data_path=data_path,
        data_path_val=data_path.replace("train", "val"),
        scaler_path=f"{root}/weather_scaler_{protocol}.pkl",
        out_root=f"{root}/ablation_results_{protocol}",
        gen_root=f"{root}/ablation_results_{protocol}/generators",
    )
    os.makedirs(paths.out_root, exist_ok=True)
    os.makedirs(paths.gen_root, exist_ok=True)
    return paths


def set_seed(seed: int) -> None:
    """Seed every RNG touched during training.

    cuDNN benchmark mode is deliberately left off: on newer GPU
    architectures an autotune cache miss forces a re-JIT that costs more
    than the speedup it buys, given how often the shapes change here.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.environ["PYTHONHASHSEED"] = str(seed)
