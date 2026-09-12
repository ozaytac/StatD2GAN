"""Loading preprocessed sequence data and inverse-scaling to physical units."""
from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np
import torch

from .config import Constants, LocationPaths


@dataclass
class LocationData:
    train_tensor: torch.Tensor
    real_train_phys: np.ndarray
    real_val_phys: np.ndarray
    scaler: object


def denorm(data_norm: np.ndarray, scaler, scalable_idx) -> np.ndarray:
    """Inverse-transform the scalable channels of a [-1, 1]-scaled array
    back to physical units. Cyclical channels are left untouched."""
    out = data_norm.copy()
    flat = out.reshape(-1, out.shape[-1])
    flat[:, scalable_idx] = scaler.inverse_transform(flat[:, scalable_idx])
    return flat.reshape(out.shape)


def load_location(paths: LocationPaths, C: Constants) -> LocationData:
    """Load a location's train/val split.

    Held-out protocol (spec A1): calibration is fit on the train split;
    every reported metric is measured against `real_val_phys`.
    """
    train_raw = np.load(paths.data_path)
    val_raw = np.load(paths.data_path_val)
    train_tensor = torch.from_numpy(train_raw).float()

    with open(paths.scaler_path, "rb") as fh:
        sd = pickle.load(fh)
    scaler = sd["scaler"] if isinstance(sd, dict) else sd

    real_train_phys = denorm(train_raw, scaler, C.scalable_idx)
    real_val_phys = denorm(val_raw, scaler, C.scalable_idx)
    print(f"[{paths.location}] Train: {tuple(train_raw.shape)} | Val: {tuple(val_raw.shape)} | "
          f"Temp mean (train): {real_train_phys[..., 0].mean():.1f} C | "
          f"MSL mean (train): {real_train_phys[..., 2].mean():.1f} hPa")
    return LocationData(train_tensor, real_train_phys, real_val_phys, scaler)
