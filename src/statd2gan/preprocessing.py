"""CSV -> normalized `.npy` sequence conversion, per location.

Implements the held-out evaluation protocol (spec A2/A3): the final
`Constants.val_years` full calendar years become the validation split, and
an `Constants.embargo_hours` gap is removed from the end of the training
data so no windowed sequence straddles the train/val boundary.
"""
from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .config import Constants, LocationPaths

SEQUENCE_LENGTH = 168
STEP = 24

SELECTED_FEATURES = [
    "temperature", "dew_point", "msl",
    "wind_speed", "wind_dir_sin", "wind_dir_cos",
    "humidity", "hour_sin", "hour_cos",
    "vpd", "pressure_tendency",
]
CYCLICAL_IDX = [4, 5, 7, 8]  # already scaled to [-1, 1] by construction


def rh_from_dewpoint(t: np.ndarray, td: np.ndarray) -> np.ndarray:
    """Magnus-Tetens relative humidity (%) from temperature and dew point (deg C)."""
    a, b = 17.625, 243.04
    return (100 * np.exp(a * td / (b + td) - a * t / (b + t))).clip(0, 100)


def compute_vpd(t: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """Vapour pressure deficit (kPa) from temperature (deg C) and RH (%)."""
    es = 0.611 * np.exp(17.27 * t / (t + 237.3))
    return np.maximum(0, es * (1 - rh / 100))


def _make_sequences(data: pd.DataFrame, seq_len: int, step: int) -> np.ndarray:
    arr = data.values.astype(np.float32)
    return np.stack([arr[i:i + seq_len] for i in range(0, len(arr) - seq_len + 1, step)])


def preprocess_location(location: str, paths: LocationPaths, base_root: str, C: Constants) -> bool:
    """Build scaled `.npy` train/val arrays + scaler for one location.

    Returns True once the arrays exist (freshly built or already cached),
    False if the source CSV is missing (the caller should skip and move on
    to the next location rather than failing the whole run).
    """
    csv_path = f"{base_root}/{location}.csv"
    if os.path.exists(paths.data_path) and os.path.exists(paths.scaler_path):
        print(f"[{location}] .npy + scaler already present, skipping preprocessing.")
        return True
    if not os.path.exists(csv_path):
        print(f"[{location}] WARNING: {csv_path} not found - skipping location.")
        return False

    df = pd.read_csv(csv_path, parse_dates=["valid_time"])
    df = df.sort_values("valid_time").reset_index(drop=True)
    print(f"[{location}] Loaded: {df.shape} | {df['valid_time'].min()} -> {df['valid_time'].max()}")

    # ---- ERA5 unit conversions ----
    df["temperature"] = df["t2m"] - 273.15  # K -> C
    df["dew_point"] = df["d2m"] - 273.15    # K -> C
    df["msl"] = df["msl"] / 100             # Pa -> hPa

    # ---- wind: u/v components -> speed + cyclical direction ----
    df["wind_speed"] = np.sqrt(df["u10"] ** 2 + df["v10"] ** 2)
    wind_dir_rad = np.arctan2(df["u10"], df["v10"])
    df["wind_dir_sin"] = np.sin(wind_dir_rad)
    df["wind_dir_cos"] = np.cos(wind_dir_rad)
    mag = np.sqrt(df["wind_dir_sin"] ** 2 + df["wind_dir_cos"] ** 2)
    assert (np.abs(mag - 1.0) < 1e-5).all(), "Wind circle violation!"

    # ---- hour-of-day cyclical encoding ----
    df["hour_sin"] = np.sin(2 * np.pi * df["valid_time"].dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["valid_time"].dt.hour / 24)

    # ---- humidity + derived physics features ----
    df["humidity"] = rh_from_dewpoint(df["temperature"], df["dew_point"])
    df["vpd"] = compute_vpd(df["temperature"], df["humidity"])
    df["pressure_tendency"] = df["msl"].diff(3).fillna(0)

    df[SELECTED_FEATURES] = df[SELECTED_FEATURES].ffill().bfill()
    assert df[SELECTED_FEATURES].isnull().sum().sum() == 0, "Missing values remain!"

    # ---- held-out split (spec A2): last C.val_years full calendar years -> val ----
    years = df["valid_time"].dt.year
    val_years = sorted(years.unique())[-C.val_years:]
    val_mask = years.isin(val_years)
    val_start = df.loc[val_mask, "valid_time"].min()

    # ---- embargo (spec A3): drop C.embargo_hours before the val boundary from train ----
    embargo_start = val_start - pd.Timedelta(hours=C.embargo_hours)
    train_df = df[df["valid_time"] < embargo_start]
    val_df = df[val_mask]

    v_months = set(val_df["valid_time"].dt.month.unique())
    assert v_months == set(range(1, 13)), f"Validation split does not cover all months: {sorted(v_months)}"
    print(f"[{location}] Split: train < {embargo_start} | val years {val_years} | embargo {C.embargo_hours}h")

    train_df = train_df[SELECTED_FEATURES]
    val_df = val_df[SELECTED_FEATURES]

    train_seq = _make_sequences(train_df, SEQUENCE_LENGTH, STEP)
    val_seq = _make_sequences(val_df, SEQUENCE_LENGTH, STEP)
    print(f"[{location}] Train sequences: {train_seq.shape} | Val: {val_seq.shape}")

    # ---- selective MinMax scaling to [-1, 1]; cyclical channels already bounded ----
    loc_scaler = MinMaxScaler(feature_range=(-1, 1))
    _n, _t, f = train_seq.shape
    loc_scaler.fit(train_seq.reshape(-1, f)[:, C.scalable_idx])

    def scale(seq: np.ndarray) -> np.ndarray:
        out = seq.copy()
        flat = out.reshape(-1, out.shape[-1])
        flat[:, C.scalable_idx] = loc_scaler.transform(flat[:, C.scalable_idx])
        return flat.reshape(out.shape)

    np.save(paths.data_path, scale(train_seq))
    np.save(paths.data_path_val, scale(val_seq))
    with open(paths.scaler_path, "wb") as fh:
        pickle.dump({"scaler": loc_scaler, "scalable_indices": list(C.scalable_idx),
                    "columns": SELECTED_FEATURES}, fh)
    print(f"[{location}] Saved -> {paths.data_path}")
    return True
