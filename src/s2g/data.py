"""Raw ERA5 CSV -> scaled train/held-out sequences (numpy only)."""
import numpy as np, pandas as pd
from .config import P, FEATS, SC_IDX

D = P["data"]
RAW_COLS = ["valid_time", "u10", "v10", "d2m", "t2m", "msl", "sp"]


def rh_from_dewpoint(t, td):
    a, b = 17.625, 243.04
    return (100 * np.exp(a * td / (b + td) - a * t / (b + t))).clip(0, 100)


def vpd(t, rh):
    return np.maximum(0, 0.611 * np.exp(17.27 * t / (t + 237.3)) * (1 - rh / 100))


def features(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.sort_values("valid_time").reset_index(drop=True).copy()
    assert df[RAW_COLS[1:]].notna().all().all(), "missing values in raw ERA5 (no silent fill)"
    assert (df.valid_time.diff().dropna() == pd.Timedelta(hours=1)).all(), "gaps in hourly series"
    f = pd.DataFrame({"valid_time": df.valid_time})
    f["temperature"], f["dew_point"], f["msl"] = df.t2m - 273.15, df.d2m - 273.15, df.msl / 100
    f["wind_speed"] = np.hypot(df.u10, df.v10)
    ang = np.arctan2(df.u10, df.v10)
    f["wind_dir_sin"], f["wind_dir_cos"] = np.sin(ang), np.cos(ang)
    h = df.valid_time.dt.hour
    f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
    f["humidity"] = rh_from_dewpoint(f.temperature, f.dew_point)
    f["vpd"] = vpd(f.temperature, f.humidity)
    f["pressure_tendency"] = f.msl.diff(3).fillna(0)     # backward-looking only
    return f


def split(f: pd.DataFrame, val_years=None, embargo=None):
    """Last `val_years` full calendar years -> held-out; `embargo` hours cut from end of train."""
    val_years = val_years or D["val_years"]
    embargo = D["embargo_hours"] if embargo is None else embargo
    ys = sorted(f.valid_time.dt.year.unique())[-val_years:]
    vm = f.valid_time.dt.year.isin(ys)
    for y in ys:                                          # each held-out year must be complete
        n = (f.valid_time.dt.year == y).sum()
        assert n in (8760, 8784), f"year {y} incomplete ({n} h)"
    v0 = f.loc[vm, "valid_time"].min()
    tr = f[f.valid_time < v0 - pd.Timedelta(hours=embargo)]
    assert set(f[vm].valid_time.dt.month) == set(range(1, 13))
    return tr[FEATS], f.loc[vm, FEATS], ys


def windows(df: pd.DataFrame, L=None, step=None) -> np.ndarray:
    L, step = L or D["seq_len"], step or D["step"]
    a = df.values.astype(np.float32)
    return np.stack([a[i:i + L] for i in range(0, len(a) - L + 1, step)])


class Scaler:
    """Min-max to [-1, 1] on SC_IDX only, fit on train windows."""
    def __init__(self, lo, hi):
        self.lo, self.hi = np.asarray(lo, float), np.asarray(hi, float)
        self.scale = 2 / (self.hi - self.lo)
        self.min = -1 - self.lo * self.scale

    @classmethod
    def fit(cls, tr):
        x = tr.reshape(-1, tr.shape[-1])[:, SC_IDX]
        return cls(x.min(0), x.max(0))

    def fwd(self, x):
        o = x.copy(); o[..., SC_IDX] = x[..., SC_IDX] * self.scale + self.min; return o

    def inv(self, x):
        o = x.copy(); o[..., SC_IDX] = (x[..., SC_IDX] - self.min) / self.scale; return o


def build(raw: pd.DataFrame):
    """-> dict(train_phys, val_phys, scaler, val_years). Physical units; scaling separate."""
    tr, va, ys = split(features(raw))
    trw, vaw = windows(tr), windows(va)
    return dict(train_phys=trw, val_phys=vaw, scaler=Scaler.fit(trw), val_years=[int(y) for y in ys])


def save(path, d):
    s = d["scaler"]
    np.savez_compressed(path, train_phys=d["train_phys"], val_phys=d["val_phys"],
                        lo=s.lo, hi=s.hi, val_years=d["val_years"])


def load(path):
    z = np.load(path)
    return dict(train_phys=z["train_phys"], val_phys=z["val_phys"],
                scaler=Scaler(z["lo"], z["hi"]), val_years=z["val_years"].tolist())
