import numpy as np, pandas as pd, pytest
from pathlib import Path
from s2g import data
from s2g.config import ROOT

RAW = ROOT / "data/raw/ankara.csv"


@pytest.fixture(scope="module")
def built():
    return data.build(pd.read_csv(RAW, parse_dates=["valid_time"]))


def test_split_is_last_two_full_years(built):
    assert built["val_years"] == [2023, 2024]


def test_embargo_and_no_overlap():
    raw = pd.read_csv(RAW, parse_dates=["valid_time"])
    f = data.features(raw)
    tr, va, _ = data.split(f)
    t = f.valid_time
    v0 = t[t.dt.year >= 2023].min()
    assert t[tr.index].max() < v0 - pd.Timedelta(hours=168)          # train ends before the embargo
    assert va.index.min() == t[t >= v0].index.min()
    # a window starting at the last train window start must end before the embargo edge
    w = data.windows(tr)
    assert len(tr) - (len(w) - 1) * 24 >= 168


def test_shapes_and_seasons(built):
    assert built["train_phys"].shape[1:] == (168, 11) and built["val_phys"].shape[1:] == (168, 11)
    assert len(built["train_phys"]) > len(built["val_phys"]) > 500


def test_scaler_roundtrip_and_range(built):
    s, x = built["scaler"], built["train_phys"]
    z = s.fwd(x)
    assert np.allclose(s.inv(z), x, atol=1e-3)
    assert z[..., data.SC_IDX].min() >= -1 - 1e-6 and z[..., data.SC_IDX].max() <= 1 + 1e-6


def test_features_physical(built):
    x = built["train_phys"]
    assert (x[..., 6] >= 0).all() and (x[..., 6] <= 100).all()
    assert np.allclose(np.hypot(x[..., 4], x[..., 5]), 1, atol=1e-4)


def test_gap_detection():
    raw = pd.read_csv(RAW, parse_dates=["valid_time"]).drop(index=[10])
    with pytest.raises(AssertionError):
        data.features(raw)
