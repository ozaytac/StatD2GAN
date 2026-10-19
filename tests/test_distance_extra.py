"""Tests for scripts/distance_extra.py. Run with pytest or as `python tests/test_distance_extra.py`.
The torch test uses an untrained (random-weight) generator on CPU and is skipped without torch."""
import importlib.util, json, sys, tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("distance_extra", ROOT / "scripts" / "distance_extra.py")
DX = importlib.util.module_from_spec(spec); spec.loader.exec_module(DX)
from s2g import metrics as M  # noqa: E402

L, NF = 168, 11


def fake_phys(rng, n, shift=0.0, ties=False):
    """Physical-unit windows with the scaled features populated (fake generator / fake data)."""
    x = rng.normal(size=(n, L, NF)).astype(np.float32)
    p = np.zeros_like(x)
    p[..., 0] = 10 + 5 * x[..., 0] + shift; p[..., 1] = p[..., 0] - 3 + x[..., 1]
    p[..., 2] = 1010 + 8 * x[..., 2]; p[..., 3] = np.abs(3 + x[..., 3]); p[..., 6] = 60 + 15 * x[..., 6]
    p[..., 9] = np.abs(1 + x[..., 9]); p[..., 10] = x[..., 10]
    if ties:                                         # heavy ties: rounded values
        p[..., [0, 2, 10]] = np.round(p[..., [0, 2, 10]])
    return p


def test_triangle_inequality_holds_for_fake_generators():
    rng = np.random.default_rng(0)
    for k in range(30):
        train, val = fake_phys(rng, 40), fake_phys(rng, 15, shift=rng.normal(0, 1))
        ev, ca = fake_phys(rng, 30, shift=rng.normal(0, 3), ties=k % 3 == 0), fake_phys(rng, 30, shift=rng.normal(0, 3))
        q = DX.distances(train, val, ev, ca)
        t = DX.triangle(q)
        assert t["ok"], (k, t)
        for f in M.ks_floor(train, val):              # floor is the frozen definition
            assert q["ks_floor"][f] == M.ks_floor(train, val)[f]


def test_gate_reproduces_frozen_pipeline_exactly_and_catches_drift():
    rng = np.random.default_rng(1)
    train, val, ev, ca = fake_phys(rng, 40), fake_phys(rng, 15, 1), fake_phys(rng, 30, 2), fake_phys(rng, 30, 2)
    rec = M.evaluate(train, val, ev, ca, calibrate=True)        # what train.finish stores in the JSON
    q = DX.distances(train, val, ev, ca)
    g = DX.gate(q, rec)
    assert g["ok"] and g["gate_raw_diff"] == 0 and g["gate_cal_diff"] == 0 and g["tie_frac_diff"] == 0
    bad = json.loads(json.dumps(rec)); bad["cal"]["ks_stat"] += 2e-4
    assert not DX.gate(q, bad)["ok"]


def test_untrained_torch_generator_end_to_end_cpu():
    try:
        import torch
    except ImportError:
        try:
            import pytest
        except ImportError:
            return "SKIP (torch not installed)"
        pytest.skip("torch not installed")
    from s2g import data, train as T
    from s2g.models import Generator
    rng = np.random.default_rng(2)
    train, val = fake_phys(rng, 40), fake_phys(rng, 15, 1)
    sc = data.Scaler.fit(train)
    G = Generator().to(T.DEV)                                  # random weights: no training
    tmp = Path(tempfile.mkdtemp()); torch.save(G.state_dict(), tmp / "x_full_model_seed7.pt")
    rec = dict(config="full_model", loc="x", seed=7)
    from s2g.config import P
    E = P["evaluation"]
    ev, ca = sc.inv(T.draw(G, 20, E["eval_draw_seed"] + 7)), sc.inv(T.draw(G, 20, E["calib_draw_seed"] + 7))
    rec.update(M.evaluate(train, val, ev, ca, calibrate=True))
    ev2, ca2 = DX.draws(tmp / "x_full_model_seed7.pt", rec, sc, n_eval=20)
    q = DX.distances(train, val, ev2, ca2)
    assert DX.gate(q, rec)["ok"] and DX.triangle(q)["ok"]


def test_triangle_check_detects_a_corrupted_distance():
    rng = np.random.default_rng(3)
    q = DX.distances(fake_phys(rng, 40), fake_phys(rng, 15, 1), fake_phys(rng, 30, 2), fake_phys(rng, 30, 2))
    f = next(iter(q["d_cal_train"]))
    q["d_cal_train"][f] = abs(q["ks_cal_val"][f] - q["ks_floor"][f]) - 1e-9
    assert not DX.triangle(q)["ok"]


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            r = f(); print(r if isinstance(r, str) else "PASS", n)
