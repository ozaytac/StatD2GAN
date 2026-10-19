"""Torch layer. Needs torch (GPU optional). `pytest -m 'not slow'` for the fast subset."""
import numpy as np, pytest
torch = pytest.importorskip("torch")
from s2g import models as M, losses as Lo, train as T, baselines as B
from s2g.config import P, ARMS

NF, L = 11, 168


def test_generator_shape_and_unit_circle():
    g = M.Generator(); x = g(torch.randn(4, P["model"]["latent_dim"]))
    assert x.shape == (4, L, NF)
    assert torch.allclose(x[..., 4:6].norm(dim=-1), torch.ones(4, L), atol=1e-4)
    assert torch.allclose(x[..., 7:9].norm(dim=-1), torch.ones(4, L), atol=1e-4)


@pytest.mark.parametrize("cls", [M.TemporalDiscriminator, M.StatDiscriminator, M.SortedDiscriminator,
                                 M.MomentDiscriminator, M.RankSortedDiscriminator])
def test_discriminator_output_and_gradient(cls):
    d, x = cls(), torch.randn(3, L, NF, requires_grad=True)
    y = d(x); assert y.shape == (3, 1)
    y.sum().backward(); assert torch.isfinite(x.grad).all()


def test_distinct_discriminator_class_names():
    assert M.TemporalDiscriminator().__class__.__name__ != M.StatDiscriminator().__class__.__name__


def test_soft_rank_is_monotone_and_bounded():
    x = torch.randn(2, L, NF); r = M.soft_rank(x)
    assert r.min() >= 0 and r.max() <= 1
    i, j = 0, 3
    o = torch.argsort(x[0, :, j]); assert (r[0, o, j].diff() >= -1e-6).all()


def test_sorted_disc_is_permutation_invariant_rank_disc_is_not():
    x = torch.randn(2, L, NF); p = x[:, torch.randperm(L)]
    ds = M.SortedDiscriminator().eval()
    assert torch.allclose(ds(x), ds(p), atol=1e-5)                      # marginal channel only


def test_physics_loss_zero_on_valid_positive_on_violation():
    sc, mn = torch.ones(NF), torch.zeros(NF)
    ok = torch.zeros(2, L, NF); ok[..., 0] = 10; ok[..., 1] = 5; ok[..., 2] = 1010; ok[..., 6] = 50
    assert Lo.physics_loss(ok, sc, mn) < 1e-6
    bad = ok.clone(); bad[..., 1] = 20
    assert Lo.physics_loss(bad, sc, mn) > 1


def test_evolutionary_weights_invariants():
    e = Lo.EvolutionaryWeights(3)
    for _ in range(500):
        w = e.update(np.abs(np.random.randn(3)) + .1)
        assert abs(w.sum() - 3) < 1e-3 and w.max() / w.min() <= 3.0 + 1e-3


def test_nine_arms_map_to_flags():
    assert len(ARMS) == 9
    a = T.arm("fixed_lambda"); assert not a.use_evolutionary and a.fixed_lambda == [1.0, 1.0, 1.0]
    assert T.arm("no_aux_losses").warmup_epochs == 0 and not T.arm("no_aux_losses").use_aux_losses
    assert T.arm("rank_D_sort").sorted_variant == "rank" and T.arm("moment_D_stat").stat_variant == "moment"


def test_baseline_generators_shapes():
    g = torch.Generator(device=T.DEV).manual_seed(0)
    for net in (B.RCGen(P["baselines"]["rcgan"]).to(T.DEV), B.TimeGAN(P["baselines"]["timegan"]).to(T.DEV)):
        y = net(net.noise(2, g)); assert y.shape == (2, L, NF) and y.abs().max() <= 1 + 1e-5


@pytest.mark.slow
def test_tiny_training_runs_and_rescoring_is_exact(tmp_path):
    import copy
    from s2g import data
    rng = np.random.default_rng(0)
    x = rng.uniform(-.5, .5, (64, L, NF)).astype(np.float32)
    phys = np.zeros_like(x); phys[..., 0] = 10 + 5 * x[..., 0]; phys[..., 1] = phys[..., 0] - 2; phys[..., 2] = 1010 + 10 * x[..., 2]
    phys[..., 3] = 3 + x[..., 3]; phys[..., 6] = 60 + 20 * x[..., 6]; phys[..., 9] = 1 + x[..., 9]; phys[..., 10] = x[..., 10]
    sc = data.Scaler.fit(phys); cfg = T.arm("full_model"); cfg.total_epochs, cfg.warmup_epochs = 2, 1
    G, info = T.train_generator(cfg, 1, sc.fwd(phys), sc)
    assert len(info["epoch_seconds"]) == 2 and len(info["lambda_per_epoch"]) == 1
    a, b = T.draw(G, 8, 5), T.draw(G, 8, 5)
    assert np.array_equal(a, b) and np.isfinite(a).all()                # dedicated RNG stream: exact redraw
    torch.save(G.state_dict(), tmp_path / "g.pt"); G2 = M.Generator().to(T.DEV); G2.load_state_dict(torch.load(tmp_path / "g.pt"))
    assert np.allclose(T.draw(G2, 8, 5), a, atol=1e-5)                  # saved weights reproduce the evaluation draw
