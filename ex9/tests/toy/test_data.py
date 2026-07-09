import pytest
import torch
from nf_assignment.toy.data import TwoMoons, make_toy_distribution


def test_two_moons_samples_are_sklearn_style_interleaving_arcs() -> None:
    torch.manual_seed(0)
    target = TwoMoons(noise=0.03)

    samples = target.sample(4096)

    assert samples.shape == (4096, 2)
    assert samples[:, 0].min() > -2.50
    assert samples[:, 0].max() < 2.50
    assert samples[:, 1].min() > -1.30
    assert samples[:, 1].max() < 1.30
    assert (samples[:, 1] > 0.5).any()
    assert (samples[:, 1] < 0.0).any()


def test_two_moons_log_prob_prefers_arc_centers() -> None:
    target = TwoMoons(noise=0.08)
    on_arc = torch.tensor(
        [
            [0.0, 1.0],
            [1.0, -0.5],
        ],
        dtype=torch.float32,
    )
    off_arc = torch.tensor(
        [
            [-1.0, -0.6],
            [2.0, 1.2],
        ],
        dtype=torch.float32,
    )

    assert torch.isfinite(target.log_prob(on_arc)).all()
    assert torch.mean(target.log_prob(on_arc)) > torch.mean(target.log_prob(off_arc))


def test_make_toy_distribution_passes_two_moons_noise() -> None:
    target = make_toy_distribution("moons", noise=0.14)

    assert isinstance(target, TwoMoons)
    assert target.noise == pytest.approx(0.14)
