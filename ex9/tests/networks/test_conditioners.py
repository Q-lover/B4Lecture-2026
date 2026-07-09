import torch
from nf_assignment.networks.mlp import MLP


def test_mlp_conditioner_shape_and_zero_init() -> None:
    conditioner = MLP([1, 8, 2], init_zeros=True)
    x = torch.randn(6, 1)

    y = conditioner(x)

    assert y.shape == (6, 2)
    assert torch.allclose(y, torch.zeros_like(y))
