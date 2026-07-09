import torch
from nf_assignment.flows.coupling import AffineCouplingBlock
from nf_assignment.flows.distributions import DiagGaussian
from nf_assignment.flows.flow import NormalizingFlow
from torch import nn


class ConstantAffineParams(nn.Module):
    def __init__(self, shift: float, log_scale: float):
        super().__init__()
        self.shift = shift
        self.log_scale = log_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shift = torch.full_like(x, self.shift)
        log_scale = torch.full_like(x, self.log_scale)
        return torch.cat([shift, log_scale], dim=1)


def test_affine_coupling_block_logdet_matches_constant_scale() -> None:
    block = AffineCouplingBlock(ConstantAffineParams(shift=1.5, log_scale=0.25))
    x = torch.tensor([[0.0, 1.0], [2.0, -3.0]])

    y, log_det = block(x)
    x_reconstructed, inv_log_det = block.inverse(y)

    assert torch.allclose(y[:, 0], x[:, 0])
    assert torch.allclose(y[:, 1], x[:, 1] * torch.exp(torch.tensor(0.25)) + 1.5)
    assert torch.allclose(log_det, torch.full((2,), 0.25))
    assert torch.allclose(inv_log_det, torch.full((2,), -0.25))
    assert torch.allclose(x_reconstructed, x)


def test_normalizing_flow_log_prob_shape() -> None:
    model = NormalizingFlow(
        DiagGaussian(2), [AffineCouplingBlock(ConstantAffineParams(0.0, 0.1))]
    )
    x = torch.randn(13, 2)

    log_prob = model.log_prob(x)

    assert log_prob.shape == (13,)
    assert torch.isfinite(log_prob).all()
