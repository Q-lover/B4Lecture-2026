import torch
from nf_assignment.flows.coupling import AffineCouplingBlock, AffineCouplingTransform
from nf_assignment.flows.flow import NormalizingFlow
from nf_assignment.flows.permutation import Permute
from nf_assignment.networks.mlp import MLP
from nf_assignment.toy.model import build_realnvp_2d
from torch import nn


class ChunkedSequenceParams(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        *,
        condition: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del condition, mask
        shift = torch.full_like(x, 0.5)
        log_scale = torch.full_like(x, -0.25)
        return torch.cat([shift, log_scale], dim=1)


def test_affine_coupling_block_reconstructs_input() -> None:
    torch.manual_seed(0)
    block = AffineCouplingBlock(MLP([1, 8, 2]))
    x = torch.randn(7, 2)

    y, forward_log_det = block(x)
    x_reconstructed, inverse_log_det = block.inverse(y)

    assert torch.allclose(x_reconstructed, x, atol=1e-6)
    assert torch.allclose(
        forward_log_det + inverse_log_det, torch.zeros_like(forward_log_det)
    )
    assert isinstance(block.transform, AffineCouplingTransform)


def test_shared_affine_coupling_transform_handles_sequence_masks() -> None:
    transform = AffineCouplingTransform(ChunkedSequenceParams())
    x = torch.randn(2, 4, 5)
    mask = torch.tensor([[[1.0, 1.0, 1.0, 0.0, 0.0]], [[1.0, 1.0, 1.0, 1.0, 0.0]]])
    x = x * mask

    y, forward_log_det = transform(x, mask=mask)
    x_reconstructed, inverse_log_det = transform.inverse(y, mask=mask)

    expected_log_det = torch.tensor([-1.5, -2.0])
    assert torch.allclose(x_reconstructed, x, atol=1e-6)
    assert torch.allclose(forward_log_det, expected_log_det)
    assert torch.allclose(
        forward_log_det + inverse_log_det, torch.zeros_like(forward_log_det)
    )


def test_permute_swap_reconstructs_input() -> None:
    transform = Permute(2, mode="swap")
    x = torch.randn(5, 2)

    y, forward_log_det = transform(x)
    x_reconstructed, inverse_log_det = transform.inverse(y)

    assert torch.allclose(x_reconstructed, x)
    assert torch.allclose(forward_log_det, torch.zeros(5))
    assert torch.allclose(inverse_log_det, torch.zeros(5))


def test_normalizing_flow_reconstructs_input() -> None:
    torch.manual_seed(1)
    model = build_realnvp_2d(num_layers=3, hidden_dims=(8,), init_zeros=False)
    z = torch.randn(11, 2)

    x, forward_log_det = model.forward_and_log_det(z)
    z_reconstructed, inverse_log_det = model.inverse_and_log_det(x)

    assert torch.allclose(z_reconstructed, z, atol=1e-5)
    assert torch.allclose(
        forward_log_det + inverse_log_det,
        torch.zeros_like(forward_log_det),
        atol=1e-5,
    )
    assert isinstance(model, NormalizingFlow)
