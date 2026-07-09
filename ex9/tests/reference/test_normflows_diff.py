from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from nf_assignment.flows.coupling import AffineCouplingBlock
from nf_assignment.flows.distributions import DiagGaussian
from nf_assignment.flows.flow import NormalizingFlow
from nf_assignment.flows.permutation import Permute
from nf_assignment.networks.mlp import MLP

REPOS_ROOT = Path(__file__).resolve().parents[3]
NORMFLOWS_ROOT = REPOS_ROOT / "normalizing-flows"


pytestmark = pytest.mark.skipif(
    not (NORMFLOWS_ROOT / "normflows").is_dir(),
    reason="repos/normalizing-flows is not available",
)


def _import_normflows():
    sys.path.insert(0, str(NORMFLOWS_ROOT))
    import normflows as nf  # noqa: PLC0415

    return nf


def test_affine_coupling_block_matches_normflows_fixed_parameters() -> None:
    nf = _import_normflows()
    torch.manual_seed(10)
    reference_net = nf.nets.MLP([1, 5, 2])
    ours_net = MLP([1, 5, 2])
    ours_net.load_state_dict(reference_net.state_dict())
    reference = nf.flows.AffineCouplingBlock(reference_net)
    ours = AffineCouplingBlock(ours_net)
    x = torch.randn(9, 2)

    ref_y, ref_log_det = reference(x)
    y, log_det = ours(x)
    ref_x, ref_inv_log_det = reference.inverse(ref_y)
    x_reconstructed, inv_log_det = ours.inverse(y)

    assert torch.allclose(y, ref_y)
    assert torch.allclose(log_det, ref_log_det)
    assert torch.allclose(x_reconstructed, ref_x)
    assert torch.allclose(inv_log_det, ref_inv_log_det)


def test_permute_swap_matches_normflows() -> None:
    nf = _import_normflows()
    x = torch.randn(4, 2)
    reference = nf.flows.Permute(2, mode="swap")
    ours = Permute(2, mode="swap")

    ref_y, ref_log_det = reference(x)
    y, log_det = ours(x)
    ref_x, ref_inv_log_det = reference.inverse(ref_y)
    x_reconstructed, inv_log_det = ours.inverse(y)

    assert torch.allclose(y, ref_y)
    assert torch.allclose(log_det, ref_log_det.to(log_det.dtype))
    assert torch.allclose(x_reconstructed, ref_x)
    assert torch.allclose(inv_log_det, ref_inv_log_det.to(inv_log_det.dtype))


def test_diag_gaussian_matches_normflows_log_prob_and_sampling() -> None:
    nf = _import_normflows()
    reference = nf.distributions.base.DiagGaussian(2)
    ours = DiagGaussian(2)
    ours.load_state_dict(reference.state_dict())
    x = torch.randn(8, 2)

    assert torch.allclose(ours.log_prob(x), reference.log_prob(x))

    torch.manual_seed(123)
    ref_samples, ref_log_prob = reference(5)
    torch.manual_seed(123)
    samples, log_prob = ours(5)

    assert torch.allclose(samples, ref_samples)
    assert torch.allclose(log_prob, ref_log_prob)


def test_normalizing_flow_log_prob_matches_normflows() -> None:
    nf = _import_normflows()
    torch.manual_seed(12)
    reference_net = nf.nets.MLP([1, 5, 2])
    ours_net = MLP([1, 5, 2])
    ours_net.load_state_dict(reference_net.state_dict())
    reference = nf.NormalizingFlow(
        nf.distributions.base.DiagGaussian(2),
        [nf.flows.AffineCouplingBlock(reference_net), nf.flows.Permute(2, mode="swap")],
    )
    ours = NormalizingFlow(
        DiagGaussian(2),
        [AffineCouplingBlock(ours_net), Permute(2, mode="swap")],
    )
    ours.base.load_state_dict(reference.q0.state_dict())
    x = torch.randn(10, 2)

    assert torch.allclose(ours.log_prob(x), reference.log_prob(x))
