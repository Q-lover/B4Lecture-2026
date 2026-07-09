"""Toy Real NVP model construction helpers."""

from __future__ import annotations

from nf_assignment.flows.coupling import AffineCouplingBlock
from nf_assignment.flows.distributions import DiagGaussian
from nf_assignment.flows.flow import NormalizingFlow
from nf_assignment.flows.permutation import Permute
from nf_assignment.networks.mlp import MLP
from nf_assignment.toy.data import TwoMoons


def build_realnvp_2d(
    *,
    num_layers: int = 16,
    hidden_dims: tuple[int, ...] = (64, 64),
    init_zeros: bool = True,
    target: TwoMoons | None = None,
) -> NormalizingFlow:
    """Build a compact 2D Real NVP model.

    The model maps tensors shaped ``[batch, 2]`` between toy data coordinates
    and a diagonal Gaussian base distribution.
    """

    transforms = []
    for _ in range(num_layers):
        transforms.append(
            AffineCouplingBlock(MLP([1, *hidden_dims, 2], init_zeros=init_zeros))
        )
        transforms.append(Permute(2, mode="swap"))
    return NormalizingFlow(DiagGaussian(2), transforms, target=target)
