"""Toy sample generation helpers."""

from __future__ import annotations

import torch
from nf_assignment.flows.flow import NormalizingFlow


@torch.no_grad()
def sample_model(model: NormalizingFlow, num_samples: int) -> torch.Tensor:
    """Generate samples from a normalizing-flow model.

    Returns:
        Tensor shaped ``[num_samples, 2]`` for the assignment toy model.
    """

    samples, _ = model.sample(num_samples)
    return samples
