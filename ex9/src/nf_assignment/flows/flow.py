"""NormalizingFlow container that combines a base distribution and transforms."""

from __future__ import annotations

import torch
from nf_assignment.flows.transforms import Transform, batch_zeros_like
from torch import nn


class NormalizingFlow(nn.Module):
    """Combine a base distribution and a sequence of invertible transforms."""

    def __init__(
        self,
        base: nn.Module,
        transforms: list[Transform],
        target: nn.Module | None = None,
    ):
        """Create a flow with an optional target distribution for diagnostics."""

        super().__init__()
        self.base = base
        self.transforms = nn.ModuleList(transforms)
        self.target = target

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Map latent samples to data samples.

        Args:
            z: Tensor shaped ``[batch, *event_shape]``.

        Returns:
            Tensor shaped ``[batch, *event_shape]`` in data space.
        """

        for transform in self.transforms:
            z, _ = transform(z)
        return z

    def forward_and_log_det(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map latent samples to data samples and accumulate log determinants.

        Args:
            z: Tensor shaped ``[batch, *event_shape]``.

        Returns:
            A pair ``(x, log_det)`` where ``x`` has the same axes as ``z`` and
            ``log_det`` is shaped ``[batch]``.
        """

        log_det = batch_zeros_like(z)
        for transform in self.transforms:
            z, current_log_det = transform(z)
            log_det = log_det + current_log_det
        return z, log_det

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Map data samples to latent samples.

        Args:
            x: Tensor shaped ``[batch, *event_shape]``.

        Returns:
            Tensor shaped ``[batch, *event_shape]`` in base-distribution space.
        """

        for transform in reversed(self.transforms):
            x, _ = transform.inverse(x)
        return x

    def inverse_and_log_det(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map data samples to latent samples and accumulate log determinants.

        Args:
            x: Tensor shaped ``[batch, *event_shape]``.

        Returns:
            A pair ``(z, log_det)`` where ``z`` has the same axes as ``x`` and
            ``log_det`` is shaped ``[batch]``.
        """

        log_det = batch_zeros_like(x)
        for transform in reversed(self.transforms):
            x, current_log_det = transform.inverse(x)
            log_det = log_det + current_log_det
        return x, log_det

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate model log probability for data samples.

        Args:
            x: Tensor shaped ``[batch, *event_shape]``.

        Returns:
            Tensor shaped ``[batch]``.
        """

        z, log_det = self.inverse_and_log_det(x)
        return self.base.log_prob(z) + log_det

    def sample(self, num_samples: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample from the flow in data space.

        Returns:
            A pair ``(x, log_prob)`` where ``x`` is shaped
            ``[num_samples, *event_shape]`` and ``log_prob`` is shaped
            ``[num_samples]``.
        """

        z, log_prob = self.base(num_samples)
        for transform in self.transforms:
            z, log_det = transform(z)
            log_prob = log_prob - log_det
        return z, log_prob

    def forward_kld(self, x: torch.Tensor) -> torch.Tensor:
        """Return the maximum-likelihood objective on data samples.

        Args:
            x: Tensor shaped ``[batch, *event_shape]``.

        Returns:
            Scalar tensor containing the mean negative log likelihood.
        """

        return -torch.mean(self.log_prob(x))

    def reverse_kld(self, num_samples: int = 1, beta: float = 1.0) -> torch.Tensor:
        """Return a reverse-KL diagnostic when a target density is attached.

        Returns:
            Scalar tensor estimated from ``num_samples`` flow samples shaped
            ``[num_samples, *event_shape]``.
        """

        if self.target is None:
            raise RuntimeError("reverse_kld requires a target distribution.")
        z, log_q = self.sample(num_samples)
        log_p = self.target.log_prob(z)
        return torch.mean(log_q) - beta * torch.mean(log_p)
