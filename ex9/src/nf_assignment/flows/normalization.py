"""Invertible normalization layers such as ActNorm."""

from __future__ import annotations

import torch
from torch import nn


def _sequence_mask(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Return a mask broadcastable to sequence features.

    Args:
        x: Sequence tensor shaped ``[batch, channels, frames]``.
        mask: Optional mask shaped ``[batch, 1, frames]``.

    Returns:
        Tensor shaped ``[batch, 1, frames]`` with the same dtype/device as ``x``.
    """

    if mask is None:
        return torch.ones(x.shape[0], 1, x.shape[2], dtype=x.dtype, device=x.device)
    return mask.to(dtype=x.dtype, device=x.device)


class ActNorm(nn.Module):
    """Glow-style activation normalization for ``[batch, channels, time]`` tensors."""

    def __init__(
        self, channels: int, *, data_dep_init: bool = False, eps: float = 1e-6
    ):
        """Create channel-wise affine normalization parameters."""

        super().__init__()
        self.channels = int(channels)
        self.eps = float(eps)
        self.initialized = not data_dep_init
        self.logs = nn.Parameter(torch.zeros(1, channels, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1))

    def _lengths(self, mask: torch.Tensor) -> torch.Tensor:
        """Count valid frames per sample.

        Args:
            mask: Tensor shaped ``[batch, 1, frames]``.

        Returns:
            Tensor shaped ``[batch]``.
        """

        return torch.sum(mask, dim=(1, 2))

    def initialize(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        """Data-dependently initialize bias and log-scale.

        Args:
            x: Tensor shaped ``[batch, channels, frames]``.
            mask: Optional mask shaped ``[batch, 1, frames]``.
        """

        mask = _sequence_mask(x, mask)
        with torch.no_grad():
            denom = torch.clamp_min(torch.sum(mask, dim=(0, 2)), 1.0)
            mean = torch.sum(x * mask, dim=(0, 2)) / denom
            mean_square = torch.sum(x.square() * mask, dim=(0, 2)) / denom
            variance = torch.clamp_min(mean_square - mean.square(), self.eps)
            std_log = 0.5 * torch.log(variance)
            self.bias.data.copy_((-mean * torch.exp(-std_log)).view_as(self.bias))
            self.logs.data.copy_((-std_log).view_as(self.logs))
        self.initialized = True

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform data to latent space.

        Args:
            x: Tensor shaped ``[batch, channels, frames]``.
            mask: Optional mask shaped ``[batch, 1, frames]``.
            condition: Ignored placeholder for the sequence-transform interface.

        Returns:
            A pair ``(z, log_det)`` where ``z`` has the same axes as ``x`` and
            ``log_det`` is shaped ``[batch]``.
        """

        del condition
        mask = _sequence_mask(x, mask)
        if not self.initialized:
            self.initialize(x, mask)
        z = (self.bias + torch.exp(self.logs) * x) * mask
        log_det = torch.sum(self.logs) * self._lengths(mask)
        return z, log_det

    def inverse(
        self,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform latent samples back to data space.

        Args:
            z: Tensor shaped ``[batch, channels, frames]``.
            mask: Optional mask shaped ``[batch, 1, frames]``.
            condition: Ignored placeholder for the sequence-transform interface.

        Returns:
            A pair ``(x, log_det)`` where ``x`` has the same axes as ``z`` and
            ``log_det`` is shaped ``[batch]``.
        """

        del condition
        mask = _sequence_mask(z, mask)
        x = (z - self.bias) * torch.exp(-self.logs) * mask
        log_det = -torch.sum(self.logs) * self._lengths(mask)
        return x, log_det
