"""Affine coupling flow primitives."""

from __future__ import annotations

import torch
from nf_assignment.flows.transforms import Transform
from torch import nn


class AffineCouplingTransform(Transform):
    """Shared affine coupling transform for toy and sequence flows."""

    def __init__(self, conditioner: nn.Module):
        """Store the neural network that predicts shift and log-scale parameters."""

        super().__init__()
        self.conditioner = conditioner

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply affine coupling in the data-to-latent direction.

        Args:
            x: Input tensor shaped ``[batch, channels]`` for toy data or
                ``[batch, channels, frames]`` for speech features. The channel
                axis is split into identity and transformed halves.
            mask: Optional speech mask shaped ``[batch, 1, frames]`` where one
                marks valid frames and zero marks padding.
            condition: Optional conditioner input shaped ``[batch, cond_channels,
                frames]`` on the same frame grid as ``x``.

        Returns:
            A pair ``(y, log_det)`` where ``y`` has the same axes as ``x`` and
            ``log_det`` is shaped ``[batch]``.
        """

        # TODO
        raise NotImplementedError

    def inverse(
        self,
        y: torch.Tensor,
        mask: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply affine coupling in the latent-to-data direction.

        Args:
            y: Latent tensor shaped ``[batch, channels]`` for toy data or
                ``[batch, channels, frames]`` for speech features.
            mask: Optional speech mask shaped ``[batch, 1, frames]``.
            condition: Optional conditioner input shaped ``[batch, cond_channels,
                frames]``.

        Returns:
            A pair ``(x, log_det)`` where ``x`` has the same axes as ``y`` and
            ``log_det`` is shaped ``[batch]``.
        """

        # TODO
        raise NotImplementedError


class AffineCouplingBlock(Transform):
    """Toy affine coupling layer using the shared coupling transform."""

    def __init__(self, param_map: nn.Module):
        """Wrap a toy conditioner that maps ``[batch, channels / 2]`` to affine params."""

        super().__init__()
        self.transform = AffineCouplingTransform(param_map)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform toy samples from data to latent axes.

        Args:
            x: Tensor shaped ``[batch, 2]`` where axis 0 is sample index and
                axis 1 contains the two toy coordinates.

        Returns:
            A pair ``(z, log_det)`` with ``z`` shaped ``[batch, 2]`` and
            ``log_det`` shaped ``[batch]``.
        """

        return self.transform(x)

    def inverse(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform toy latent samples back to data axes.

        Args:
            y: Tensor shaped ``[batch, 2]`` where axis 0 is sample index and
                axis 1 contains latent coordinates.

        Returns:
            A pair ``(x, log_det)`` with ``x`` shaped ``[batch, 2]`` and
            ``log_det`` shaped ``[batch]``.
        """

        return self.transform.inverse(y)


class SequenceAffineCoupling(Transform):
    """Speech affine coupling layer using the shared coupling transform."""

    def __init__(self, channels: int, conditioner: nn.Module):
        """Create a channel-split speech coupling transform."""

        super().__init__()
        if channels % 2 != 0:
            raise ValueError("channels must be even for channel-split coupling.")
        self.channels = int(channels)
        self.conditioner = conditioner
        self.transform = AffineCouplingTransform(conditioner)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform speech features from data to latent axes.

        Args:
            x: Tensor shaped ``[batch, coded_sp_channels, frames]``.
            mask: Optional mask shaped ``[batch, 1, frames]``.
            condition: Optional condition tensor shaped ``[batch, cond_channels,
                frames]``.

        Returns:
            A pair ``(z, log_det)`` with ``z`` shaped like ``x`` and ``log_det``
            shaped ``[batch]``.
        """

        return self.transform(x, mask=mask, condition=condition)

    def inverse(
        self,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform speech latent features back to data axes.

        Args:
            z: Tensor shaped ``[batch, coded_sp_channels, frames]``.
            mask: Optional mask shaped ``[batch, 1, frames]``.
            condition: Optional condition tensor shaped ``[batch, cond_channels,
                frames]``.

        Returns:
            A pair ``(x, log_det)`` with ``x`` shaped like ``z`` and ``log_det``
            shaped ``[batch]``.
        """

        return self.transform.inverse(z, mask=mask, condition=condition)
