"""Transform interfaces and sequential flow containers."""

from __future__ import annotations

import torch
from torch import nn


def batch_zeros_like(x: torch.Tensor) -> torch.Tensor:
    """Return a per-sample zero log-determinant tensor.

    Args:
        x: Tensor with a leading batch axis, for example ``[batch, channels]``
            or ``[batch, channels, frames]``.

    Returns:
        Tensor shaped ``[batch]``.
    """

    return torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)


class Transform(nn.Module):
    """Base class for invertible transforms."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map data to latent coordinates.

        Args:
            x: Tensor with leading batch axis and transform-specific event axes.

        Returns:
            A pair ``(z, log_det)`` where ``z`` has the same event axes as ``x``
            and ``log_det`` is shaped ``[batch]``.
        """

        raise NotImplementedError

    def inverse(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map latent coordinates back to data.

        Args:
            y: Tensor with leading batch axis and transform-specific event axes.

        Returns:
            A pair ``(x, log_det)`` where ``x`` has the same event axes as ``y``
            and ``log_det`` is shaped ``[batch]``.
        """

        raise NotImplementedError


class FlowSequential(Transform):
    """Apply invertible transforms in sequence."""

    def __init__(self, *transforms: Transform):
        """Store transforms in the order used for the forward map."""

        super().__init__()
        self.transforms = nn.ModuleList(transforms)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply each transform and accumulate log determinants.

        Args:
            x: Tensor shaped ``[batch, *event_shape]``.

        Returns:
            A pair ``(z, log_det)`` with ``z`` shaped like ``x`` and ``log_det``
            shaped ``[batch]``.
        """

        log_det = batch_zeros_like(x)
        for transform in self.transforms:
            x, current_log_det = transform(x)
            log_det = log_det + current_log_det
        return x, log_det

    def inverse(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply inverse transforms in reverse order.

        Args:
            y: Tensor shaped ``[batch, *event_shape]``.

        Returns:
            A pair ``(x, log_det)`` with ``x`` shaped like ``y`` and ``log_det``
            shaped ``[batch]``.
        """

        log_det = batch_zeros_like(y)
        for transform in reversed(self.transforms):
            y, current_log_det = transform.inverse(y)
            log_det = log_det + current_log_det
        return y, log_det
