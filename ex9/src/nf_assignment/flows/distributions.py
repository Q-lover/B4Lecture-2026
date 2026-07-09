"""Base distributions used by normalizing flows."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn


class DiagGaussian(nn.Module):
    """Multivariate Gaussian with diagonal covariance."""

    def __init__(self, shape: int | Sequence[int], trainable: bool = True):
        """Create a Gaussian event distribution.

        Args:
            shape: Event shape excluding the sample axis. For toy data this is
                ``2`` or ``(2,)``; for sequence tensors it can include channels
                and frames.
            trainable: Whether mean and log-scale are learnable parameters.
        """

        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = tuple(shape)
        self.n_dim = len(self.shape)
        self.event_size = math.prod(self.shape)
        loc = torch.zeros(1, *self.shape)
        log_scale = torch.zeros(1, *self.shape)
        if trainable:
            self.loc = nn.Parameter(loc)
            self.log_scale = nn.Parameter(log_scale)
        else:
            self.register_buffer("loc", loc)
            self.register_buffer("log_scale", log_scale)
        self.temperature: float | None = None

    def _current_log_scale(self) -> torch.Tensor:
        """Return the broadcastable log-scale tensor after temperature adjustment.

        Returns:
            Tensor shaped ``[1, *event_shape]``.
        """

        if self.temperature is None:
            return self.log_scale
        return self.log_scale + math.log(self.temperature)

    def forward(self, num_samples: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw samples and return their log probabilities.

        Args:
            num_samples: Number of independent samples on the leading axis.

        Returns:
            A pair ``(z, log_prob)`` where ``z`` is shaped
            ``[num_samples, *event_shape]`` and ``log_prob`` is shaped
            ``[num_samples]``.
        """

        eps = torch.randn(
            (num_samples, *self.shape), dtype=self.loc.dtype, device=self.loc.device
        )
        log_scale = self._current_log_scale()
        z = self.loc + torch.exp(log_scale) * eps
        log_prob = -0.5 * self.event_size * math.log(2.0 * math.pi) - torch.sum(
            log_scale + 0.5 * eps.pow(2), dim=list(range(1, self.n_dim + 1))
        )
        return z, log_prob

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """Evaluate the Gaussian log probability.

        Args:
            z: Tensor shaped ``[batch, *event_shape]``.

        Returns:
            Tensor shaped ``[batch]`` with one log probability per sample.
        """

        log_scale = self._current_log_scale()
        normalized = (z - self.loc) / torch.exp(log_scale)
        return -0.5 * self.event_size * math.log(2.0 * math.pi) - torch.sum(
            log_scale + 0.5 * normalized.pow(2), dim=list(range(1, self.n_dim + 1))
        )

    def sample(self, num_samples: int = 1) -> torch.Tensor:
        """Draw samples without returning log probabilities.

        Returns:
            Tensor shaped ``[num_samples, *event_shape]``.
        """

        samples, _ = self.forward(num_samples)
        return samples
