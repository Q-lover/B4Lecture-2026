"""Invertible permutations and channel-mixing transforms."""

from __future__ import annotations

import torch
from nf_assignment.flows.transforms import Transform, batch_zeros_like
from torch import nn
from torch.nn import functional as F


class Permute(Transform):
    """Permute features along dimension 1."""

    def __init__(self, num_channels: int, mode: str = "shuffle"):
        """Create a fixed channel permutation for toy or sequence tensors."""

        super().__init__()
        self.num_channels = num_channels
        self.mode = mode
        if mode == "shuffle":
            perm = torch.randperm(num_channels)
            inv_perm = torch.empty_like(perm).scatter_(
                dim=0, index=perm, src=torch.arange(num_channels)
            )
            self.register_buffer("perm", perm)
            self.register_buffer("inv_perm", inv_perm)
        elif mode != "swap":
            raise NotImplementedError(f"Unsupported permutation mode: {mode}")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the channel permutation.

        Args:
            x: Tensor shaped ``[batch, channels]`` or
                ``[batch, channels, frames]``.

        Returns:
            A pair ``(y, log_det)`` where ``y`` has the same axes as ``x`` and
            ``log_det`` is a zero tensor shaped ``[batch]``.
        """

        if self.mode == "shuffle":
            y = x[:, self.perm, ...]
        else:
            x1 = x[:, : self.num_channels // 2, ...]
            x2 = x[:, self.num_channels // 2 :, ...]
            y = torch.cat([x2, x1], dim=1)
        return y, batch_zeros_like(x)

    def inverse(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Undo the channel permutation.

        Args:
            y: Tensor shaped ``[batch, channels]`` or
                ``[batch, channels, frames]``.

        Returns:
            A pair ``(x, log_det)`` where ``x`` has the same axes as ``y`` and
            ``log_det`` is a zero tensor shaped ``[batch]``.
        """

        if self.mode == "shuffle":
            x = y[:, self.inv_perm, ...]
        else:
            y1 = y[:, : (self.num_channels + 1) // 2, ...]
            y2 = y[:, (self.num_channels + 1) // 2 :, ...]
            x = torch.cat([y2, y1], dim=1)
        return x, batch_zeros_like(y)


class InvConvNear(Transform):
    """Glow-TTS near-neighbor invertible 1x1 convolution for sequence features."""

    def __init__(self, channels: int, n_split: int = 4, *, no_jacobian: bool = False):
        """Create the grouped near-neighbor invertible channel mixer."""

        super().__init__()
        if n_split % 2 != 0:
            raise ValueError("n_split must be even.")
        if channels % n_split != 0:
            raise ValueError("channels must be divisible by n_split.")
        self.channels = int(channels)
        self.n_split = int(n_split)
        self.no_jacobian = bool(no_jacobian)
        q, _ = torch.linalg.qr(torch.randn(self.n_split, self.n_split))
        if torch.det(q) < 0:
            q[:, 0] = -q[:, 0]
        self.weight = nn.Parameter(q)

    def _lengths(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Return valid frame counts per sequence.

        Args:
            x: Tensor shaped ``[batch, channels, frames]``.
            mask: Optional mask shaped ``[batch, 1, frames]``.

        Returns:
            Tensor shaped ``[batch]``.
        """

        if mask is None:
            return torch.full((x.shape[0],), x.shape[2], dtype=x.dtype, device=x.device)
        return torch.sum(mask.to(dtype=x.dtype, device=x.device), dim=(1, 2))

    def _mix(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Mix channels with the near-neighbor 1x1 convolution pattern.

        Args:
            x: Tensor shaped ``[batch, channels, frames]``.
            weight: Square mixing matrix shaped ``[n_split, n_split]``.

        Returns:
            Tensor shaped ``[batch, channels, frames]``.
        """

        batch, channels, time = x.shape
        x = x.view(batch, 2, channels // self.n_split, self.n_split // 2, time)
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        x = x.view(batch, self.n_split, channels // self.n_split, time)
        weight = weight.view(self.n_split, self.n_split, 1, 1)
        z = F.conv2d(x, weight)
        z = z.view(batch, 2, self.n_split // 2, channels // self.n_split, time)
        z = z.permute(0, 1, 3, 2, 4).contiguous()
        return z.view(batch, channels, time)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the invertible channel mixer.

        Args:
            x: Tensor shaped ``[batch, channels, frames]``.
            mask: Optional mask shaped ``[batch, 1, frames]``.
            condition: Ignored placeholder for the sequence-transform interface.

        Returns:
            A pair ``(z, log_det)`` where ``z`` has the same axes as ``x`` and
            ``log_det`` is shaped ``[batch]``.
        """

        del condition
        z = self._mix(x, self.weight)
        if mask is not None:
            z = z * mask
        if self.no_jacobian:
            log_det = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        else:
            _, log_abs_det = torch.linalg.slogdet(self.weight)
            log_det = (
                log_abs_det * (self.channels / self.n_split) * self._lengths(x, mask)
            )
        return z, log_det

    def inverse(
        self,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Undo the invertible channel mixer.

        Args:
            z: Tensor shaped ``[batch, channels, frames]``.
            mask: Optional mask shaped ``[batch, 1, frames]``.
            condition: Ignored placeholder for the sequence-transform interface.

        Returns:
            A pair ``(x, log_det)`` where ``x`` has the same axes as ``z`` and
            ``log_det`` is shaped ``[batch]``.
        """

        del condition
        weight_inv = torch.inverse(self.weight.float()).to(dtype=self.weight.dtype)
        x = self._mix(z, weight_inv)
        if mask is not None:
            x = x * mask
        if self.no_jacobian:
            log_det = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        else:
            _, log_abs_det = torch.linalg.slogdet(self.weight)
            log_det = (
                -log_abs_det * (self.channels / self.n_split) * self._lengths(z, mask)
            )
        return x, log_det
