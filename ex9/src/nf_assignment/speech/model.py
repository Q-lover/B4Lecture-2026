"""WORLD coded spectral envelope conditional-flow builders."""

from __future__ import annotations

import torch
from nf_assignment.flows.coupling import SequenceAffineCoupling
from nf_assignment.flows.normalization import ActNorm
from nf_assignment.flows.permutation import InvConvNear
from nf_assignment.networks.wavenet import WaveNetConditioner
from torch import nn


class ConditionalSequenceFlow(nn.Module):
    """Conditional normalizing flow for ``[batch, channels, time]`` speech features."""

    def __init__(self, transforms: list[nn.Module]):
        """Store sequence transforms in data-to-latent order."""

        super().__init__()
        self.transforms = nn.ModuleList(transforms)

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map data features to latent features and return the log determinant.

        Args:
            x: Tensor shaped ``[batch, coded_sp_channels, frames]``.
            condition: Optional tensor shaped ``[batch, condition_channels,
                frames]``.
            mask: Optional tensor shaped ``[batch, 1, frames]``.

        Returns:
            A pair ``(z, log_det)`` where ``z`` has the same axes as ``x`` and
            ``log_det`` is shaped ``[batch]``.
        """

        log_det = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        for transform in self.transforms:
            x, current_log_det = transform(x, mask=mask, condition=condition)
            log_det = log_det + current_log_det
        return x, log_det

    def inverse(
        self,
        z: torch.Tensor,
        condition: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map latent features back to data features and return the inverse log determinant.

        Args:
            z: Tensor shaped ``[batch, coded_sp_channels, frames]``.
            condition: Optional tensor shaped ``[batch, condition_channels,
                frames]``.
            mask: Optional tensor shaped ``[batch, 1, frames]``.

        Returns:
            A pair ``(x, log_det)`` where ``x`` has the same axes as ``z`` and
            ``log_det`` is shaped ``[batch]``.
        """

        log_det = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        for transform in reversed(self.transforms):
            z, current_log_det = transform.inverse(z, mask=mask, condition=condition)
            log_det = log_det + current_log_det
        return z, log_det


def build_speech_flow(
    *,
    coded_sp_channels: int = 48,
    condition_channels: int = 256,
    hidden_channels: int = 64,
    num_blocks: int = 4,
    num_layers_per_block: int = 4,
    kernel_size: int = 5,
    dilation_rate: int = 2,
    n_split: int = 4,
    dropout: float = 0.0,
    data_dep_init: bool = False,
) -> ConditionalSequenceFlow:
    """Build a compact Glow-TTS-style conditional sequence flow.

    The resulting model consumes ``coded_sp`` tensors shaped
    ``[batch, coded_sp_channels, frames]`` and condition tensors shaped
    ``[batch, condition_channels, frames]``.
    """

    if coded_sp_channels % 2 != 0:
        raise ValueError("coded_sp_channels must be even.")
    transforms: list[nn.Module] = []
    for _ in range(num_blocks):
        transforms.append(ActNorm(coded_sp_channels, data_dep_init=data_dep_init))
        transforms.append(InvConvNear(coded_sp_channels, n_split=n_split))
        conditioner = WaveNetConditioner(
            coded_sp_channels // 2,
            coded_sp_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            num_layers=num_layers_per_block,
            condition_channels=condition_channels,
            dropout=dropout,
            zero_init=True,
        )
        transforms.append(
            SequenceAffineCoupling(
                coded_sp_channels,
                conditioner,
            )
        )
    return ConditionalSequenceFlow(transforms)
