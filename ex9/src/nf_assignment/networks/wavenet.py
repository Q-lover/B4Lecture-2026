"""Glow-TTS style dilated Conv1d conditioners."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils import weight_norm


def fused_add_tanh_sigmoid_multiply(
    input_a: torch.Tensor,
    input_b: torch.Tensor,
    channels: int,
) -> torch.Tensor:
    """Apply the WaveNet gated activation.

    Args:
        input_a: Tensor shaped ``[batch, 2 * channels, frames]`` from a
            dilated convolution.
        input_b: Tensor shaped ``[batch, 2 * channels, frames]`` from the
            optional condition projection.
        channels: Number of hidden channels in each gate half.

    Returns:
        Tensor shaped ``[batch, channels, frames]``.
    """

    acts = input_a + input_b
    tanh_act = torch.tanh(acts[:, :channels, :])
    sigmoid_act = torch.sigmoid(acts[:, channels:, :])
    return tanh_act * sigmoid_act


class WaveNetStack(nn.Module):
    """Dilated Conv1d residual stack used inside sequence coupling conditioners."""

    def __init__(
        self,
        *,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        num_layers: int,
        condition_channels: int = 0,
        dropout: float = 0.0,
    ):
        """Create a masked residual Conv1d stack.

        Args:
            hidden_channels: Number of channels in the residual stream.
            kernel_size: Odd Conv1d kernel size along the frame axis.
            dilation_rate: Exponential dilation base.
            num_layers: Number of residual layers.
            condition_channels: Optional condition feature channels.
            dropout: Dropout probability inside residual layers.
        """

        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd.")
        if hidden_channels % 2 != 0:
            raise ValueError("hidden_channels must be even.")
        self.hidden_channels = int(hidden_channels)
        self.kernel_size = int(kernel_size)
        self.dilation_rate = int(dilation_rate)
        self.num_layers = int(num_layers)
        self.condition_channels = int(condition_channels)
        self.dropout = float(dropout)

        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        self.drop = nn.Dropout(dropout)
        if condition_channels:
            self.cond_layer = weight_norm(
                nn.Conv1d(condition_channels, 2 * hidden_channels * num_layers, 1)
            )

        for layer_index in range(num_layers):
            dilation = dilation_rate**layer_index
            padding = (kernel_size * dilation - dilation) // 2
            self.in_layers.append(
                weight_norm(
                    nn.Conv1d(
                        hidden_channels,
                        2 * hidden_channels,
                        kernel_size,
                        dilation=dilation,
                        padding=padding,
                    )
                )
            )
            if layer_index < num_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels
            self.res_skip_layers.append(
                weight_norm(nn.Conv1d(hidden_channels, res_skip_channels, 1))
            )

    def forward(
        self,
        x: torch.Tensor,
        *,
        condition: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the WaveNet residual stack.

        Args:
            x: Tensor shaped ``[batch, hidden_channels, frames]``.
            condition: Optional tensor shaped ``[batch, condition_channels,
                frames]`` on the same frame grid as ``x``.
            mask: Optional tensor shaped ``[batch, 1, frames]`` marking valid
                frames.

        Returns:
            Tensor shaped ``[batch, hidden_channels, frames]``.
        """

        if mask is None:
            mask = torch.ones(x.shape[0], 1, x.shape[2], dtype=x.dtype, device=x.device)
        output = torch.zeros_like(x)
        encoded_condition = None
        if condition is not None:
            if not self.condition_channels:
                raise ValueError(
                    "condition was provided but condition_channels is zero."
                )
            encoded_condition = self.cond_layer(condition)

        for layer_index in range(self.num_layers):
            acts = self.drop(self.in_layers[layer_index](x))
            if encoded_condition is None:
                condition_slice = torch.zeros_like(acts)
            else:
                offset = layer_index * 2 * self.hidden_channels
                condition_slice = encoded_condition[
                    :, offset : offset + 2 * self.hidden_channels
                ]
            acts = fused_add_tanh_sigmoid_multiply(
                acts,
                condition_slice,
                self.hidden_channels,
            )
            res_skip = self.res_skip_layers[layer_index](acts)
            if layer_index < self.num_layers - 1:
                x = (x + res_skip[:, : self.hidden_channels, :]) * mask
                output = output + res_skip[:, self.hidden_channels :, :]
            else:
                output = output + res_skip
        return output * mask


class WaveNetConditioner(nn.Module):
    """Conditioner that maps a coupling half plus optional condition to affine params."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        num_layers: int,
        condition_channels: int = 0,
        dropout: float = 0.0,
        zero_init: bool = True,
    ):
        """Create a conditioner that predicts affine coupling parameters.

        Args:
            in_channels: Channels in the identity half of the coupling input.
            out_channels: Output channels, typically twice the transformed half
                channels for shift and log-scale.
            hidden_channels: Width of the WaveNet residual stream.
            kernel_size: Odd Conv1d kernel size along frames.
            dilation_rate: Exponential dilation base.
            num_layers: Number of residual layers.
            condition_channels: Channels in the external condition tensor.
            dropout: Dropout probability inside the WaveNet stack.
            zero_init: Whether to zero-initialize the final projection.
        """

        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.hidden_channels = int(hidden_channels)
        self.condition_channels = int(condition_channels)

        self.start = weight_norm(nn.Conv1d(in_channels, hidden_channels, 1))
        self.wn = WaveNetStack(
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            num_layers=num_layers,
            condition_channels=condition_channels,
            dropout=dropout,
        )
        self.end = nn.Conv1d(hidden_channels, out_channels, 1)
        if zero_init:
            self.end.weight.data.zero_()
            self.end.bias.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        *,
        condition: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict shift and log-scale channels for a sequence coupling layer.

        Args:
            x: Coupling identity half shaped ``[batch, in_channels, frames]``.
            condition: Optional tensor shaped ``[batch, condition_channels,
                frames]``.
            mask: Optional tensor shaped ``[batch, 1, frames]``.

        Returns:
            Tensor shaped ``[batch, out_channels, frames]``.
        """

        if mask is None:
            mask = torch.ones(x.shape[0], 1, x.shape[2], dtype=x.dtype, device=x.device)
        hidden = self.start(x) * mask
        hidden = self.wn(hidden, condition=condition, mask=mask)
        return self.end(hidden) * mask
