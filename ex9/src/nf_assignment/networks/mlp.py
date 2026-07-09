"""MLP conditioners for low-dimensional toy flows."""

from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    """Multilayer perceptron matching the normflows toy conditioner contract."""

    def __init__(
        self,
        layers: list[int],
        *,
        leaky: float = 0.0,
        init_zeros: bool = False,
        dropout: float | None = None,
    ):
        """Build a feed-forward conditioner.

        Args:
            layers: Widths including input and output sizes. For the toy affine
                coupling this is typically ``[1, hidden..., 2]``.
            leaky: Negative slope for hidden LeakyReLU activations.
            init_zeros: Whether to zero-initialize the final affine layer.
            dropout: Optional dropout probability inserted before the final layer.
        """

        super().__init__()
        if len(layers) < 2:
            raise ValueError("layers must include at least input and output sizes.")
        modules: list[nn.Module] = []
        for idx in range(len(layers) - 2):
            modules.append(nn.Linear(layers[idx], layers[idx + 1]))
            modules.append(nn.LeakyReLU(leaky))
        if dropout is not None:
            modules.append(nn.Dropout(p=dropout))
        modules.append(nn.Linear(layers[-2], layers[-1]))
        if init_zeros:
            nn.init.zeros_(modules[-1].weight)
            nn.init.zeros_(modules[-1].bias)
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map toy coupling identity coordinates to affine parameters.

        Args:
            x: Tensor shaped ``[batch, in_features]`` where axis 0 indexes
                samples and axis 1 indexes input coordinates.

        Returns:
            Tensor shaped ``[batch, out_features]``.
        """

        return self.net(x)
