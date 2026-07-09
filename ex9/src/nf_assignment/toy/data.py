"""Toy dataset utilities."""

from __future__ import annotations

import math

import torch
from torch import nn


class TwoMoons(nn.Module):
    """Sklearn-style interleaving two-moons target distribution.

    The two modes follow the same upper/lower half-circle geometry used by
    ``sklearn.datasets.make_moons``.  The analytic log density is approximated
    by a uniform mixture of isotropic Gaussian kernels placed along the two
    arcs; sampling draws an arc point and then adds isotropic noise.
    """

    def __init__(
        self,
        noise: float = 0.18,
        num_log_prob_centers: int = 256,
        scale: float = 1.55,
    ):
        """Create fixed arc centers used for log-density evaluation."""

        super().__init__()
        if noise <= 0.0:
            raise ValueError("noise must be positive.")
        if num_log_prob_centers < 2:
            raise ValueError("num_log_prob_centers must be at least 2.")
        self.n_dims = 2
        self.noise = float(noise)
        self.scale = float(scale)
        theta = torch.linspace(0.0, math.pi, num_log_prob_centers, dtype=torch.float32)
        upper = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
        lower = torch.stack([1.0 - torch.cos(theta), 0.5 - torch.sin(theta)], dim=1)
        centers = torch.cat([upper, lower], dim=0)
        centers = (
            centers - torch.tensor([0.5, 0.25], dtype=torch.float32)
        ) * self.scale
        self.register_buffer("arc_centers", centers)

    def _arc_points(
        self, theta: torch.Tensor, moon_index: torch.Tensor
    ) -> torch.Tensor:
        upper = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
        lower = torch.stack([1.0 - torch.cos(theta), 0.5 - torch.sin(theta)], dim=1)
        centers = torch.where(moon_index[:, None].bool(), lower, upper)
        return (
            centers
            - torch.tensor([0.5, 0.25], dtype=centers.dtype, device=centers.device)
        ) * self.scale

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """Evaluate an approximate normalized two-moons log density.

        Args:
            z: Tensor shaped ``[batch, 2]`` where axis 1 contains ``x0`` and
                ``x1`` toy coordinates.

        Returns:
            Tensor shaped ``[batch]``.
        """

        if z.shape[-1] != self.n_dims:
            raise ValueError(
                f"expected last dimension {self.n_dims}, got {z.shape[-1]}"
            )
        centers = self.arc_centers.to(device=z.device, dtype=z.dtype)
        sigma = torch.as_tensor(self.noise, dtype=z.dtype, device=z.device)
        diff = z[:, None, :] - centers[None, :, :]
        squared_distance = torch.sum(diff.pow(2), dim=-1)
        component_log_prob = (
            -0.5 * squared_distance / sigma.pow(2)
            - math.log(2.0 * math.pi)
            - 2.0 * torch.log(sigma)
        )
        return torch.logsumexp(component_log_prob, dim=1) - math.log(
            float(len(centers))
        )

    def sample(self, num_samples: int = 1) -> torch.Tensor:
        """Draw exactly ``num_samples`` two-moons samples.

        Returns:
            Tensor shaped ``[num_samples, 2]``.
        """

        theta = (
            torch.rand(
                num_samples,
                dtype=self.arc_centers.dtype,
                device=self.arc_centers.device,
            )
            * math.pi
        )
        moon_index = torch.randint(0, 2, (num_samples,), device=self.arc_centers.device)
        centers = self._arc_points(theta, moon_index)
        noise = torch.randn_like(centers) * self.noise
        return centers + noise


class EightGaussians(nn.Module):
    """Eight Gaussian modes arranged on a circle."""

    def __init__(self, radius: float = 2.0, scale: float = 0.08):
        """Create the mode centers and isotropic sampling scale."""

        super().__init__()
        centers = []
        for idx in range(8):
            angle = 2.0 * math.pi * idx / 8
            centers.append([radius * math.sin(angle), radius * math.cos(angle)])
        self.register_buffer("centers", torch.tensor(centers, dtype=torch.float32))
        self.scale = scale

    def sample(self, num_samples: int = 1) -> torch.Tensor:
        """Draw samples from the eight-mode mixture.

        Returns:
            Tensor shaped ``[num_samples, 2]``.
        """

        indices = torch.randint(
            0, len(self.centers), (num_samples,), device=self.centers.device
        )
        noise = torch.randn(
            num_samples, 2, dtype=self.centers.dtype, device=self.centers.device
        )
        return self.centers[indices] + self.scale * noise


def make_toy_distribution(name: str, *, noise: float | None = None) -> nn.Module:
    """Create a toy target distribution by name."""

    normalized = name.lower().replace("-", "_")
    if normalized in {"moons", "two_moons"}:
        return TwoMoons(noise=0.18 if noise is None else noise)
    if normalized in {"8gaussians", "eight_gaussians"}:
        return EightGaussians()
    raise ValueError(f"Unknown toy distribution: {name}")
