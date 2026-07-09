"""Toy loss and sample visualization helpers."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

BASE_PLOT_RANGE = (-2.8, 2.8)
TWO_MOONS_XLIM = (-2.8, 2.8)
TWO_MOONS_YLIM = (-1.75, 1.75)
TWO_MOONS_PLOT_RANGE = TWO_MOONS_XLIM


def plot_loss_curve(
    history: list[dict[str, float | int]],
    path: str | Path,
    *,
    ylabel: str = "loss",
) -> None:
    """Plot training loss against step."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [int(row["step"]) for row in history]
    losses = [float(row["loss"]) for row in history]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(steps, losses)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_sample_comparison(
    target_samples: np.ndarray,
    generated_samples: np.ndarray,
    path: str | Path,
    *,
    xlim: tuple[float, float] = TWO_MOONS_XLIM,
    ylim: tuple[float, float] = TWO_MOONS_YLIM,
) -> None:
    """Plot target and generated two-dimensional samples side by side."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    for ax, samples, title in zip(
        axes,
        [target_samples, generated_samples],
        ["target", "generated"],
        strict=True,
    ):
        ax.scatter(samples[:, 0], samples[:, 1], s=7, alpha=0.45)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _evaluate_log_prob_grid(
    log_prob_fn: Callable[[torch.Tensor], torch.Tensor],
    points: torch.Tensor,
    grid_size: int,
    chunk_size: int,
) -> np.ndarray:
    """Evaluate a log-density function over a flattened 2D grid.

    Args:
        log_prob_fn: Callable that accepts a tensor shaped ``[points, 2]`` and
            returns log probabilities shaped ``[points]``.
        points: Tensor shaped ``[grid_size * grid_size, 2]``.
        grid_size: Number of x/y bins in the square grid.
        chunk_size: Maximum number of points evaluated per call.

    Returns:
        NumPy array shaped ``[grid_size, grid_size]``.
    """

    values = []
    with torch.no_grad():
        for start in range(0, len(points), chunk_size):
            values.append(
                log_prob_fn(points[start : start + chunk_size]).detach().cpu()
            )
    log_prob = torch.cat(values, dim=0).reshape(grid_size, grid_size)
    finite = torch.isfinite(log_prob)
    relative_density = torch.zeros_like(log_prob)
    if torch.any(finite):
        max_log_prob = torch.max(log_prob[finite])
        shifted = torch.clamp(log_prob[finite] - max_log_prob, min=-80.0)
        relative_density[finite] = torch.exp(shifted)
    return relative_density.numpy()


def plot_density_heatmaps(
    target_log_prob: Callable[[torch.Tensor], torch.Tensor],
    model_log_prob: Callable[[torch.Tensor], torch.Tensor],
    path: str | Path,
    *,
    device: torch.device | str | None = None,
    grid_size: int = 180,
    chunk_size: int = 65536,
    xlim: tuple[float, float] = TWO_MOONS_XLIM,
    ylim: tuple[float, float] = TWO_MOONS_YLIM,
) -> None:
    """Plot target and model relative densities evaluated on a 2D grid.

    ``target_log_prob`` and ``model_log_prob`` must accept tensors shaped
    ``[points, 2]`` and return log probabilities shaped ``[points]``.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if grid_size < 2:
        raise ValueError("grid_size must be at least 2.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")

    plot_device = torch.device("cpu") if device is None else torch.device(device)
    xs = torch.linspace(xlim[0], xlim[1], grid_size, device=plot_device)
    ys = torch.linspace(ylim[0], ylim[1], grid_size, device=plot_device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    points = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)

    target_density = _evaluate_log_prob_grid(
        target_log_prob, points, grid_size, chunk_size
    )
    model_density = _evaluate_log_prob_grid(
        model_log_prob, points, grid_size, chunk_size
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True)
    extent = [xlim[0], xlim[1], ylim[0], ylim[1]]
    for ax, density, title in zip(
        axes,
        [target_density, model_density],
        ["target relative density", "model relative density"],
        strict=True,
    ):
        image = ax.imshow(
            density,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_title(title)
        ax.set_xlabel("x0")
        ax.set_ylabel("x1")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_warped_base_grid(
    base_log_prob: Callable[[torch.Tensor], torch.Tensor],
    model_log_prob: Callable[[torch.Tensor], torch.Tensor],
    flow_forward: Callable[[torch.Tensor], torch.Tensor],
    path: str | Path,
    *,
    device: torch.device | str | None = None,
    density_grid_size: int = 180,
    base_grid_size: int = 17,
    line_points: int = 256,
    chunk_size: int = 65536,
    data_xlim: tuple[float, float] = TWO_MOONS_XLIM,
    data_ylim: tuple[float, float] = TWO_MOONS_YLIM,
    base_xlim: tuple[float, float] = BASE_PLOT_RANGE,
    base_ylim: tuple[float, float] = BASE_PLOT_RANGE,
) -> None:
    """Plot a base-space grid next to its flow deformation on model density.

    ``base_log_prob`` and ``model_log_prob`` accept tensors shaped
    ``[points, 2]`` and return ``[points]`` log probabilities. ``flow_forward``
    maps base-grid tensors shaped ``[points, 2]`` to data-space tensors with the
    same axes.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if density_grid_size < 2:
        raise ValueError("density_grid_size must be at least 2.")
    if base_grid_size < 2:
        raise ValueError("base_grid_size must be at least 2.")
    if line_points < 2:
        raise ValueError("line_points must be at least 2.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")

    plot_device = torch.device("cpu") if device is None else torch.device(device)

    base_density_x = torch.linspace(
        base_xlim[0], base_xlim[1], density_grid_size, device=plot_device
    )
    base_density_y = torch.linspace(
        base_ylim[0], base_ylim[1], density_grid_size, device=plot_device
    )
    base_yy, base_xx = torch.meshgrid(base_density_y, base_density_x, indexing="ij")
    base_points = torch.stack([base_xx.reshape(-1), base_yy.reshape(-1)], dim=1)
    base_density = _evaluate_log_prob_grid(
        base_log_prob, base_points, density_grid_size, chunk_size
    )

    data_density_x = torch.linspace(
        data_xlim[0], data_xlim[1], density_grid_size, device=plot_device
    )
    data_density_y = torch.linspace(
        data_ylim[0], data_ylim[1], density_grid_size, device=plot_device
    )
    data_yy, data_xx = torch.meshgrid(data_density_y, data_density_x, indexing="ij")
    data_points = torch.stack([data_xx.reshape(-1), data_yy.reshape(-1)], dim=1)
    model_density = _evaluate_log_prob_grid(
        model_log_prob, data_points, density_grid_size, chunk_size
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
    base_ax, data_ax = axes
    base_image = base_ax.imshow(
        base_density,
        origin="lower",
        extent=[base_xlim[0], base_xlim[1], base_ylim[0], base_ylim[1]],
        aspect="equal",
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
    )
    fig.colorbar(base_image, ax=base_ax, fraction=0.046, pad=0.04)

    data_image = data_ax.imshow(
        model_density,
        origin="lower",
        extent=[data_xlim[0], data_xlim[1], data_ylim[0], data_ylim[1]],
        aspect="equal",
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
    )
    fig.colorbar(data_image, ax=data_ax, fraction=0.046, pad=0.04)

    base_x = torch.linspace(base_xlim[0], base_xlim[1], line_points, device=plot_device)
    base_y = torch.linspace(base_ylim[0], base_ylim[1], line_points, device=plot_device)
    constants_x = torch.linspace(
        base_xlim[0], base_xlim[1], base_grid_size, device=plot_device
    )
    constants_y = torch.linspace(
        base_ylim[0], base_ylim[1], base_grid_size, device=plot_device
    )

    with torch.no_grad():
        for value in constants_x:
            line = torch.stack([torch.full_like(base_y, value), base_y], dim=1)
            base_line = line.detach().cpu().numpy()
            warped = flow_forward(line).detach().cpu().numpy()
            base_ax.plot(
                base_line[:, 0],
                base_line[:, 1],
                color="white",
                linewidth=0.65,
                alpha=0.75,
            )
            data_ax.plot(
                warped[:, 0], warped[:, 1], color="white", linewidth=0.65, alpha=0.65
            )
        for value in constants_y:
            line = torch.stack([base_x, torch.full_like(base_x, value)], dim=1)
            base_line = line.detach().cpu().numpy()
            warped = flow_forward(line).detach().cpu().numpy()
            base_ax.plot(
                base_line[:, 0],
                base_line[:, 1],
                color="#35d0ba",
                linewidth=0.65,
                alpha=0.75,
            )
            data_ax.plot(
                warped[:, 0], warped[:, 1], color="#35d0ba", linewidth=0.65, alpha=0.65
            )

    base_ax.set_title("base density with square grid")
    base_ax.set_xlabel("z0")
    base_ax.set_ylabel("z1")
    base_ax.set_xlim(base_xlim)
    base_ax.set_ylim(base_ylim)
    base_ax.grid(False)

    data_ax.set_title("model density with warped grid")
    data_ax.set_xlabel("x0")
    data_ax.set_ylabel("x1")
    data_ax.set_xlim(data_xlim)
    data_ax.set_ylim(data_ylim)
    data_ax.grid(False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
