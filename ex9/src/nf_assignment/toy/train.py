"""Toy training helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
from nf_assignment.flows.flow import NormalizingFlow


def forward_kld_step(
    model: NormalizingFlow,
    optimizer: torch.optim.Optimizer,
    batch: torch.Tensor,
) -> float:
    """Run one maximum-likelihood optimization step on target samples.

    Args:
        model: Flow that consumes toy tensors shaped ``[batch, 2]``.
        optimizer: Optimizer updated by this step.
        batch: Target samples shaped ``[batch, 2]``.

    Returns:
        Scalar Python loss value.
    """

    optimizer.zero_grad(set_to_none=True)
    loss = model.forward_kld(batch)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())


def train_forward_kld(
    model: NormalizingFlow,
    optimizer: torch.optim.Optimizer,
    target: torch.nn.Module,
    *,
    batch_size: int,
    num_steps: int,
    log_every: int = 1,
    progress_callback: Callable[[dict[str, float | int]], None] | None = None,
) -> list[dict[str, float | int]]:
    """Train by maximum likelihood on samples from the target distribution.

    The target distribution must return sample tensors shaped ``[batch, 2]``.
    """

    history: list[dict[str, float | int]] = []
    for step in range(1, num_steps + 1):
        batch = target.sample(batch_size)
        loss = forward_kld_step(model, optimizer, batch)
        if step == 1 or step % log_every == 0 or step == num_steps:
            entry = {"step": step, "loss": loss}
            history.append(entry)
            if progress_callback is not None:
                progress_callback(entry)
    return history


def iter_batches(data: torch.Tensor, batch_size: int) -> Iterable[torch.Tensor]:
    """Yield contiguous mini-batches from an in-memory tensor.

    Args:
        data: Tensor shaped ``[items, *event_shape]``.
        batch_size: Maximum number of items on the leading axis.

    Yields:
        Tensor shaped ``[batch, *event_shape]``.
    """

    for start in range(0, len(data), batch_size):
        yield data[start : start + batch_size]
