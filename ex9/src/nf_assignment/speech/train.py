"""Speech feature-flow training helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from math import log, pi
from typing import Any

import torch
from torch.utils.data import DataLoader


def resolve_device(requested: str) -> torch.device:
    """Resolve ``auto``/explicit torch device strings."""

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device was requested, but torch.cuda.is_available() is False."
        )
    return device


def build_optimizer(
    model: torch.nn.Module, train_config: dict[str, Any]
) -> torch.optim.Optimizer:
    """Build the speech-flow optimizer from config."""

    optimizer_config = train_config.get("optimizer", {})
    name = str(optimizer_config.get("name", "adam")).lower()
    lr = float(optimizer_config.get("lr", 2e-4))
    weight_decay = float(optimizer_config.get("weight_decay", 0.0))
    if name != "adam":
        raise ValueError(f"Unsupported optimizer: {name}")
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensor values in a speech batch to the requested device.

    Expected tensor entries are ``coded_sp`` shaped
    ``[batch, coded_sp_channels, frames]``, ``condition`` shaped
    ``[batch, condition_channels, frames]``, ``mask`` shaped
    ``[batch, 1, frames]``, and ``lengths`` shaped ``[batch]``.
    """

    moved = dict(batch)
    for key in ("coded_sp", "condition", "lengths", "mask"):
        moved[key] = batch[key].to(device)
    return moved


def crop_batch_segments(
    batch: dict[str, Any],
    *,
    segment_frames: int,
    generator: torch.Generator | None = None,
) -> dict[str, Any]:
    """Randomly crop padded utterance batches to fixed-length training segments.

    Args:
        batch: Batch dictionary containing ``coded_sp`` shaped
            ``[batch, coded_sp_channels, frames]``, ``condition`` shaped
            ``[batch, condition_channels, frames]``, ``mask`` shaped
            ``[batch, 1, frames]``, and ``lengths`` shaped ``[batch]``.
        segment_frames: Number of frames to keep on the time axis. Values less
            than or equal to zero disable cropping.
        generator: Optional PyTorch RNG for crop starts.

    Returns:
        Batch dictionary with the same keys and at most ``segment_frames`` on
        the frame axis.
    """

    if segment_frames <= 0 or batch["coded_sp"].shape[2] <= segment_frames:
        return batch

    coded_sp = batch["coded_sp"]
    condition = batch["condition"]
    lengths = batch["lengths"]
    batch_size, coded_channels, _ = coded_sp.shape
    condition_channels = condition.shape[1]
    cropped_coded = torch.zeros(
        batch_size,
        coded_channels,
        segment_frames,
        dtype=coded_sp.dtype,
        device=coded_sp.device,
    )
    cropped_condition = torch.zeros(
        batch_size,
        condition_channels,
        segment_frames,
        dtype=condition.dtype,
        device=condition.device,
    )
    cropped_mask = torch.zeros(
        batch_size,
        1,
        segment_frames,
        dtype=batch["mask"].dtype,
        device=batch["mask"].device,
    )
    cropped_lengths = torch.minimum(
        lengths,
        torch.full_like(lengths, segment_frames),
    )

    for index, length_tensor in enumerate(lengths):
        length = int(length_tensor.item())
        crop_length = min(length, segment_frames)
        if length > segment_frames:
            high = length - segment_frames + 1
            start = int(torch.randint(high, (1,), generator=generator).item())
        else:
            start = 0
        end = start + crop_length
        cropped_coded[index, :, :crop_length] = coded_sp[index, :, start:end]
        cropped_condition[index, :, :crop_length] = condition[index, :, start:end]
        cropped_mask[index, :, :crop_length] = 1.0

    cropped = dict(batch)
    cropped["coded_sp"] = cropped_coded
    cropped["condition"] = cropped_condition
    cropped["lengths"] = cropped_lengths
    cropped["mask"] = cropped_mask
    return cropped


def speech_nll_loss(model: torch.nn.Module, batch: dict[str, Any]) -> torch.Tensor:
    """Return mean negative log likelihood per valid scalar feature.

    Args:
        model: Conditional flow consuming ``coded_sp``, ``condition``, and
            ``mask`` tensors.
        batch: Batch dictionary with ``coded_sp`` shaped
            ``[batch, coded_sp_channels, frames]``, ``condition`` shaped
            ``[batch, condition_channels, frames]``, and ``mask`` shaped
            ``[batch, 1, frames]``.

    Returns:
        Scalar tensor containing the average negative log likelihood.
    """

    z, log_det = model(
        batch["coded_sp"],
        condition=batch["condition"],
        mask=batch["mask"],
    )
    mask = batch["mask"].to(dtype=z.dtype, device=z.device)
    log_base = -0.5 * (z.square() + log(2.0 * pi))
    log_base = (log_base * mask).sum(dim=(1, 2))
    log_prob = log_base + log_det
    valid_values = torch.clamp_min(mask.sum(dim=(1, 2)) * z.shape[1], 1.0)
    return torch.mean(-log_prob / valid_values)


def iterate_loader(loader: DataLoader) -> Iterator[dict[str, Any]]:
    """Yield batches forever from a finite dataloader.

    Yields:
        Batch dictionaries produced by ``collate_speech_features``.
    """

    while True:
        yield from loader


def train_speech_flow(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    *,
    device: torch.device,
    num_steps: int,
    segment_frames: int,
    log_every: int,
    seed: int,
    progress_callback: Callable[[dict[str, float | int]], None] | None = None,
) -> list[dict[str, float | int]]:
    """Train a conditional speech flow and return logged loss history.

    The dataloader must yield ``coded_sp`` tensors shaped
    ``[batch, coded_sp_channels, frames]``, ``condition`` tensors shaped
    ``[batch, condition_channels, frames]``, ``mask`` tensors shaped
    ``[batch, 1, frames]``, and ``lengths`` tensors shaped ``[batch]``.
    """

    model.train()
    history: list[dict[str, float | int]] = []
    batches = iterate_loader(loader)
    generator = torch.Generator()
    generator.manual_seed(seed)
    for step in range(1, num_steps + 1):
        batch = move_batch_to_device(next(batches), device)
        batch = crop_batch_segments(
            batch, segment_frames=segment_frames, generator=generator
        )
        optimizer.zero_grad(set_to_none=True)
        loss = speech_nll_loss(model, batch)
        loss.backward()
        optimizer.step()
        if step == 1 or step % log_every == 0 or step == num_steps:
            entry = {"loss": float(loss.detach().cpu()), "step": step}
            history.append(entry)
            if progress_callback is not None:
                progress_callback(entry)
    return history
