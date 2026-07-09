"""Speech loss, spectrogram, and waveform visualization helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _hz_to_mel(frequency_hz: np.ndarray | float) -> np.ndarray | float:
    """Convert frequencies in Hz to mel values."""

    return 2595.0 * np.log10(1.0 + np.asarray(frequency_hz) / 700.0)


def _mel_to_hz(mels: np.ndarray | float) -> np.ndarray | float:
    """Convert mel values to frequencies in Hz."""

    return 700.0 * (np.power(10.0, np.asarray(mels) / 2595.0) - 1.0)


def _mel_filterbank(
    *,
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float,
    f_max: float,
) -> np.ndarray:
    """Build a triangular mel filterbank.

    Returns:
        Array shaped ``[n_mels, n_fft / 2 + 1]``.
    """

    frequencies = np.linspace(0.0, sample_rate / 2.0, n_fft // 2 + 1)
    mel_edges = np.linspace(_hz_to_mel(f_min), _hz_to_mel(f_max), n_mels + 2)
    hz_edges = _mel_to_hz(mel_edges)
    filters = np.zeros((n_mels, frequencies.shape[0]), dtype=np.float64)
    for index in range(n_mels):
        lower, center, upper = hz_edges[index : index + 3]
        if center <= lower or upper <= center:
            continue
        left = (frequencies - lower) / (center - lower)
        right = (upper - frequencies) / (upper - center)
        filters[index] = np.maximum(0.0, np.minimum(left, right))
        area = np.sum(filters[index])
        if area > 0.0:
            filters[index] /= area
    return filters


def waveform_to_log_mel_spectrogram(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 80,
    f_min: float = 30.0,
    f_max: float | None = None,
    eps: float = 1e-10,
) -> np.ndarray:
    """Compute a simple log-mel spectrogram shaped ``[mel, frame]``.

    Args:
        waveform: Mono waveform shaped ``[samples]`` or multi-channel waveform
            shaped ``[samples, channels]``.

    Returns:
        Array shaped ``[n_mels, frames]``.
    """

    waveform = np.asarray(waveform, dtype=np.float64)
    if waveform.ndim == 2:
        waveform = np.mean(waveform, axis=1)
    if waveform.ndim != 1:
        raise ValueError(f"expected mono waveform, got shape {waveform.shape}")
    if n_fft <= 0 or hop_length <= 0 or n_mels <= 0:
        raise ValueError("n_fft, hop_length, and n_mels must be positive.")
    if waveform.shape[0] < n_fft:
        waveform = np.pad(waveform, (0, n_fft - waveform.shape[0]))
    remainder = (waveform.shape[0] - n_fft) % hop_length
    if remainder:
        waveform = np.pad(waveform, (0, hop_length - remainder))

    frames = np.lib.stride_tricks.sliding_window_view(waveform, n_fft)[::hop_length]
    window = np.hanning(n_fft)
    power = np.square(np.abs(np.fft.rfft(frames * window[None, :], axis=1)))
    filters = _mel_filterbank(
        sample_rate=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        f_min=f_min,
        f_max=float(f_max or sample_rate / 2.0),
    )
    mel_power = power @ filters.T
    return np.log10(np.maximum(mel_power.T, eps)).astype(np.float32)


def plot_loss_curve(
    history: list[dict[str, float | int]], output_path: str | Path
) -> None:
    """Plot speech training loss history."""

    import matplotlib.pyplot as plt

    steps = [int(row["step"]) for row in history]
    losses = [float(row["loss"]) for row in history]
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.plot(steps, losses)
    ax.set_xlabel("step")
    ax.set_ylabel("negative log likelihood")
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_coded_sp_comparison(
    target_coded_sp: np.ndarray,
    generated_coded_sp: np.ndarray,
    output_path: str | Path,
) -> None:
    """Plot target and generated coded spectral envelopes side by side.

    Both inputs are arrays shaped ``[frames, coded_sp_channels]``.
    """

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1, 2, figsize=(9, 3.5), constrained_layout=True, sharey=True
    )
    panels = [
        ("target coded_sp", target_coded_sp),
        ("generated coded_sp", generated_coded_sp),
    ]
    for ax, (title, values) in zip(axes, panels, strict=True):
        image = ax.imshow(
            values.T, aspect="auto", origin="lower", interpolation="nearest"
        )
        ax.set_title(title)
        ax.set_xlabel("frame")
        ax.set_ylabel("channel")
        fig.colorbar(image, ax=ax, shrink=0.85)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_spectral_envelope_comparison(
    target_sp: np.ndarray,
    generated_sp: np.ndarray,
    output_path: str | Path,
    *,
    eps: float = 1e-10,
) -> None:
    """Plot decoded target and generated WORLD spectral envelopes side by side.

    Both inputs are arrays shaped ``[frames, fft_bins]``.
    """

    import matplotlib.pyplot as plt

    target_log_sp = np.log10(np.maximum(np.asarray(target_sp, dtype=np.float64), eps))
    generated_log_sp = np.log10(
        np.maximum(np.asarray(generated_sp, dtype=np.float64), eps)
    )
    vmin = float(min(np.min(target_log_sp), np.min(generated_log_sp)))
    vmax = float(max(np.max(target_log_sp), np.max(generated_log_sp)))
    fig, axes = plt.subplots(
        1, 2, figsize=(9, 3.5), constrained_layout=True, sharey=True
    )
    panels = [
        ("target decoded sp", target_log_sp),
        ("generated decoded sp", generated_log_sp),
    ]
    for ax, (title, values) in zip(axes, panels, strict=True):
        image = ax.imshow(
            values.T,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(title)
        ax.set_xlabel("frame")
        ax.set_ylabel("frequency bin")
        fig.colorbar(image, ax=ax, shrink=0.85)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_mel_spectrogram_comparison(
    target_waveform: np.ndarray,
    target_sample_rate: int,
    generated_waveform: np.ndarray,
    generated_sample_rate: int,
    output_path: str | Path,
) -> None:
    """Plot target and generated waveform log-mel spectrograms side by side.

    Waveform inputs are arrays shaped ``[samples]``.
    """

    import matplotlib.pyplot as plt

    target_mel = waveform_to_log_mel_spectrogram(target_waveform, target_sample_rate)
    generated_mel = waveform_to_log_mel_spectrogram(
        generated_waveform, generated_sample_rate
    )
    vmin = float(min(np.min(target_mel), np.min(generated_mel)))
    vmax = float(max(np.max(target_mel), np.max(generated_mel)))
    fig, axes = plt.subplots(
        1, 2, figsize=(9, 3.5), constrained_layout=True, sharey=True
    )
    panels = [
        ("target waveform mel", target_mel, target_sample_rate),
        ("generated waveform mel", generated_mel, generated_sample_rate),
    ]
    for ax, (title, values, sample_rate) in zip(axes, panels, strict=True):
        duration_sec = values.shape[1] * 256 / sample_rate
        image = ax.imshow(
            values,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            extent=[0.0, duration_sec, 0.0, values.shape[0]],
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(title)
        ax.set_xlabel("time [sec]")
        ax.set_ylabel("mel channel")
        fig.colorbar(image, ax=ax, shrink=0.85)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
