"""Frame-rate alignment between WORLD and condition features."""

from __future__ import annotations

from typing import Any

import numpy as np


def linear_resample_frames(features: np.ndarray, target_frames: int) -> np.ndarray:
    """Linearly resample a ``[frames, channels]`` feature matrix to ``target_frames``."""

    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(
            f"expected features with shape [frames, channels], got {features.shape}"
        )
    if target_frames <= 0:
        raise ValueError("target_frames must be positive.")
    if features.shape[0] == 0:
        raise ValueError("features must contain at least one frame.")
    if features.shape[0] == target_frames:
        return features.copy()
    if features.shape[0] == 1:
        return np.repeat(features, target_frames, axis=0)

    source_x = np.linspace(0.0, 1.0, features.shape[0], dtype=np.float64)
    target_x = np.linspace(0.0, 1.0, target_frames, dtype=np.float64)
    aligned = np.empty((target_frames, features.shape[1]), dtype=np.float32)
    for channel in range(features.shape[1]):
        aligned[:, channel] = np.interp(target_x, source_x, features[:, channel])
    return aligned


def crop_or_pad_frames(features: np.ndarray, target_frames: int) -> np.ndarray:
    """Crop or edge-pad a ``[frames, channels]`` feature matrix to ``target_frames``."""

    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(
            f"expected features with shape [frames, channels], got {features.shape}"
        )
    if target_frames <= 0:
        raise ValueError("target_frames must be positive.")
    if features.shape[0] == 0:
        raise ValueError("features must contain at least one frame.")
    if features.shape[0] >= target_frames:
        return features[:target_frames].copy()
    padding = np.repeat(features[-1:, :], target_frames - features.shape[0], axis=0)
    return np.concatenate([features, padding], axis=0).astype(np.float32, copy=False)


def repeat_upsample_frames(features: np.ndarray, factor: int) -> np.ndarray:
    """Nearest-neighbor upsample a frame matrix by repeating frames.

    Args:
        features: Array shaped ``[frames, channels]``.

    Returns:
        Array shaped ``[frames * factor, channels]``.
    """

    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(
            f"expected features with shape [frames, channels], got {features.shape}"
        )
    if factor <= 0:
        raise ValueError("factor must be positive.")
    if features.shape[0] == 0:
        raise ValueError("features must contain at least one frame.")
    return np.repeat(features, factor, axis=0).astype(np.float32, copy=False)


def normalize_rows(features: np.ndarray) -> np.ndarray:
    """Clamp negatives and normalize each frame to sum to one.

    Args:
        features: Array shaped ``[frames, channels]``.

    Returns:
        Array shaped ``[frames, channels]``.
    """

    features = np.maximum(np.asarray(features, dtype=np.float32), 0.0)
    if features.ndim != 2:
        raise ValueError(
            f"expected features with shape [frames, channels], got {features.shape}"
        )
    row_sum = features.sum(axis=1, keepdims=True)
    return np.divide(features, np.maximum(row_sum, np.finfo(np.float32).eps)).astype(
        np.float32
    )


def frame_feature_summary(
    *,
    name: str,
    features: np.ndarray,
    aligned: np.ndarray,
    target_frames: int,
    duration_sec: float | None = None,
    theoretical_frame_period_ms: float | None = None,
) -> dict[str, Any]:
    """Return factual shape and frame-rate metadata for a feature matrix.

    ``features`` and ``aligned`` are arrays shaped ``[frames, channels]``.
    """

    frames = int(features.shape[0])
    effective_frame_period_ms = None
    if duration_sec is not None and frames > 0:
        effective_frame_period_ms = 1000.0 * duration_sec / frames
    return {
        "aligned_shape": list(aligned.shape),
        "aligned_to_target_frame_count": int(aligned.shape[0]),
        "channels": int(features.shape[1]),
        "effective_frame_period_ms": effective_frame_period_ms,
        "frame_difference_vs_target": int(frames - target_frames),
        "frames": frames,
        "name": name,
        "shape": list(features.shape),
        "target_to_feature_frame_ratio": (
            float(target_frames / frames) if frames else None
        ),
        "theoretical_frame_period_ms": theoretical_frame_period_ms,
    }
