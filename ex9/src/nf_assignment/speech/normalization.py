"""Channel-wise normalization helpers for cached speech frame features."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FeatureNormalizer:
    """Channel-wise affine normalization for frame features shaped ``[frames, channels]``."""

    mean: np.ndarray
    std: np.ndarray
    eps: float = 1e-6

    @classmethod
    def from_stats(
        cls, stats: dict[str, Any], *, eps: float = 1e-6
    ) -> FeatureNormalizer:
        """Create a normalizer from one feature entry in ``feature_statistics.json``."""

        mean = np.asarray(stats["mean"], dtype=np.float32)
        std = np.asarray(stats["std"], dtype=np.float32)
        if mean.ndim != 1 or std.ndim != 1 or mean.shape != std.shape:
            raise ValueError(
                "normalization statistics must contain matching 1-D mean/std arrays."
            )
        return cls(mean=mean, std=std, eps=eps)

    def normalize(self, features: np.ndarray) -> np.ndarray:
        """Apply channel-wise normalization.

        Args:
            features: Array shaped ``[frames, channels]``.

        Returns:
            Array shaped ``[frames, channels]``.
        """

        features = np.asarray(features, dtype=np.float32)
        return (features - self.mean) / np.maximum(self.std, self.eps)

    def denormalize(self, features: np.ndarray) -> np.ndarray:
        """Undo channel-wise normalization.

        Args:
            features: Array shaped ``[frames, channels]``.

        Returns:
            Array shaped ``[frames, channels]``.
        """

        features = np.asarray(features, dtype=np.float32)
        return features * np.maximum(self.std, self.eps) + self.mean


def load_feature_normalizers(
    path: str | Path,
    *,
    split: str = "train",
    eps: float = 1e-6,
) -> dict[str, FeatureNormalizer]:
    """Load channel-wise normalizers from ``feature_statistics.json``."""

    with Path(path).open(encoding="utf-8") as handle:
        data = json.load(handle)
    if split not in data:
        raise KeyError(f"statistics split is not available: {split}")
    features = data[split].get("features", {})
    return {
        feature_name: FeatureNormalizer.from_stats(feature_stats, eps=eps)
        for feature_name, feature_stats in features.items()
    }
