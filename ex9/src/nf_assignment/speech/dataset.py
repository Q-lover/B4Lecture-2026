"""Dataset utilities for cached WORLD-envelope speech features."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from nf_assignment.speech.conditions import ConditionSpec, parse_condition_spec
from nf_assignment.speech.data import normalize_speaker_list
from nf_assignment.speech.features.world import (
    WorldFeatureConfig,
    analyze_world,
    load_mono_audio,
)
from nf_assignment.speech.normalization import (
    FeatureNormalizer,
    load_feature_normalizers,
)
from torch.utils.data import Dataset


def read_feature_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Read the feature-cache JSONL manifest."""

    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _normalize_speaker_filter(
    speakers: str | Sequence[str] | None,
) -> tuple[str, ...] | None:
    """Normalize an optional speaker filter from config or CLI values."""

    if speakers is None:
        return None
    if isinstance(speakers, str):
        values = tuple(part.strip() for part in speakers.split(","))
    else:
        values = tuple(str(speaker).strip() for speaker in speakers)
    return normalize_speaker_list(values)


class SpeechFeatureDataset(Dataset):
    """Load one cached utterance as target ``coded_sp`` plus one aligned condition feature."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        condition: str | Sequence[str] | ConditionSpec,
        split: str | None = None,
        speakers: str | Sequence[str] | None = None,
        path_root: str | Path | None = None,
        statistics_path: str | Path | None = None,
        statistics_split: str = "train",
        normalize: bool = True,
    ):
        """Create a feature-cache dataset.

        Each item returns ``coded_sp`` and ``condition`` tensors shaped
        ``[channels, frames]`` before collation. The manifest rows point to
        cached NumPy arrays shaped ``[frames, channels]``.
        """

        self.manifest_path = Path(manifest_path)
        self.cache_dir = self.manifest_path.parent
        self.condition_spec = parse_condition_spec(condition)
        self.condition = self.condition_spec.name
        self.condition_components = self.condition_spec.components
        self.path_root = (
            Path(path_root) if path_root is not None else self._infer_path_root()
        )
        self.split = split
        self.speakers = _normalize_speaker_filter(speakers)
        rows = read_feature_manifest(self.manifest_path)
        speaker_set = set(self.speakers) if self.speakers is not None else None
        self.rows = [
            row
            for row in rows
            if (split is None or row.get("split") == split)
            and (
                speaker_set is None
                or str(row.get("speaker", "")).lower() in speaker_set
            )
        ]
        if not self.rows:
            raise ValueError(
                "No feature manifest rows matched the requested selection: "
                f"split={split!r}, speakers={self.speakers!r}."
            )
        self.row_index_by_speaker_utterance = {
            (str(row.get("speaker")), str(row.get("utterance_id"))): index
            for index, row in enumerate(self.rows)
        }

        self.normalizers: dict[str, FeatureNormalizer] = {}
        if normalize:
            stats_path = (
                Path(statistics_path)
                if statistics_path is not None
                else (self.cache_dir / "feature_statistics.json")
            )
            normalizers = load_feature_normalizers(stats_path, split=statistics_split)
            names = {"world_coded_sp", *self.condition_components}
            self.normalizers = {
                name: normalizers[name] for name in names if name in normalizers
            }

    def _infer_path_root(self) -> Path:
        """Infer the root used to resolve relative manifest paths."""

        summary_path = self.cache_dir / "feature_summary.json"
        if summary_path.exists():
            with summary_path.open(encoding="utf-8") as handle:
                summary = json.load(handle)
            path_root = summary.get("path_root")
            if path_root:
                return Path(path_root)
        return Path.cwd()

    def resolve_path(self, value: str | Path) -> Path:
        """Resolve an absolute or relative feature-cache path."""

        path = Path(value)
        if path.is_absolute():
            return path
        candidates = [
            self.path_root / path,
            Path.cwd() / path,
            self.cache_dir / path,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def __len__(self) -> int:
        """Return the number of manifest rows after split/speaker filtering."""

        return len(self.rows)

    def index_for_speaker_utterance(self, *, speaker: str, utterance_id: str) -> int:
        """Return the dataset index for a speaker utterance in the loaded split."""

        key = (speaker, utterance_id)
        if key not in self.row_index_by_speaker_utterance:
            raise KeyError(f"feature row is not available for speaker/utterance: {key}")
        return self.row_index_by_speaker_utterance[key]

    def _condition_components(self, row: dict[str, Any]) -> list[str]:
        """Return cached component names required by this dataset condition."""

        condition_paths = row.get("aligned_condition_paths", {})
        components = list(self.condition_components)
        missing = [name for name in components if name not in condition_paths]
        if missing:
            utterance_id = row.get("utterance_id")
            raise KeyError(
                f"condition components are not available for {utterance_id}: "
                f"{self.condition} needs {missing}"
            )
        return components

    def _load_condition(self, row: dict[str, Any]) -> np.ndarray:
        """Load and concatenate condition components.

        Returns:
            Array shaped ``[frames, condition_channels]``.
        """

        condition_paths = row.get("aligned_condition_paths", {})
        components = self._condition_components(row)
        arrays = []
        for component in components:
            features = np.load(self.resolve_path(condition_paths[component])).astype(
                np.float32
            )
            if features.ndim != 2:
                raise ValueError(
                    f"cached condition component must be 2-D [frames, channels]: {component}"
                )
            if component in self.normalizers:
                features = self.normalizers[component].normalize(features)
            arrays.append(features)
        frame_counts = {array.shape[0] for array in arrays}
        if len(frame_counts) != 1:
            raise ValueError(
                f"condition component frame mismatch for {row.get('utterance_id')}: "
                f"{self.condition} has {sorted(frame_counts)}"
            )
        return np.concatenate(arrays, axis=1) if len(arrays) > 1 else arrays[0]

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Load one utterance from the feature cache.

        Returns:
            Dictionary containing ``coded_sp`` shaped
            ``[coded_sp_channels, frames]`` and ``condition`` shaped
            ``[condition_channels, frames]`` as PyTorch tensors.
        """

        row = self.rows[index]
        coded_sp = np.load(self.resolve_path(row["world_coded_sp_path"])).astype(
            np.float32
        )
        condition = self._load_condition(row)
        if coded_sp.ndim != 2 or condition.ndim != 2:
            raise ValueError(
                "cached features must be 2-D arrays shaped [frames, channels]."
            )
        if coded_sp.shape[0] != condition.shape[0]:
            raise ValueError(
                f"frame mismatch for {row.get('utterance_id')}: "
                f"coded_sp={coded_sp.shape[0]}, {self.condition}={condition.shape[0]}"
            )

        if "world_coded_sp" in self.normalizers:
            coded_sp = self.normalizers["world_coded_sp"].normalize(coded_sp)

        return {
            "coded_sp": torch.from_numpy(coded_sp.T.copy()),
            "condition": torch.from_numpy(condition.T.copy()),
            "condition_name": self.condition,
            "item_id": row.get("item_id"),
            "length": int(coded_sp.shape[0]),
            "metadata": row,
            "speaker": row.get("speaker"),
            "speaker_id": row.get("speaker_id"),
            "split": row.get("split"),
            "utterance_id": row.get("utterance_id"),
        }

    def load_synthesis_features(
        self,
        index: int,
        *,
        world_config: WorldFeatureConfig | None = None,
        target_sample_rate: int | None = None,
        audio_loader=load_mono_audio,
        world_analyzer=analyze_world,
    ) -> dict[str, Any]:
        """Analyze WORLD F0/AP features from the manifest wav path for waveform synthesis."""

        row = self.rows[index]
        wav_path = self.resolve_path(row["wav_path"])
        waveform, sample_rate = audio_loader(
            wav_path, target_sample_rate=target_sample_rate
        )
        config = world_config or WorldFeatureConfig(
            frame_period_ms=float(row.get("world_frame_period_ms", 10.0))
        )
        world = world_analyzer(waveform, sample_rate, config)
        return {
            "aperiodicity": world.aperiodicity,
            "f0": world.f0,
            "fft_size": world.fft_size,
            "frame_period_ms": world.frame_period_ms,
            "sample_rate": world.sample_rate,
            "time_axis": world.time_axis,
            "utterance_id": row.get("utterance_id"),
            "wav_path": wav_path.as_posix(),
        }


def collate_speech_features(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad variable-length speech feature samples into batched tensors.

    Args:
        samples: Items whose ``coded_sp`` tensors are shaped
            ``[coded_sp_channels, frames]`` and whose ``condition`` tensors are
            shaped ``[condition_channels, frames]``.

    Returns:
        Dictionary with ``coded_sp`` shaped ``[batch, coded_sp_channels,
        max_frames]``, ``condition`` shaped ``[batch, condition_channels,
        max_frames]``, ``mask`` shaped ``[batch, 1, max_frames]``, and
        ``lengths`` shaped ``[batch]``.
    """

    if not samples:
        raise ValueError("samples must not be empty.")
    lengths = torch.tensor([sample["length"] for sample in samples], dtype=torch.long)
    batch_size = len(samples)
    max_length = int(torch.max(lengths).item())
    coded_channels = int(samples[0]["coded_sp"].shape[0])
    condition_channels = int(samples[0]["condition"].shape[0])
    coded_sp = torch.zeros(batch_size, coded_channels, max_length, dtype=torch.float32)
    condition = torch.zeros(
        batch_size, condition_channels, max_length, dtype=torch.float32
    )
    mask = torch.zeros(batch_size, 1, max_length, dtype=torch.float32)

    for index, sample in enumerate(samples):
        length = int(sample["length"])
        coded_sp[index, :, :length] = sample["coded_sp"]
        condition[index, :, :length] = sample["condition"]
        mask[index, :, :length] = 1.0

    return {
        "coded_sp": coded_sp,
        "condition": condition,
        "condition_name": samples[0]["condition_name"],
        "lengths": lengths,
        "mask": mask,
        "metadata": [sample["metadata"] for sample in samples],
        "speaker_ids": [sample.get("speaker_id") for sample in samples],
        "speakers": [sample.get("speaker") for sample in samples],
        "utterance_ids": [sample["utterance_id"] for sample in samples],
    }
