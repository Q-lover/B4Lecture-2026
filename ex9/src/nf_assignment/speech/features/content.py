"""HuBERT-Soft and PPG condition feature wrappers."""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from nf_assignment.speech.features.alignment import (
    crop_or_pad_frames,
    frame_feature_summary,
    normalize_rows,
    repeat_upsample_frames,
)

HUBERT_EXPECTED_SAMPLE_RATE = 16000
HUBERT_FRAME_PERIOD_MS = 20.0
HUBERT_ALIGNMENT_PADDING_MS = 10.0
HUBERT_TO_WORLD_UPSAMPLE_FACTOR = 2
PPG_ALIGNMENT_PADDING_MS = 5.0


@dataclass(frozen=True)
class ResampledConditionFeature:
    """A raw condition feature and its WORLD-frame-aligned counterpart."""

    name: str
    raw: np.ndarray
    aligned: np.ndarray
    theoretical_frame_period_ms: float | None = None
    metadata: dict[str, Any] | None = None

    def summary(
        self, *, target_frames: int, duration_sec: float | None = None
    ) -> dict[str, Any]:
        """Return factual shape and frame-rate metadata."""

        summary = frame_feature_summary(
            name=self.name,
            features=self.raw,
            aligned=self.aligned,
            target_frames=target_frames,
            duration_sec=duration_sec,
            theoretical_frame_period_ms=self.theoretical_frame_period_ms,
        )
        if self.metadata:
            summary["metadata"] = self.metadata
        return summary


FeatureExtractor = Callable[[np.ndarray, int], np.ndarray]


def _samples_from_ms(sample_rate: int, milliseconds: float) -> int:
    """Convert milliseconds to a nearest integer sample count."""

    return int(round(float(sample_rate) * milliseconds / 1000.0))


def _zero_pad_waveform(waveform: np.ndarray, pad_samples_each_side: int) -> np.ndarray:
    """Pad a mono waveform with zeros on both sides.

    Args:
        waveform: Array shaped ``[samples]``.
        pad_samples_each_side: Number of zero samples to add at the beginning
            and end.

    Returns:
        Array shaped ``[samples + 2 * pad_samples_each_side]``.
    """

    waveform = np.asarray(waveform)
    if waveform.ndim != 1:
        raise ValueError(
            f"expected mono waveform with shape [samples], got {waveform.shape}"
        )
    if pad_samples_each_side < 0:
        raise ValueError("pad_samples_each_side must be non-negative.")
    if pad_samples_each_side == 0:
        return waveform.copy()
    return np.pad(
        waveform, (pad_samples_each_side, pad_samples_each_side), mode="constant"
    )


def resolve_device(*, gpu: int | None = 0, cpu: bool = False) -> torch.device:
    """Resolve an inference device for optional content extractors."""

    if cpu or gpu is None or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(f"cuda:{gpu}")


def load_hubert_soft_model(
    *,
    hubert_repo: str | Path | None = None,
    device: torch.device | str | None = None,
    progress: bool = True,
) -> torch.nn.Module:
    """Load a HuBERT-Soft content encoder from a local repo or torch hub."""

    resolved_device = torch.device(
        device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    if hubert_repo is None:
        model = torch.hub.load(
            "bshall/hubert:main",
            "hubert_soft",
            trust_repo=True,
            progress=progress,
        )
    else:
        repo = Path(hubert_repo).resolve()
        repo_text = str(repo)
        if repo_text not in sys.path:
            sys.path.insert(0, repo_text)
        model = torch.hub.load(
            repo_text,
            "hubert_soft",
            source="local",
            trust_repo=True,
            progress=progress,
        )
    return model.to(resolved_device).eval()


def extract_hubert_soft(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    model: torch.nn.Module,
    device: torch.device | str,
) -> np.ndarray:
    """Extract HuBERT-Soft units as ``[frames, channels]`` float32 features.

    The temporary model input tensor is shaped ``[batch=1, channels=1,
    samples]``.
    """

    if sample_rate != HUBERT_EXPECTED_SAMPLE_RATE:
        raise ValueError("HuBERT-Soft extraction expects 16 kHz audio.")
    resolved_device = torch.device(device)
    wav = torch.from_numpy(np.asarray(waveform, dtype=np.float32))[None, None, :]
    wav = wav.to(resolved_device)
    with torch.inference_mode():
        units = model.units(wav)
    return units.squeeze(0).detach().cpu().numpy().astype(np.float32)


def extract_ppg(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    gpu: int | None = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extract PPGs as ``[frames, phonemes]`` float32 features.

    The temporary model input tensor is shaped ``[batch=1, channels=1,
    samples]``.
    """

    import ppgs

    audio = torch.from_numpy(np.asarray(waveform, dtype=np.float32))[None, None, :]
    with torch.inference_mode():
        ppg = ppgs.from_audio(audio, sample_rate, gpu=gpu)
    ppg_array = ppg.squeeze(0).transpose(0, 1).detach().cpu().numpy().astype(np.float32)
    metadata = {
        "hopsize_samples": int(ppgs.HOPSIZE),
        "phoneme_count": int(len(ppgs.PHONEMES)),
        "sample_rate": int(ppgs.SAMPLE_RATE),
        "theoretical_frame_period_ms": 1000.0
        * float(ppgs.HOPSIZE)
        / float(ppgs.SAMPLE_RATE),
    }
    return normalize_rows(ppg_array), metadata


def extract_resampled_condition_features(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    target_frame_count: int,
    conditions: Sequence[str] = ("hubert_soft", "ppg"),
    hubert_repo: str | Path | None = None,
    hubert_model: torch.nn.Module | None = None,
    device: torch.device | str | None = None,
    gpu: int | None = 0,
    hubert_extractor: FeatureExtractor | None = None,
    ppg_extractor: FeatureExtractor | None = None,
) -> dict[str, ResampledConditionFeature]:
    """Extract condition features and align them to the target WORLD frame count.

    Args:
        waveform: Mono waveform array shaped ``[samples]``.
        sample_rate: Audio sample rate in Hz.
        target_frame_count: Number of WORLD frames to align to.

    Returns:
        Mapping from condition name to raw and aligned arrays. Raw and aligned
        arrays are shaped ``[frames, channels]``.
    """

    requested = tuple(conditions)
    unsupported = sorted(set(requested) - {"hubert_soft", "ppg"})
    if unsupported:
        raise ValueError(f"Unsupported condition features: {unsupported}")
    if target_frame_count <= 0:
        raise ValueError("target_frame_count must be positive.")

    resolved_device = (
        torch.device(device) if device is not None else resolve_device(gpu=gpu)
    )
    features: dict[str, ResampledConditionFeature] = {}

    if "hubert_soft" in requested:
        hubert_pad_samples = _samples_from_ms(sample_rate, HUBERT_ALIGNMENT_PADDING_MS)
        hubert_waveform = _zero_pad_waveform(waveform, hubert_pad_samples)
        if hubert_extractor is not None:
            raw_hubert = hubert_extractor(hubert_waveform, sample_rate)
        else:
            model = hubert_model or load_hubert_soft_model(
                hubert_repo=hubert_repo,
                device=resolved_device,
            )
            raw_hubert = extract_hubert_soft(
                hubert_waveform,
                sample_rate,
                model=model,
                device=resolved_device,
            )
        aligned_hubert = crop_or_pad_frames(
            repeat_upsample_frames(raw_hubert, HUBERT_TO_WORLD_UPSAMPLE_FACTOR),
            target_frame_count,
        )
        features["hubert_soft"] = ResampledConditionFeature(
            name="hubert_soft",
            raw=raw_hubert,
            aligned=aligned_hubert,
            theoretical_frame_period_ms=HUBERT_FRAME_PERIOD_MS,
            metadata={
                "alignment_method": "waveform_padding_then_repeat_upsample",
                "input_padding_ms_each_side": HUBERT_ALIGNMENT_PADDING_MS,
                "input_padding_samples_each_side": hubert_pad_samples,
                "native_frame_period_ms": HUBERT_FRAME_PERIOD_MS,
                "raw_is_extracted_from_padded_waveform": True,
                "target_frame_fit": "crop_or_edge_pad",
                "upsample_factor": HUBERT_TO_WORLD_UPSAMPLE_FACTOR,
            },
        )

    if "ppg" in requested:
        ppg_pad_samples = _samples_from_ms(sample_rate, PPG_ALIGNMENT_PADDING_MS)
        ppg_waveform = _zero_pad_waveform(waveform, ppg_pad_samples)
        if ppg_extractor is not None:
            raw_ppg = normalize_rows(ppg_extractor(ppg_waveform, sample_rate))
            ppg_metadata = {}
            ppg_frame_period = None
        else:
            raw_ppg, ppg_metadata = extract_ppg(ppg_waveform, sample_rate, gpu=gpu)
            ppg_frame_period = ppg_metadata["theoretical_frame_period_ms"]
        ppg_metadata = {
            **ppg_metadata,
            "alignment_method": "waveform_padding_then_crop_or_pad",
            "input_padding_ms_each_side": PPG_ALIGNMENT_PADDING_MS,
            "input_padding_samples_each_side": ppg_pad_samples,
            "raw_is_extracted_from_padded_waveform": True,
            "target_frame_fit": "crop_or_edge_pad",
        }
        aligned_ppg = normalize_rows(crop_or_pad_frames(raw_ppg, target_frame_count))
        features["ppg"] = ResampledConditionFeature(
            name="ppg",
            raw=raw_ppg,
            aligned=aligned_ppg,
            theoretical_frame_period_ms=ppg_frame_period,
            metadata=ppg_metadata,
        )

    return features
