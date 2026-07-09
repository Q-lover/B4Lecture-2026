"""WORLD F0, spectral envelope, aperiodicity, and synthesis wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class WorldFeatureConfig:
    """WORLD analysis settings used by the speech feature cache."""

    frame_period_ms: float = 10.0
    coded_sp_dim: int = 48
    f0_floor: float = 71.0
    f0_ceil: float = 800.0


@dataclass(frozen=True)
class WorldFeatureBundle:
    """WORLD analysis output plus low-dimensional spectral envelope."""

    f0: np.ndarray
    time_axis: np.ndarray
    spectral_envelope: np.ndarray
    aperiodicity: np.ndarray
    coded_sp: np.ndarray
    coded_ap: np.ndarray
    sample_rate: int
    frame_period_ms: float

    @property
    def frame_count(self) -> int:
        """Return the number of WORLD frames on the time axis."""

        return int(self.f0.shape[0])

    @property
    def fft_size(self) -> int:
        """Return the FFT size implied by spectral-envelope frequency bins."""

        return int((self.spectral_envelope.shape[1] - 1) * 2)

    def summary(self) -> dict[str, Any]:
        """Return factual shape and voicing metadata."""

        return {
            "ap_shape": list(self.aperiodicity.shape),
            "coded_ap_shape": list(self.coded_ap.shape),
            "coded_sp_shape": list(self.coded_sp.shape),
            "f0_shape": list(self.f0.shape),
            "fft_size": self.fft_size,
            "frame_count": self.frame_count,
            "frame_period_ms": self.frame_period_ms,
            "sample_rate": self.sample_rate,
            "sp_shape": list(self.spectral_envelope.shape),
            "time_axis_shape": list(self.time_axis.shape),
            "voiced_f0_mean_hz": voiced_f0_mean(self.f0),
            "voiced_fraction": voiced_fraction(self.f0),
        }


def load_mono_audio(
    path: str | Path,
    *,
    target_sample_rate: int | None = None,
) -> tuple[np.ndarray, int]:
    """Load audio as mono float64, optionally resampling to ``target_sample_rate``.

    Returns:
        A waveform array shaped ``[samples]`` and the resolved sample rate.
    """

    import soundfile as sf
    from scipy.signal import resample_poly

    waveform, sample_rate = sf.read(path, dtype="float64", always_2d=False)
    if waveform.ndim == 2:
        waveform = np.mean(waveform, axis=1)
    waveform = np.ascontiguousarray(waveform, dtype=np.float64)
    if target_sample_rate is not None and sample_rate != target_sample_rate:
        divisor = gcd(int(sample_rate), int(target_sample_rate))
        waveform = resample_poly(
            waveform,
            int(target_sample_rate) // divisor,
            int(sample_rate) // divisor,
        ).astype(np.float64, copy=False)
        sample_rate = int(target_sample_rate)
    return waveform, int(sample_rate)


def analyze_world(
    waveform: np.ndarray,
    sample_rate: int,
    config: WorldFeatureConfig,
) -> WorldFeatureBundle:
    """Analyze waveform with WORLD and encode the spectral envelope as ``coded_sp``.

    Args:
        waveform: Mono waveform array shaped ``[samples]``.
        sample_rate: Audio sample rate in Hz.
        config: WORLD analysis and coding settings.

    Returns:
        Bundle containing F0/time arrays shaped ``[frames]``, full spectral
        envelope and aperiodicity arrays shaped ``[frames, fft_bins]``,
        ``coded_sp`` shaped ``[frames, coded_sp_dim]``, and ``coded_ap`` shaped
        ``[frames, coded_ap_channels]``.
    """

    import pyworld as pw

    waveform = np.ascontiguousarray(waveform, dtype=np.float64)
    f0, time_axis = pw.dio(
        waveform,
        sample_rate,
        f0_floor=config.f0_floor,
        f0_ceil=config.f0_ceil,
        frame_period=config.frame_period_ms,
    )
    refined_f0 = pw.stonemask(waveform, f0, time_axis, sample_rate)
    spectral_envelope = pw.cheaptrick(waveform, refined_f0, time_axis, sample_rate)
    aperiodicity = pw.d4c(waveform, refined_f0, time_axis, sample_rate)
    coded_sp = pw.code_spectral_envelope(
        spectral_envelope,
        sample_rate,
        config.coded_sp_dim,
    ).astype(np.float32)
    coded_ap = pw.code_aperiodicity(
        np.ascontiguousarray(aperiodicity, dtype=np.float64),
        sample_rate,
    ).astype(np.float32)
    return WorldFeatureBundle(
        f0=refined_f0.astype(np.float64, copy=False),
        time_axis=time_axis.astype(np.float64, copy=False),
        spectral_envelope=spectral_envelope.astype(np.float64, copy=False),
        aperiodicity=aperiodicity.astype(np.float64, copy=False),
        coded_sp=coded_sp,
        coded_ap=coded_ap,
        sample_rate=sample_rate,
        frame_period_ms=config.frame_period_ms,
    )


def decode_spectral_envelope(
    coded_sp: np.ndarray, sample_rate: int, fft_size: int
) -> np.ndarray:
    """Decode WORLD ``coded_sp`` back to a spectral envelope.

    Args:
        coded_sp: Array shaped ``[frames, coded_sp_dim]``.

    Returns:
        Array shaped ``[frames, fft_size / 2 + 1]``.
    """

    import pyworld as pw

    return pw.decode_spectral_envelope(
        np.ascontiguousarray(coded_sp, dtype=np.float64),
        sample_rate,
        fft_size,
    )


def decode_aperiodicity(
    coded_ap: np.ndarray, sample_rate: int, fft_size: int
) -> np.ndarray:
    """Decode WORLD ``coded_ap`` back to full-band aperiodicity.

    Args:
        coded_ap: Array shaped ``[frames, coded_ap_channels]``.

    Returns:
        Array shaped ``[frames, fft_size / 2 + 1]``.
    """

    import pyworld as pw

    return pw.decode_aperiodicity(
        np.ascontiguousarray(coded_ap, dtype=np.float64),
        sample_rate,
        fft_size,
    )


def world_aux_features(
    f0: np.ndarray,
    coded_ap: np.ndarray,
    *,
    vuv_f0: np.ndarray | None = None,
) -> np.ndarray:
    """Build frame-wise WORLD auxiliary condition features.

    The returned feature is ``[log1p(f0), vuv, coded_ap...]`` with shape
    ``[frames, 2 + coded_ap_channels]``. ``vuv_f0`` can be supplied when
    ``f0`` has already been shifted but voicing should follow the original F0.
    """

    f0 = np.asarray(f0, dtype=np.float64).reshape(-1)
    coded_ap = np.asarray(coded_ap, dtype=np.float32)
    if coded_ap.ndim != 2:
        raise ValueError(
            f"expected coded_ap with shape [frames, channels], got {coded_ap.shape}"
        )
    if coded_ap.shape[0] != f0.shape[0]:
        raise ValueError(
            f"frame mismatch: f0={f0.shape[0]}, coded_ap={coded_ap.shape[0]}"
        )
    voicing_source = (
        f0 if vuv_f0 is None else np.asarray(vuv_f0, dtype=np.float64).reshape(-1)
    )
    if voicing_source.shape[0] != f0.shape[0]:
        raise ValueError(
            f"frame mismatch: f0={f0.shape[0]}, vuv_f0={voicing_source.shape[0]}"
        )
    log1p_f0 = np.log1p(np.maximum(f0, 0.0)).astype(np.float32)[:, None]
    vuv = (voicing_source > 0.0).astype(np.float32)[:, None]
    return np.concatenate(
        [log1p_f0, vuv, coded_ap.astype(np.float32, copy=False)], axis=1
    )


def synthesize_world(
    f0: np.ndarray,
    spectral_envelope: np.ndarray,
    aperiodicity: np.ndarray,
    sample_rate: int,
    *,
    frame_period_ms: float,
) -> np.ndarray:
    """Synthesize waveform from WORLD features.

    Args:
        f0: Array shaped ``[frames]`` in Hz.
        spectral_envelope: Array shaped ``[frames, fft_bins]``.
        aperiodicity: Array shaped ``[frames, fft_bins]``.

    Returns:
        Waveform array shaped ``[samples]``.
    """

    import pyworld as pw

    waveform = pw.synthesize(
        np.ascontiguousarray(f0, dtype=np.float64),
        np.ascontiguousarray(spectral_envelope, dtype=np.float64),
        np.ascontiguousarray(aperiodicity, dtype=np.float64),
        sample_rate,
        frame_period_ms,
    )
    return np.asarray(waveform, dtype=np.float64)


def voiced_fraction(f0: np.ndarray) -> float:
    """Return the fraction of frames with positive F0.

    Args:
        f0: Array shaped ``[frames]`` in Hz.
    """

    if len(f0) == 0:
        return 0.0
    return float(np.mean(np.asarray(f0) > 0.0))


def voiced_f0_mean(f0: np.ndarray) -> float | None:
    """Return the mean F0 over voiced frames, or ``None`` when no voiced frame exists.

    Args:
        f0: Array shaped ``[frames]`` in Hz.
    """

    f0 = np.asarray(f0, dtype=np.float64)
    voiced = f0[f0 > 0.0]
    if len(voiced) == 0:
        return None
    return float(np.mean(voiced))


def shift_f0_by_voiced_mean(
    source_f0: np.ndarray,
    *,
    source_mean_hz: float | None,
    target_mean_hz: float | None,
) -> np.ndarray:
    """Shift voiced F0 by the target/source voiced-mean ratio.

    Args:
        source_f0: Array shaped ``[frames]`` in Hz.

    Returns:
        Array shaped ``[frames]`` in Hz.
    """

    shifted = np.asarray(source_f0, dtype=np.float64).copy()
    if source_mean_hz is None or target_mean_hz is None or source_mean_hz <= 0.0:
        return shifted
    voiced = shifted > 0.0
    shifted[voiced] *= float(target_mean_hz) / float(source_mean_hz)
    return shifted
