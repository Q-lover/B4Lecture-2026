"""WORLD coded spectral envelope generation and synthesis helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from nf_assignment.speech.conditions import (
    WORLD_AUX_CONDITION,
    ConditionSpec,
    parse_condition_spec,
)
from nf_assignment.speech.features.alignment import crop_or_pad_frames
from nf_assignment.speech.features.content import extract_resampled_condition_features
from nf_assignment.speech.features.world import (
    WorldFeatureBundle,
    decode_spectral_envelope,
    shift_f0_by_voiced_mean,
    synthesize_world,
    voiced_f0_mean,
    world_aux_features,
)


def generate_coded_sp(
    model: torch.nn.Module,
    *,
    condition: torch.Tensor,
    mask: torch.Tensor,
    latent_scale: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample normalized coded spectral envelopes from a conditional flow.

    Args:
        model: Conditional flow whose inverse maps latent features to
            normalized ``coded_sp`` features.
        condition: Tensor shaped ``[batch, condition_channels, frames]``.
        mask: Tensor shaped ``[batch, 1, frames]``.
        latent_scale: Standard deviation multiplier for base Gaussian samples.
        generator: Optional PyTorch RNG.

    Returns:
        Tensor shaped ``[batch, coded_sp_channels, frames]``.
    """

    batch, _, frames = condition.shape
    target_channels = model.transforms[0].channels
    z = torch.randn(
        batch,
        target_channels,
        frames,
        dtype=condition.dtype,
        device=condition.device,
        generator=generator,
    )
    z = z * float(latent_scale) * mask.to(dtype=z.dtype, device=z.device)
    coded_sp, _ = model.inverse(z, condition=condition, mask=mask)
    return coded_sp * mask.to(dtype=coded_sp.dtype, device=coded_sp.device)


def extract_vc_condition(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    condition: str | Sequence[str] | ConditionSpec,
    frame_count: int,
    normalizer: Any | None = None,
    normalizers: dict[str, Any] | None = None,
    device: torch.device | str | None = None,
    hubert_model: torch.nn.Module | None = None,
    gpu: int | None = 0,
    source_world: WorldFeatureBundle | None = None,
    target_voiced_mean_f0_hz: float | None = None,
) -> dict[str, Any]:
    """Extract a source-side content condition batch for VC inference.

    Args:
        waveform: Source mono waveform array shaped ``[samples]``.
        sample_rate: Source sample rate in Hz.
        condition: Content condition name or component list.
        frame_count: WORLD frame count to align to.

    Returns:
        Dictionary containing ``condition`` shaped
        ``[batch=1, condition_channels, frames]``, ``mask`` shaped
        ``[batch=1, 1, frames]``, and NumPy arrays shaped ``[frames, channels]``
        for aligned source-side components.
    """

    condition_spec = parse_condition_spec(condition)
    base_condition = condition_spec.single_content_condition()
    use_world_aux = condition_spec.uses_world_aux
    features = extract_resampled_condition_features(
        waveform,
        sample_rate,
        target_frame_count=frame_count,
        conditions=(base_condition,),
        hubert_model=hubert_model,
        device=device,
        gpu=gpu,
    )
    feature = features[base_condition]
    content = feature.aligned.astype(np.float32, copy=False)
    components: dict[str, np.ndarray] = {base_condition: content}
    aligned = content
    normalized_components = []
    component_normalizers = normalizers or {}
    if base_condition in component_normalizers:
        normalized_components.append(
            component_normalizers[base_condition].normalize(content)
        )
    else:
        normalized_components.append(content)
    shifted_f0 = None
    if use_world_aux:
        if source_world is None:
            raise ValueError(
                "source_world is required when world_aux is a condition component."
            )
        shifted_f0 = shift_f0_by_voiced_mean(
            source_world.f0,
            source_mean_hz=voiced_f0_mean(source_world.f0),
            target_mean_hz=target_voiced_mean_f0_hz,
        )
        aux = crop_or_pad_frames(
            world_aux_features(
                shifted_f0, source_world.coded_ap, vuv_f0=source_world.f0
            ),
            frame_count,
        )
        aux = aux.astype(np.float32, copy=False)
        aligned = np.concatenate([content, aux], axis=1)
        components[WORLD_AUX_CONDITION] = aux
        if WORLD_AUX_CONDITION in component_normalizers:
            normalized_components.append(
                component_normalizers[WORLD_AUX_CONDITION].normalize(aux)
            )
        else:
            normalized_components.append(aux)
    if normalizers is not None:
        normalized = np.concatenate(normalized_components, axis=1)
    elif normalizer is not None:
        normalized = normalizer.normalize(aligned)
    else:
        normalized = aligned
    condition_tensor = torch.from_numpy(normalized.T.copy()).unsqueeze(0)
    mask = torch.ones(1, 1, int(aligned.shape[0]), dtype=torch.float32)
    if device is not None:
        resolved_device = torch.device(device)
        condition_tensor = condition_tensor.to(resolved_device)
        mask = mask.to(resolved_device)
    return {
        "aligned": aligned,
        "base_condition": base_condition,
        "components": components,
        "condition": condition_tensor,
        "condition_components": list(condition_spec.components),
        "condition_name": condition_spec.name,
        "feature": feature,
        "length": int(aligned.shape[0]),
        "mask": mask,
        "raw": feature.raw,
        "shifted_f0": shifted_f0,
        "uses_world_aux": use_world_aux,
    }


def fit_world_frames(
    f0: np.ndarray,
    aperiodicity: np.ndarray,
    *,
    frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop or edge-pad WORLD F0/AP arrays to match generated coded_sp length.

    Args:
        f0: Array shaped ``[frames]`` in Hz.
        aperiodicity: Array shaped ``[frames, fft_bins]``.
        frames: Desired output frame count.

    Returns:
        A pair ``(f0, aperiodicity)`` shaped ``[frames]`` and
        ``[frames, fft_bins]``.
    """

    fitted_f0 = crop_or_pad_frames(
        np.asarray(f0, dtype=np.float64).reshape(-1, 1), frames
    )[:, 0]
    fitted_ap = crop_or_pad_frames(np.asarray(aperiodicity, dtype=np.float64), frames)
    return fitted_f0.astype(np.float64, copy=False), fitted_ap.astype(
        np.float64, copy=False
    )


def synthesize_generated_world(
    coded_sp: np.ndarray,
    *,
    synthesis_features: dict[str, Any],
    target_voiced_mean_f0_hz: float | None,
) -> dict[str, Any]:
    """Decode generated coded_sp and synthesize waveform with shifted source F0/AP.

    Args:
        coded_sp: Generated WORLD-coded spectral envelope shaped
            ``[frames, coded_sp_dim]``.
        synthesis_features: Source WORLD features including F0/AP arrays.
        target_voiced_mean_f0_hz: Target voiced-F0 mean used for F0 shifting.

    Returns:
        Dictionary containing decoded ``spectral_envelope`` shaped
        ``[frames, fft_bins]``, shifted ``f0`` shaped ``[frames]``,
        ``aperiodicity`` shaped ``[frames, fft_bins]``, and ``waveform`` shaped
        ``[samples]``.
    """

    frames = int(coded_sp.shape[0])
    source_f0, source_ap = fit_world_frames(
        synthesis_features["f0"],
        synthesis_features["aperiodicity"],
        frames=frames,
    )
    shifted_f0 = shift_f0_by_voiced_mean(
        source_f0,
        source_mean_hz=voiced_f0_mean(source_f0),
        target_mean_hz=target_voiced_mean_f0_hz,
    )
    spectral_envelope = decode_spectral_envelope(
        coded_sp,
        int(synthesis_features["sample_rate"]),
        int(synthesis_features["fft_size"]),
    )
    waveform = synthesize_world(
        shifted_f0,
        spectral_envelope,
        source_ap,
        int(synthesis_features["sample_rate"]),
        frame_period_ms=float(synthesis_features["frame_period_ms"]),
    )
    return {
        "aperiodicity": source_ap,
        "f0": shifted_f0,
        "spectral_envelope": spectral_envelope,
        "waveform": waveform,
    }


def synthesize_target_world(
    coded_sp: np.ndarray,
    *,
    synthesis_features: dict[str, Any],
) -> dict[str, Any]:
    """Decode target coded_sp and synthesize waveform with target F0/AP.

    Args:
        coded_sp: Target WORLD-coded spectral envelope shaped
            ``[frames, coded_sp_dim]``.
        synthesis_features: Target WORLD features including F0/AP arrays.

    Returns:
        Dictionary containing decoded ``spectral_envelope`` shaped
        ``[frames, fft_bins]``, ``f0`` shaped ``[frames]``, ``aperiodicity``
        shaped ``[frames, fft_bins]``, and ``waveform`` shaped ``[samples]``.
    """

    frames = int(coded_sp.shape[0])
    target_f0, target_ap = fit_world_frames(
        synthesis_features["f0"],
        synthesis_features["aperiodicity"],
        frames=frames,
    )
    spectral_envelope = decode_spectral_envelope(
        coded_sp,
        int(synthesis_features["sample_rate"]),
        int(synthesis_features["fft_size"]),
    )
    waveform = synthesize_world(
        target_f0,
        spectral_envelope,
        target_ap,
        int(synthesis_features["sample_rate"]),
        frame_period_ms=float(synthesis_features["frame_period_ms"]),
    )
    return {
        "aperiodicity": target_ap,
        "f0": target_f0,
        "spectral_envelope": spectral_envelope,
        "waveform": waveform,
    }
