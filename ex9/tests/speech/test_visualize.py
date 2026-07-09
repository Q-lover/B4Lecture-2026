"""Speech visualization tests."""

from __future__ import annotations

import numpy as np
from nf_assignment.speech.visualize import (
    plot_mel_spectrogram_comparison,
    plot_spectral_envelope_comparison,
    waveform_to_log_mel_spectrogram,
)


def test_waveform_to_log_mel_spectrogram_returns_finite_mel_frames() -> None:
    sample_rate = 16000
    time = np.arange(sample_rate // 2, dtype=np.float32) / sample_rate
    waveform = np.sin(2.0 * np.pi * 220.0 * time)

    mel = waveform_to_log_mel_spectrogram(
        waveform,
        sample_rate,
        n_fft=512,
        hop_length=128,
        n_mels=40,
    )

    assert mel.shape[0] == 40
    assert mel.shape[1] > 1
    assert np.isfinite(mel).all()


def test_plot_mel_spectrogram_comparison_writes_png(tmp_path) -> None:
    sample_rate = 16000
    time = np.arange(sample_rate // 4, dtype=np.float32) / sample_rate
    target = np.sin(2.0 * np.pi * 220.0 * time)
    generated = np.sin(2.0 * np.pi * 330.0 * time)
    output_path = tmp_path / "mel.png"

    plot_mel_spectrogram_comparison(
        target,
        sample_rate,
        generated,
        sample_rate,
        output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_spectral_envelope_comparison_writes_png(tmp_path) -> None:
    target = np.exp(np.linspace(-3.0, 1.0, 24, dtype=np.float64)).reshape(4, 6)
    generated = np.exp(np.linspace(-2.5, 1.5, 24, dtype=np.float64)).reshape(4, 6)
    output_path = tmp_path / "sp.png"

    plot_spectral_envelope_comparison(target, generated, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
