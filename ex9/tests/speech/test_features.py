"""Speech feature extraction and alignment tests."""

import sys
from importlib import import_module
from types import SimpleNamespace

import numpy as np
from nf_assignment.speech.extract_features import (
    _world_aux_condition,
    select_manifest_rows,
)
from nf_assignment.speech.features.alignment import (
    crop_or_pad_frames,
    linear_resample_frames,
    normalize_rows,
    repeat_upsample_frames,
)
from nf_assignment.speech.features.content import extract_resampled_condition_features
from nf_assignment.speech.features.world import (
    WorldFeatureBundle,
    decode_aperiodicity,
    decode_spectral_envelope,
    synthesize_world,
    voiced_f0_mean,
    world_aux_features,
)


def test_linear_resample_frames_interpolates() -> None:
    features = np.array([[0.0], [10.0], [20.0]], dtype=np.float32)

    aligned = linear_resample_frames(features, 5)

    np.testing.assert_allclose(aligned[:, 0], [0.0, 5.0, 10.0, 15.0, 20.0])


def test_normalize_rows_clamps_and_normalizes() -> None:
    features = np.array([[1.0, 1.0], [-1.0, 3.0]], dtype=np.float32)

    normalized = normalize_rows(features)

    np.testing.assert_allclose(normalized.sum(axis=1), [1.0, 1.0])
    np.testing.assert_allclose(normalized[1], [0.0, 1.0])


def test_crop_or_pad_frames_edge_pads_last_frame() -> None:
    features = np.array([[0.0], [1.0]], dtype=np.float32)

    padded = crop_or_pad_frames(features, 4)
    cropped = crop_or_pad_frames(features, 1)

    np.testing.assert_allclose(padded[:, 0], [0.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(cropped[:, 0], [0.0])


def test_repeat_upsample_frames_repeats_each_frame() -> None:
    features = np.array([[0.0], [1.0]], dtype=np.float32)

    upsampled = repeat_upsample_frames(features, 2)

    np.testing.assert_allclose(upsampled[:, 0], [0.0, 0.0, 1.0, 1.0])


def test_extract_resampled_condition_features_with_fake_extractors() -> None:
    waveform = np.zeros(1600, dtype=np.float64)

    def fake_hubert(wav: np.ndarray, sample_rate: int) -> np.ndarray:
        assert sample_rate == 16000
        assert wav.shape == (1920,)
        return np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)

    def fake_ppg(wav: np.ndarray, sample_rate: int) -> np.ndarray:
        assert sample_rate == 16000
        assert wav.shape == (1760,)
        return np.array(
            [
                [0.9, 0.1],
                [0.2, 0.8],
                [0.5, 0.5],
            ],
            dtype=np.float32,
        )

    features = extract_resampled_condition_features(
        waveform,
        16000,
        target_frame_count=4,
        conditions=("hubert_soft", "ppg"),
        device="cpu",
        gpu=None,
        hubert_extractor=fake_hubert,
        ppg_extractor=fake_ppg,
    )

    assert features["hubert_soft"].raw.shape == (2, 2)
    assert features["hubert_soft"].aligned.shape == (4, 2)
    assert features["ppg"].raw.shape == (3, 2)
    assert features["ppg"].aligned.shape == (4, 2)
    assert features["hubert_soft"].metadata["alignment_method"] == (
        "waveform_padding_then_repeat_upsample"
    )
    assert (
        features["ppg"].metadata["alignment_method"]
        == "waveform_padding_then_crop_or_pad"
    )
    np.testing.assert_allclose(
        features["hubert_soft"].aligned,
        np.array([[0.0, 1.0], [0.0, 1.0], [2.0, 3.0], [2.0, 3.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        features["ppg"].aligned.sum(axis=1), np.ones(4), atol=1e-6
    )


def test_select_manifest_rows_filters_split_and_utterance() -> None:
    rows = [
        {"split": "train", "utterance_id": "a"},
        {"split": "valid", "utterance_id": "b"},
        {"split": "train", "utterance_id": "c"},
    ]

    selected = select_manifest_rows(
        rows,
        split="train",
        speakers=None,
        utterance_id="c",
        max_utterances=2,
    )

    assert selected == [{"split": "train", "utterance_id": "c"}]


def test_select_manifest_rows_accepts_all_splits() -> None:
    rows = [
        {"split": "train", "utterance_id": "a"},
        {"split": "valid", "utterance_id": "b"},
        {"split": "test", "utterance_id": "c"},
    ]

    selected = select_manifest_rows(
        rows,
        split="all",
        speakers=None,
        utterance_id=None,
        max_utterances=3,
    )

    assert selected == rows


def test_select_manifest_rows_filters_speaker() -> None:
    rows = [
        {"speaker": "bdl", "split": "train", "utterance_id": "a"},
        {"speaker": "slt", "split": "train", "utterance_id": "a"},
    ]

    selected = select_manifest_rows(
        rows,
        split="train",
        speakers=("slt",),
        utterance_id=None,
        max_utterances=2,
    )

    assert selected == [{"speaker": "slt", "split": "train", "utterance_id": "a"}]


def test_select_manifest_rows_limits_utterances_per_speaker() -> None:
    rows = [
        {"speaker": "bdl", "split": "train", "utterance_id": "a"},
        {"speaker": "bdl", "split": "train", "utterance_id": "b"},
        {"speaker": "slt", "split": "train", "utterance_id": "a"},
        {"speaker": "slt", "split": "train", "utterance_id": "b"},
    ]

    selected = select_manifest_rows(
        rows,
        split="train",
        speakers=("bdl", "slt"),
        utterance_id=None,
        max_utterances=1,
    )

    assert selected == [
        {"speaker": "bdl", "split": "train", "utterance_id": "a"},
        {"speaker": "slt", "split": "train", "utterance_id": "a"},
    ]


def test_voiced_f0_mean_ignores_unvoiced_frames() -> None:
    assert voiced_f0_mean(np.array([0.0, 100.0, 0.0, 200.0])) == 150.0
    assert voiced_f0_mean(np.zeros(3)) is None


def test_world_aux_features_use_log1p_f0_vuv_and_coded_ap() -> None:
    f0 = np.array([0.0, 100.0, 200.0], dtype=np.float64)
    vuv_f0 = np.array([0.0, 50.0, 0.0], dtype=np.float64)
    coded_ap = np.array([[0.1], [0.2], [0.3]], dtype=np.float32)

    aux = world_aux_features(f0, coded_ap, vuv_f0=vuv_f0)

    assert aux.shape == (3, 3)
    np.testing.assert_allclose(aux[:, 0], np.log1p(f0).astype(np.float32))
    np.testing.assert_allclose(aux[:, 1], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(aux[:, 2], [0.1, 0.2, 0.3])


def test_extract_feature_helpers_build_world_aux_condition() -> None:
    world = WorldFeatureBundle(
        f0=np.array([0.0, 100.0], dtype=np.float64),
        time_axis=np.array([0.0, 0.01], dtype=np.float64),
        spectral_envelope=np.ones((2, 5), dtype=np.float64),
        aperiodicity=np.ones((2, 5), dtype=np.float64),
        coded_sp=np.ones((2, 4), dtype=np.float32),
        coded_ap=np.array([[0.1], [0.2]], dtype=np.float32),
        sample_rate=16000,
        frame_period_ms=10.0,
    )

    aux = _world_aux_condition(world)

    assert aux.name == "world_aux"
    assert aux.aligned.shape == (2, 3)
    np.testing.assert_allclose(aux.aligned[:, 0], np.log1p(world.f0))
    np.testing.assert_allclose(aux.aligned[:, 1], [0.0, 1.0])
    np.testing.assert_allclose(aux.aligned[:, 2], [0.1, 0.2])


def test_world_wrappers_pass_c_contiguous_arrays(monkeypatch) -> None:
    def fake_decode_spectral_envelope(coded_sp, sample_rate, fft_size):
        assert coded_sp.flags.c_contiguous
        assert coded_sp.dtype == np.float64
        assert sample_rate == 16000
        assert fft_size == 8
        return np.ones((coded_sp.shape[0], 5), dtype=np.float64)

    def fake_synthesize(
        f0, spectral_envelope, aperiodicity, sample_rate, frame_period_ms
    ):
        assert f0.flags.c_contiguous
        assert spectral_envelope.flags.c_contiguous
        assert aperiodicity.flags.c_contiguous
        assert sample_rate == 16000
        assert frame_period_ms == 10.0
        return np.zeros(16, dtype=np.float64)

    monkeypatch.setitem(
        sys.modules,
        "pyworld",
        SimpleNamespace(
            decode_aperiodicity=lambda coded_ap, sample_rate, fft_size: np.ones(
                (coded_ap.shape[0], 5), dtype=np.float64
            ),
            decode_spectral_envelope=fake_decode_spectral_envelope,
            synthesize=fake_synthesize,
        ),
    )
    coded_sp = np.arange(24, dtype=np.float32).reshape(4, 6)[:, ::2]
    spectral_envelope = decode_spectral_envelope(coded_sp, 16000, 8)
    decoded_ap = decode_aperiodicity(
        np.ones((4, 1), dtype=np.float32)[:, ::-1], 16000, 8
    )
    waveform = synthesize_world(
        np.arange(4, dtype=np.float32)[::-1],
        spectral_envelope[:, ::-1],
        np.ones((4, 5), dtype=np.float32)[:, ::-1],
        16000,
        frame_period_ms=10.0,
    )

    assert decoded_ap.shape == (4, 5)
    assert waveform.shape == (16,)


def test_extract_features_console_target_is_importable() -> None:
    module = import_module("nf_assignment.speech.extract_features")

    assert callable(module.main)
