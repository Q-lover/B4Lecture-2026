"""Speech feature-cache dataset tests."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from nf_assignment.speech.dataset import (
    FeatureNormalizer,
    SpeechFeatureDataset,
    collate_speech_features,
    load_feature_normalizers,
    read_feature_manifest,
)
from nf_assignment.speech.features.world import (
    WorldFeatureBundle,
    shift_f0_by_voiced_mean,
)


def _write_json(path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle)
        handle.write("\n")


def _write_jsonl(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle)
            handle.write("\n")


def _fake_feature_cache(tmp_path, *, include_secondary_speaker: bool = False):
    cache_dir = tmp_path / "feature_cache"
    aligned_dir = cache_dir / "aligned"
    rows = []
    utterance_specs = [
        ("slt", 0, "utt1", "train", 3),
        ("slt", 0, "utt2", "valid", 5),
    ]
    if include_secondary_speaker:
        utterance_specs.extend(
            [
                ("bdl", 1, "utt1", "train", 4),
                ("bdl", 1, "utt2", "valid", 6),
            ]
        )
    speaker_set = sorted({speaker for speaker, *_ in utterance_specs})
    speaker_to_id = {speaker: speaker_id for speaker, speaker_id, *_ in utterance_specs}
    for index, (speaker, speaker_id, utterance_id, split, frames) in enumerate(
        utterance_specs
    ):
        utterance_dir = aligned_dir / speaker / utterance_id
        utterance_dir.mkdir(parents=True)
        coded_sp = np.arange(frames * 2, dtype=np.float32).reshape(frames, 2) + index
        condition = np.arange(frames * 3, dtype=np.float32).reshape(frames, 3) + 10.0
        world_aux = np.stack(
            [
                np.log1p(np.arange(frames, dtype=np.float32)),
                np.ones(frames, dtype=np.float32),
                np.full(frames, 0.25, dtype=np.float32),
            ],
            axis=1,
        )
        coded_path = utterance_dir / "world_coded_sp.npy"
        condition_path = utterance_dir / "hubert_soft.npy"
        aux_path = utterance_dir / "world_aux.npy"
        np.save(coded_path, coded_sp)
        np.save(condition_path, condition)
        np.save(aux_path, world_aux)
        wav_path = tmp_path / "wav" / speaker / f"{utterance_id}.wav"
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        wav_path.touch()
        rows.append(
            {
                "aligned_condition_paths": {
                    "hubert_soft": condition_path.relative_to(tmp_path).as_posix(),
                    "world_aux": aux_path.relative_to(tmp_path).as_posix(),
                },
                "item_id": f"{speaker}_{utterance_id}",
                "sample_rate": 16000,
                "speaker": speaker,
                "speaker_id": speaker_id,
                "speaker_set": speaker_set,
                "speaker_to_id": speaker_to_id,
                "split": split,
                "utterance_id": utterance_id,
                "wav_path": wav_path.relative_to(tmp_path).as_posix(),
                "world_coded_sp_path": coded_path.relative_to(tmp_path).as_posix(),
                "world_frame_period_ms": 10.0,
            }
        )

    _write_jsonl(cache_dir / "feature_manifest.jsonl", rows)
    _write_json(cache_dir / "feature_summary.json", {"path_root": tmp_path.as_posix()})
    _write_json(
        cache_dir / "feature_statistics.json",
        {
            "train": {
                "features": {
                    "hubert_soft": {"mean": [10.0, 10.0, 10.0], "std": [2.0, 2.0, 2.0]},
                    "world_aux": {"mean": [0.0, 1.0, 0.25], "std": [1.0, 1.0, 1.0]},
                    "world_coded_sp": {"mean": [1.0, 2.0], "std": [2.0, 4.0]},
                }
            }
        },
    )
    return cache_dir


def test_feature_normalizer_round_trips() -> None:
    normalizer = FeatureNormalizer.from_stats({"mean": [1.0, 2.0], "std": [2.0, 4.0]})
    features = np.array([[3.0, 10.0]], dtype=np.float32)

    normalized = normalizer.normalize(features)
    restored = normalizer.denormalize(normalized)

    np.testing.assert_allclose(normalized, [[1.0, 2.0]])
    np.testing.assert_allclose(restored, features)


def test_speech_feature_dataset_loads_normalized_split(tmp_path) -> None:
    cache_dir = _fake_feature_cache(tmp_path)

    dataset = SpeechFeatureDataset(
        cache_dir / "feature_manifest.jsonl",
        condition="hubert_soft",
        split="train",
    )
    sample = dataset[0]

    assert len(dataset) == 1
    assert dataset.index_for_speaker_utterance(speaker="slt", utterance_id="utt1") == 0
    assert sample["speaker"] == "slt"
    assert sample["speaker_id"] == 0
    assert sample["coded_sp"].shape == (2, 3)
    assert sample["condition"].shape == (3, 3)
    torch.testing.assert_close(sample["coded_sp"][:, 0], torch.tensor([-0.5, -0.25]))
    torch.testing.assert_close(sample["condition"][:, 0], torch.tensor([0.0, 0.5, 1.0]))


def test_speech_feature_dataset_filters_by_speakers(tmp_path) -> None:
    cache_dir = _fake_feature_cache(tmp_path, include_secondary_speaker=True)

    dataset = SpeechFeatureDataset(
        cache_dir / "feature_manifest.jsonl",
        condition="hubert_soft",
        split="train",
        speakers=["bdl"],
        normalize=False,
    )
    sample = dataset[0]

    assert len(dataset) == 1
    assert dataset.speakers == ("bdl",)
    assert sample["speaker"] == "bdl"
    assert sample["speaker_id"] == 1
    assert sample["utterance_id"] == "utt1"
    assert dataset.index_for_speaker_utterance(speaker="bdl", utterance_id="utt1") == 0
    with pytest.raises(KeyError):
        dataset.index_for_speaker_utterance(speaker="slt", utterance_id="utt1")


def test_speech_feature_dataset_loads_combined_world_aux_condition(tmp_path) -> None:
    cache_dir = _fake_feature_cache(tmp_path)

    dataset = SpeechFeatureDataset(
        cache_dir / "feature_manifest.jsonl",
        condition=["hubert_soft", "world_aux"],
        split="train",
    )
    sample = dataset[0]

    assert sample["condition_name"] == "hubert_soft+world_aux"
    assert sample["condition"].shape == (6, 3)
    torch.testing.assert_close(
        sample["condition"][:, 0],
        torch.tensor([0.0, 0.5, 1.0, 0.0, 0.0, 0.0]),
    )


def test_speech_feature_dataset_uses_explicit_components_over_extra_combined_path(
    tmp_path,
) -> None:
    cache_dir = _fake_feature_cache(tmp_path)
    manifest_path = cache_dir / "feature_manifest.jsonl"
    rows = read_feature_manifest(manifest_path)
    for row in rows:
        coded_path = tmp_path / row["world_coded_sp_path"]
        frames = np.load(coded_path).shape[0]
        extra_path = coded_path.parent / "prejoined_condition.npy"
        np.save(extra_path, np.full((frames, 6), 999.0, dtype=np.float32))
        row["aligned_condition_paths"]["prejoined_condition"] = extra_path.relative_to(
            tmp_path
        ).as_posix()
    _write_jsonl(manifest_path, rows)

    dataset = SpeechFeatureDataset(
        manifest_path,
        condition=["hubert_soft", "world_aux"],
        split="train",
    )
    sample = dataset[0]

    torch.testing.assert_close(
        sample["condition"][:, 0],
        torch.tensor([0.0, 0.5, 1.0, 0.0, 0.0, 0.0]),
    )


def test_collate_speech_features_pads_to_batch_max_length(tmp_path) -> None:
    cache_dir = _fake_feature_cache(tmp_path)
    dataset = SpeechFeatureDataset(
        cache_dir / "feature_manifest.jsonl",
        condition="hubert_soft",
        normalize=False,
    )

    batch = collate_speech_features([dataset[0], dataset[1]])

    assert batch["coded_sp"].shape == (2, 2, 5)
    assert batch["condition"].shape == (2, 3, 5)
    assert batch["lengths"].tolist() == [3, 5]
    torch.testing.assert_close(
        batch["mask"][0, 0], torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0])
    )
    torch.testing.assert_close(batch["coded_sp"][0, :, 3:], torch.zeros(2, 2))


def test_dataset_loads_manifest_wav_synthesis_features(tmp_path) -> None:
    cache_dir = _fake_feature_cache(tmp_path)
    dataset = SpeechFeatureDataset(
        cache_dir / "feature_manifest.jsonl",
        condition="hubert_soft",
        normalize=False,
    )

    def fake_audio_loader(path, target_sample_rate=None):
        assert path.name == "utt1.wav"
        assert target_sample_rate is None
        return np.zeros(160, dtype=np.float64), 16000

    def fake_world_analyzer(waveform, sample_rate, config):
        assert waveform.shape == (160,)
        assert sample_rate == 16000
        assert config.frame_period_ms == 10.0
        return WorldFeatureBundle(
            f0=np.array([0.0, 100.0, 120.0], dtype=np.float64),
            time_axis=np.array([0.0, 0.01, 0.02], dtype=np.float64),
            spectral_envelope=np.ones((3, 4), dtype=np.float64),
            aperiodicity=np.ones((3, 4), dtype=np.float64),
            coded_sp=np.ones((3, 2), dtype=np.float32),
            coded_ap=np.ones((3, 1), dtype=np.float32),
            sample_rate=16000,
            frame_period_ms=10.0,
        )

    features = dataset.load_synthesis_features(
        0,
        audio_loader=fake_audio_loader,
        world_analyzer=fake_world_analyzer,
    )

    assert features["f0"].shape == (3,)
    assert features["aperiodicity"].shape == (3, 4)
    assert features["fft_size"] == 6
    assert features["frame_period_ms"] == 10.0
    assert features["wav_path"].endswith("utt1.wav")


def test_load_feature_normalizers_requires_existing_split(tmp_path) -> None:
    cache_dir = _fake_feature_cache(tmp_path)

    normalizers = load_feature_normalizers(
        cache_dir / "feature_statistics.json", split="train"
    )

    assert set(normalizers) == {"hubert_soft", "world_aux", "world_coded_sp"}


def test_shift_f0_by_voiced_mean_preserves_unvoiced_frames() -> None:
    shifted = shift_f0_by_voiced_mean(
        np.array([0.0, 100.0, 200.0]),
        source_mean_hz=100.0,
        target_mean_hz=150.0,
    )

    np.testing.assert_allclose(shifted, [0.0, 150.0, 300.0])
