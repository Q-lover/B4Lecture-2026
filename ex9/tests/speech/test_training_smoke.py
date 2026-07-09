"""Speech training and sampling smoke tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import torch
from nf_assignment.speech.dataset import FeatureNormalizer, collate_speech_features
from nf_assignment.speech.features.content import ResampledConditionFeature
from nf_assignment.speech.features.world import WorldFeatureBundle
from nf_assignment.speech.model import build_speech_flow
from nf_assignment.speech.sample import (
    extract_vc_condition,
    fit_world_frames,
    generate_coded_sp,
    synthesize_target_world,
)
from nf_assignment.speech.train import (
    crop_batch_segments,
    speech_nll_loss,
    train_speech_flow,
)
from nf_assignment.training.checkpoints import load_checkpoint
from torch.utils.data import DataLoader


def _sample(length: int) -> dict:
    return {
        "coded_sp": torch.randn(4, length),
        "condition": torch.randn(3, length),
        "condition_name": "fake",
        "length": length,
        "metadata": {"utterance_id": f"utt{length}"},
        "utterance_id": f"utt{length}",
    }


def _write_jsonl(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle)
            handle.write("\n")


def _write_text(path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_cli_feature_cache(tmp_path) -> str:
    cache_dir = tmp_path / "feature_cache"
    rows = []
    for index, split in enumerate(["train", "valid"]):
        utterance_dir = cache_dir / "aligned" / "slt" / f"utt{index}"
        utterance_dir.mkdir(parents=True)
        frames = 4
        coded_sp = np.arange(frames * 2, dtype=np.float32).reshape(frames, 2) + index
        condition = np.arange(frames * 3, dtype=np.float32).reshape(frames, 3) + 10.0
        coded_path = utterance_dir / "world_coded_sp.npy"
        condition_path = utterance_dir / "hubert_soft.npy"
        wav_path = cache_dir / "wav" / "slt" / f"utt{index}.wav"
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        wav_path.touch()
        np.save(coded_path, coded_sp)
        np.save(condition_path, condition)
        rows.append(
            {
                "aligned_condition_paths": {"hubert_soft": condition_path.as_posix()},
                "item_id": f"slt_utt{index}",
                "speaker": "slt",
                "speaker_id": 0,
                "speaker_set": ["slt"],
                "speaker_to_id": {"slt": 0},
                "split": split,
                "utterance_id": f"utt{index}",
                "wav_path": wav_path.as_posix(),
                "world_coded_sp_path": coded_path.as_posix(),
            }
        )
    manifest_path = cache_dir / "feature_manifest.jsonl"
    _write_jsonl(manifest_path, rows)
    return manifest_path.as_posix()


def test_speech_nll_loss_is_finite() -> None:
    torch.manual_seed(0)
    model = build_speech_flow(
        coded_sp_channels=4,
        condition_channels=3,
        hidden_channels=4,
        num_blocks=1,
        num_layers_per_block=2,
        kernel_size=3,
    )
    batch = collate_speech_features([_sample(6), _sample(4)])

    loss = speech_nll_loss(model, batch)

    assert torch.isfinite(loss)


def test_crop_batch_segments_pads_short_items_and_crops_long_items() -> None:
    torch.manual_seed(1)
    batch = collate_speech_features([_sample(6), _sample(3)])

    cropped = crop_batch_segments(batch, segment_frames=4)

    assert cropped["coded_sp"].shape == (2, 4, 4)
    assert cropped["condition"].shape == (2, 3, 4)
    assert cropped["lengths"].tolist() == [4, 3]
    torch.testing.assert_close(
        cropped["mask"][1, 0], torch.tensor([1.0, 1.0, 1.0, 0.0])
    )


def test_crop_batch_segments_zero_keeps_full_utterances() -> None:
    batch = collate_speech_features([_sample(6), _sample(3)])

    cropped = crop_batch_segments(batch, segment_frames=0)

    assert cropped is batch
    assert cropped["coded_sp"].shape == (2, 4, 6)
    assert cropped["condition"].shape == (2, 3, 6)
    assert cropped["lengths"].tolist() == [6, 3]


def test_train_speech_flow_runs_one_step() -> None:
    torch.manual_seed(2)
    model = build_speech_flow(
        coded_sp_channels=4,
        condition_channels=3,
        hidden_channels=4,
        num_blocks=1,
        num_layers_per_block=2,
        kernel_size=3,
    )
    loader = DataLoader(
        [_sample(6), _sample(5)],
        batch_size=2,
        collate_fn=collate_speech_features,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    history = train_speech_flow(
        model,
        optimizer,
        loader,
        device=torch.device("cpu"),
        num_steps=1,
        segment_frames=4,
        log_every=1,
        seed=123,
    )

    assert len(history) == 1
    assert history[0]["step"] == 1
    assert np.isfinite(history[0]["loss"])


def test_train_speech_flow_reports_progress() -> None:
    torch.manual_seed(4)
    model = build_speech_flow(
        coded_sp_channels=4,
        condition_channels=3,
        hidden_channels=4,
        num_blocks=1,
        num_layers_per_block=2,
        kernel_size=3,
    )
    loader = DataLoader(
        [_sample(6), _sample(5)],
        batch_size=2,
        collate_fn=collate_speech_features,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    progress: list[dict[str, float | int]] = []

    history = train_speech_flow(
        model,
        optimizer,
        loader,
        device=torch.device("cpu"),
        num_steps=2,
        segment_frames=4,
        log_every=1,
        seed=123,
        progress_callback=progress.append,
    )

    assert progress == history
    assert [entry["step"] for entry in progress] == [1, 2]


def test_train_speech_script_resumes_checkpoint(tmp_path) -> None:
    repo_dir = __file__
    for _ in range(3):
        repo_dir = os.path.dirname(repo_dir)
    manifest_path = _write_cli_feature_cache(tmp_path)
    data_config = tmp_path / "data.yaml"
    model_config = tmp_path / "model.yaml"
    train_config = tmp_path / "train.yaml"
    first_output = tmp_path / "first"
    resumed_output = tmp_path / "resumed"
    _write_text(
        data_config,
        "splits:\n" "  train: train\n" "  valid: valid\n",
    )
    _write_text(
        model_config,
        "flow: glowtts_style\n"
        "target_channels: 2\n"
        "conditioner: wavenet\n"
        "num_blocks: 1\n"
        "n_split: 2\n"
        "n_sqz: 1\n"
        "hidden_channels: 4\n"
        "kernel_size: 3\n"
        "dilation_rate: 1\n"
        "num_conditioner_layers: 1\n"
        "dropout: 0.1\n"
        "coupling: affine\n",
    )
    _write_text(
        train_config,
        "device: cpu\n"
        "seed: 5\n"
        f"feature_manifest: {manifest_path}\n"
        "condition: hubert_soft\n"
        "speakers: [slt]\n"
        "train_split: train\n"
        "valid_split: valid\n"
        "statistics_split: train\n"
        "normalize: false\n"
        "batch_size: 1\n"
        "segment_frames: 0\n"
        "num_steps: 1\n"
        "optimizer:\n"
        "  name: adam\n"
        "  lr: 0.0002\n"
        "  weight_decay: 0.0\n"
        "log_every: 1\n",
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        os.path.join(repo_dir, "src") + os.pathsep + env.get("PYTHONPATH", "")
    )
    script = os.path.join(repo_dir, "scripts", "train_speech.py")
    common_args = [
        sys.executable,
        script,
        "--data-config",
        data_config.as_posix(),
        "--model-config",
        model_config.as_posix(),
        "--train-config",
        train_config.as_posix(),
        "--device",
        "cpu",
    ]

    subprocess.run(
        [*common_args, "--output-dir", first_output.as_posix()],
        cwd=repo_dir,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            sys.executable,
            script,
            "--resume",
            (first_output / "checkpoint.pt").as_posix(),
            "--output-dir",
            resumed_output.as_posix(),
            "--num-steps",
            "1",
            "--device",
            "cpu",
        ],
        cwd=repo_dir,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    metrics = json.loads((resumed_output / "metrics.json").read_text())
    checkpoint = load_checkpoint(resumed_output / "checkpoint.pt", map_location="cpu")

    assert metrics["resume_previous_steps"] == 1
    assert metrics["total_steps"] == 2
    assert metrics["batch_size"] == 1
    assert metrics["segment_frames"] == 0
    assert metrics["speakers"] == ["slt"]
    assert metrics["feature_manifest"] == manifest_path
    assert checkpoint["metadata"]["resume_previous_steps"] == 1
    assert checkpoint["metadata"]["total_steps"] == 2
    assert checkpoint["metadata"]["condition"] == "hubert_soft"
    assert checkpoint["metadata"]["condition_components"] == ["hubert_soft"]
    assert checkpoint["metadata"]["speakers"] == ["slt"]
    assert "optimizer_state_dict" in checkpoint


def test_generate_coded_sp_respects_mask() -> None:
    torch.manual_seed(3)
    model = build_speech_flow(
        coded_sp_channels=4,
        condition_channels=3,
        hidden_channels=4,
        num_blocks=1,
        num_layers_per_block=2,
        kernel_size=3,
    )
    batch = collate_speech_features([_sample(5)])

    generated = generate_coded_sp(
        model,
        condition=batch["condition"],
        mask=batch["mask"],
        latent_scale=0.0,
    )

    assert generated.shape == batch["coded_sp"].shape
    torch.testing.assert_close(
        generated * (1.0 - batch["mask"]), torch.zeros_like(generated)
    )


def test_extract_vc_condition_uses_source_frames_and_normalizer(monkeypatch) -> None:
    def fake_extract_resampled_condition_features(
        waveform,
        sample_rate,
        *,
        target_frame_count,
        conditions,
        hubert_model=None,
        device=None,
        gpu=0,
    ):
        assert waveform.shape == (160,)
        assert sample_rate == 16000
        assert target_frame_count == 4
        assert conditions == ("hubert_soft",)
        return {
            "hubert_soft": ResampledConditionFeature(
                name="hubert_soft",
                raw=np.ones((2, 3), dtype=np.float32),
                aligned=np.array(
                    [
                        [10.0, 12.0, 14.0],
                        [12.0, 14.0, 16.0],
                        [14.0, 16.0, 18.0],
                        [16.0, 18.0, 20.0],
                    ],
                    dtype=np.float32,
                ),
            )
        }

    monkeypatch.setattr(
        "nf_assignment.speech.sample.extract_resampled_condition_features",
        fake_extract_resampled_condition_features,
    )
    normalizer = FeatureNormalizer(
        mean=np.array([10.0, 10.0, 10.0], dtype=np.float32),
        std=np.array([2.0, 4.0, 8.0], dtype=np.float32),
    )

    result = extract_vc_condition(
        np.zeros(160, dtype=np.float64),
        16000,
        condition="hubert_soft",
        frame_count=4,
        normalizer=normalizer,
        device="cpu",
    )

    assert result["length"] == 4
    assert result["condition"].shape == (1, 3, 4)
    assert result["mask"].shape == (1, 1, 4)
    torch.testing.assert_close(result["mask"], torch.ones(1, 1, 4))
    torch.testing.assert_close(
        result["condition"][0, :, 0],
        torch.tensor([0.0, 0.5, 0.5]),
    )
    np.testing.assert_allclose(result["aligned"][0], [10.0, 12.0, 14.0])


def test_extract_vc_condition_appends_shifted_source_world_aux(monkeypatch) -> None:
    def fake_extract_resampled_condition_features(
        waveform,
        sample_rate,
        *,
        target_frame_count,
        conditions,
        hubert_model=None,
        device=None,
        gpu=0,
    ):
        assert conditions == ("hubert_soft",)
        assert target_frame_count == 4
        return {
            "hubert_soft": ResampledConditionFeature(
                name="hubert_soft",
                raw=np.ones((2, 2), dtype=np.float32),
                aligned=np.array(
                    [
                        [1.0, 2.0],
                        [3.0, 4.0],
                        [5.0, 6.0],
                        [7.0, 8.0],
                    ],
                    dtype=np.float32,
                ),
            )
        }

    monkeypatch.setattr(
        "nf_assignment.speech.sample.extract_resampled_condition_features",
        fake_extract_resampled_condition_features,
    )
    source_world = WorldFeatureBundle(
        f0=np.array([0.0, 100.0, 200.0, 0.0], dtype=np.float64),
        time_axis=np.arange(4, dtype=np.float64) * 0.01,
        spectral_envelope=np.ones((4, 5), dtype=np.float64),
        aperiodicity=np.ones((4, 5), dtype=np.float64),
        coded_sp=np.ones((4, 2), dtype=np.float32),
        coded_ap=np.array([[0.1], [0.2], [0.3], [0.4]], dtype=np.float32),
        sample_rate=16000,
        frame_period_ms=10.0,
    )

    result = extract_vc_condition(
        np.zeros(160, dtype=np.float64),
        16000,
        condition=["hubert_soft", "world_aux"],
        frame_count=4,
        normalizers={
            "hubert_soft": FeatureNormalizer(
                mean=np.array([1.0, 2.0], dtype=np.float32),
                std=np.array([2.0, 2.0], dtype=np.float32),
            ),
            "world_aux": FeatureNormalizer(
                mean=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                std=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            ),
        },
        source_world=source_world,
        target_voiced_mean_f0_hz=300.0,
        device="cpu",
    )

    expected_shifted_f0 = np.array([0.0, 200.0, 400.0, 0.0], dtype=np.float64)
    assert result["uses_world_aux"] is True
    assert result["base_condition"] == "hubert_soft"
    assert result["condition_name"] == "hubert_soft+world_aux"
    assert result["condition_components"] == ["hubert_soft", "world_aux"]
    assert result["condition"].shape == (1, 5, 4)
    np.testing.assert_allclose(result["shifted_f0"], expected_shifted_f0)
    np.testing.assert_allclose(
        result["aligned"][:, :2], [[1, 2], [3, 4], [5, 6], [7, 8]]
    )
    np.testing.assert_allclose(result["aligned"][:, 2], np.log1p(expected_shifted_f0))
    np.testing.assert_allclose(result["aligned"][:, 3], [0.0, 1.0, 1.0, 0.0])
    np.testing.assert_allclose(result["aligned"][:, 4], [0.1, 0.2, 0.3, 0.4])
    torch.testing.assert_close(
        result["condition"][0, :, 1],
        torch.tensor([1.0, 1.0, np.log1p(200.0), 1.0, 0.2], dtype=torch.float32),
    )


def test_fit_world_frames_matches_generated_length() -> None:
    f0 = np.array([0.0, 100.0], dtype=np.float64)
    ap = np.ones((2, 3), dtype=np.float64)

    fitted_f0, fitted_ap = fit_world_frames(f0, ap, frames=4)

    np.testing.assert_allclose(fitted_f0, [0.0, 100.0, 100.0, 100.0])
    assert fitted_ap.shape == (4, 3)


def test_synthesize_target_world_uses_target_f0_and_ap(monkeypatch) -> None:
    captured = {}

    def fake_decode(coded_sp, sample_rate, fft_size):
        captured["decode"] = (coded_sp.copy(), sample_rate, fft_size)
        return np.full((3, 5), 2.0, dtype=np.float64)

    def fake_synthesize(
        f0, spectral_envelope, aperiodicity, sample_rate, *, frame_period_ms
    ):
        captured["synthesize"] = (
            f0.copy(),
            spectral_envelope.copy(),
            aperiodicity.copy(),
            sample_rate,
            frame_period_ms,
        )
        return np.arange(4, dtype=np.float64)

    monkeypatch.setattr(
        "nf_assignment.speech.sample.decode_spectral_envelope", fake_decode
    )
    monkeypatch.setattr("nf_assignment.speech.sample.synthesize_world", fake_synthesize)

    coded_sp = np.ones((3, 4), dtype=np.float32)
    result = synthesize_target_world(
        coded_sp,
        synthesis_features={
            "aperiodicity": np.arange(10, dtype=np.float64).reshape(2, 5),
            "f0": np.array([0.0, 120.0], dtype=np.float64),
            "fft_size": 8,
            "frame_period_ms": 10.0,
            "sample_rate": 16000,
        },
    )

    np.testing.assert_allclose(captured["decode"][0], coded_sp)
    assert captured["decode"][1:] == (16000, 8)
    np.testing.assert_allclose(captured["synthesize"][0], [0.0, 120.0, 120.0])
    np.testing.assert_allclose(
        captured["synthesize"][2],
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [5, 6, 7, 8, 9]],
    )
    assert captured["synthesize"][3:] == (16000, 10.0)
    np.testing.assert_allclose(result["waveform"], [0.0, 1.0, 2.0, 3.0])
