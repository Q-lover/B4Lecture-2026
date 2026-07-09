import csv
import json
import wave
from importlib import import_module
from pathlib import Path

from nf_assignment.speech.data import (
    build_cmu_arctic_inventory_manifest,
    normalize_speaker_list,
    parse_cmu_arctic_prompts,
    select_vc_sample_items,
)


def _write_wav(path: Path, *, sample_rate: int = 16000, frames: int = 160) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * frames)


def test_parse_cmu_arctic_prompts() -> None:
    prompts = parse_cmu_arctic_prompts(
        '( arctic_a0001 "Author of the danger trail." )\n'
        '( arctic_a0002 "He said \\"hello\\"." )\n'
    )

    assert prompts == {
        "arctic_a0001": "Author of the danger trail.",
        "arctic_a0002": 'He said "hello".',
    }


def test_prepare_data_console_target_is_importable() -> None:
    module = import_module("nf_assignment.speech.prepare_data")

    assert callable(module.main)


def test_normalize_speaker_list_expands_all_and_removes_duplicates() -> None:
    speakers = normalize_speaker_list(("bdl", "all", "bdl"))

    assert speakers[0] == "bdl"
    assert "slt" in speakers
    assert len(speakers) == len(set(speakers))


def test_build_cmu_arctic_inventory_manifest_uses_speaker_rows_and_shared_splits(
    tmp_path,
) -> None:
    root = tmp_path / "cmu_arctic"
    prompt_lines = []
    for index in range(1, 7):
        utterance_id = f"arctic_a{index:04d}"
        prompt_lines.append(f'( {utterance_id} "Sentence {index}." )')
        _write_wav(root / "cmu_us_bdl_arctic" / "wav" / f"{utterance_id}.wav")
        _write_wav(root / "cmu_us_slt_arctic" / "wav" / f"{utterance_id}.wav")
    (root / "cmuarctic.data").write_text("\n".join(prompt_lines), encoding="utf-8")

    manifest_path = tmp_path / "inventory.csv"
    split_path = tmp_path / "splits.json"
    summary_path = tmp_path / "summary.json"
    summary = build_cmu_arctic_inventory_manifest(
        root=root,
        manifest_path=manifest_path,
        split_path=split_path,
        summary_path=summary_path,
        speakers=("bdl", "slt"),
        max_utterances=6,
        valid_count=1,
        test_count=1,
        seed=7,
        path_base=tmp_path,
    )

    with manifest_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    split_doc = json.loads(split_path.read_text(encoding="utf-8"))

    assert len(rows) == 12
    assert summary["inventory_rows"] == 12
    assert summary["split_counts"] == {"train": 4, "valid": 1, "test": 1}
    assert summary["mismatched_sample_rate_rows"] == 0
    assert {row["speaker"] for row in rows} == {"bdl", "slt"}
    assert {row["speaker_id"] for row in rows} == {"0", "1"}
    assert "speaker_set" not in rows[0]
    assert "speaker_to_id" not in rows[0]
    assert all(row["wav_path"].startswith("cmu_arctic/") for row in rows)
    assert len({(row["speaker"], row["utterance_id"]) for row in rows}) == len(rows)
    assert split_doc["split_kind"] == "utterance_id"
    assert sorted(
        split_doc["splits"]["train"]
        + split_doc["splits"]["valid"]
        + split_doc["splits"]["test"]
    ) == sorted({row["utterance_id"] for row in rows})


def test_build_cmu_arctic_inventory_manifest_keeps_non_common_speaker_utterances(
    tmp_path,
) -> None:
    root = tmp_path / "cmu_arctic"
    prompt_lines = []
    for index in range(1, 7):
        utterance_id = f"arctic_a{index:04d}"
        prompt_lines.append(f'( {utterance_id} "Sentence {index}." )')
        _write_wav(root / "cmu_us_bdl_arctic" / "wav" / f"{utterance_id}.wav")
        if index <= 4:
            _write_wav(root / "cmu_us_slt_arctic" / "wav" / f"{utterance_id}.wav")
    (root / "cmuarctic.data").write_text("\n".join(prompt_lines), encoding="utf-8")

    manifest_path = tmp_path / "inventory.csv"
    split_path = tmp_path / "splits.json"
    summary_path = tmp_path / "summary.json"
    summary = build_cmu_arctic_inventory_manifest(
        root=root,
        manifest_path=manifest_path,
        split_path=split_path,
        summary_path=summary_path,
        speakers=("bdl", "slt"),
        max_utterances=6,
        valid_count=1,
        test_count=1,
        seed=7,
        path_base=tmp_path,
    )

    with manifest_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    split_doc = json.loads(split_path.read_text(encoding="utf-8"))

    assert len(rows) == 10
    assert summary["inventory_rows"] == 10
    assert summary["selected_utterances"] == 6
    assert summary["union_utterance_count"] == 6
    assert summary["common_utterance_count"] == 4
    assert summary["selected_utterances_per_speaker"] == {"bdl": 6, "slt": 4}
    assert split_doc["utterance_counts_by_speaker"] == {"bdl": 6, "slt": 4}
    assert ("bdl", "arctic_a0006") in {
        (row["speaker"], row["utterance_id"]) for row in rows
    }
    assert ("slt", "arctic_a0006") not in {
        (row["speaker"], row["utterance_id"]) for row in rows
    }


def test_select_vc_sample_items_resolves_source_and_target_from_inventory() -> None:
    rows = [
        {
            "speaker": "bdl",
            "split": "valid",
            "text": "Sentence 1.",
            "utterance_id": "arctic_a0001",
            "wav_path": "bdl/arctic_a0001.wav",
        },
        {
            "speaker": "slt",
            "split": "valid",
            "text": "Sentence 1.",
            "utterance_id": "arctic_a0001",
            "wav_path": "slt/arctic_a0001.wav",
        },
        {
            "speaker": "bdl",
            "split": "train",
            "text": "Sentence 2.",
            "utterance_id": "arctic_a0002",
            "wav_path": "bdl/arctic_a0002.wav",
        },
        {
            "speaker": "slt",
            "split": "train",
            "text": "Sentence 2.",
            "utterance_id": "arctic_a0002",
            "wav_path": "slt/arctic_a0002.wav",
        },
    ]

    selected = select_vc_sample_items(
        inventory_rows=rows,
        source_speaker="bdl",
        target_speaker="slt",
        split="valid",
        utterance_ids=None,
        max_items=2,
    )

    assert len(selected) == 1
    assert selected[0]["sample_id"] == "bdl_to_slt_arctic_a0001"
    assert selected[0]["source_wav"] == "bdl/arctic_a0001.wav"
    assert selected[0]["target_wav"] == "slt/arctic_a0001.wav"


def test_select_vc_sample_items_uses_pair_intersection_for_automatic_selection() -> (
    None
):
    rows = [
        {
            "speaker": "bdl",
            "split": "valid",
            "utterance_id": "utt1",
            "wav_path": "bdl/utt1.wav",
        },
        {
            "speaker": "slt",
            "split": "valid",
            "utterance_id": "utt1",
            "wav_path": "slt/utt1.wav",
        },
        {
            "speaker": "bdl",
            "split": "valid",
            "utterance_id": "utt2",
            "wav_path": "bdl/utt2.wav",
        },
        {
            "speaker": "slt",
            "split": "valid",
            "utterance_id": "utt3",
            "wav_path": "slt/utt3.wav",
        },
        {
            "speaker": "bdl",
            "split": "train",
            "utterance_id": "utt4",
            "wav_path": "bdl/utt4.wav",
        },
        {
            "speaker": "slt",
            "split": "train",
            "utterance_id": "utt4",
            "wav_path": "slt/utt4.wav",
        },
    ]

    selected = select_vc_sample_items(
        inventory_rows=rows,
        source_speaker="bdl",
        target_speaker="slt",
        split="valid",
        utterance_ids=None,
        max_items=10,
    )

    assert [item["utterance_id"] for item in selected] == ["utt1"]


def test_select_vc_sample_items_accepts_explicit_utterance_ids() -> None:
    rows = [
        {
            "speaker": "bdl",
            "split": "train",
            "utterance_id": "utt2",
            "wav_path": "bdl/utt2.wav",
        },
        {
            "speaker": "slt",
            "split": "train",
            "utterance_id": "utt2",
            "wav_path": "slt/utt2.wav",
        },
    ]

    selected = select_vc_sample_items(
        inventory_rows=rows,
        source_speaker="bdl",
        target_speaker="slt",
        split="valid",
        utterance_ids=["utt2"],
        max_items=1,
    )

    assert selected[0]["utterance_id"] == "utt2"
