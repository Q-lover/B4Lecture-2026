"""CMU ARCTIC manifest and split utilities."""

from __future__ import annotations

import random
import re
import tarfile
import urllib.request
import wave
from pathlib import Path
from typing import Any

from nf_assignment.utils.io import ensure_dir, write_csv_rows, write_json

CMU_ARCTIC_PAGE_URL = "http://festvox.org/cmu_arctic/"
CMU_ARCTIC_PROMPTS_URL = "http://festvox.org/cmu_arctic/cmuarctic.data"
CMU_ARCTIC_ARCHIVE_BASE_URL = "http://festvox.org/cmu_arctic/packed"
KNOWN_CMU_ARCTIC_SPEAKERS = (
    "aew",
    "ahw",
    "aup",
    "awb",
    "axb",
    "bdl",
    "clb",
    "eey",
    "fem",
    "gka",
    "jmk",
    "ksp",
    "ljm",
    "lnh",
    "rms",
    "rxr",
    "slp",
    "slt",
)
DEFAULT_MAX_UTTERANCES = 1132
DEFAULT_VALID_COUNT = 100
DEFAULT_TEST_COUNT = 32
DEFAULT_SEED = 20260625
PROMPT_RE = re.compile(
    r'^\(\s*(?P<utterance_id>arctic_[ab]\d{4})\s+"(?P<text>.*)"\s*\)\s*$'
)


def speaker_archive_url(speaker: str) -> str:
    """Return the official Festvox packed archive URL for a CMU ARCTIC speaker."""

    return f"{CMU_ARCTIC_ARCHIVE_BASE_URL}/cmu_us_{speaker}_arctic.tar.bz2"


def normalize_speaker_list(speakers: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Normalize speaker names while preserving order and rejecting duplicates."""

    normalized: list[str] = []
    seen: set[str] = set()
    for speaker in speakers:
        value = speaker.strip().lower()
        if not value:
            continue
        if value == "all":
            expanded = KNOWN_CMU_ARCTIC_SPEAKERS
        else:
            expanded = (value,)
        for item in expanded:
            if item in seen:
                continue
            normalized.append(item)
            seen.add(item)
    if not normalized:
        raise ValueError("at least one speaker must be specified.")
    return tuple(normalized)


def parse_cmu_arctic_prompts(text: str) -> dict[str, str]:
    """Parse the ``cmuarctic.data`` prompt file into utterance text by ID."""

    prompts: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = PROMPT_RE.match(stripped)
        if match is None:
            continue
        utterance_id = match.group("utterance_id")
        prompts[utterance_id] = match.group("text").replace(r"\"", '"')
    return prompts


def load_cmu_arctic_prompts(path: str | Path) -> dict[str, str]:
    """Load the prompt list distributed by CMU ARCTIC."""

    return parse_cmu_arctic_prompts(Path(path).read_text(encoding="utf-8"))


def download_file(url: str, destination: str | Path) -> Path:
    """Download ``url`` to ``destination`` if the file is not already present."""

    destination = Path(destination)
    ensure_dir(destination.parent)
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    with urllib.request.urlopen(url, timeout=60) as response:
        with destination.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
    return destination


def _safe_extract_tar(archive_path: Path, output_dir: Path) -> None:
    """Extract a tar archive while rejecting members that escape ``output_dir``."""

    output_dir = output_dir.resolve()
    with tarfile.open(archive_path) as archive:
        members = archive.getmembers()
        for member in members:
            target = (output_dir / member.name).resolve()
            if output_dir not in target.parents and target != output_dir:
                raise ValueError(f"Unsafe archive member path: {member.name}")
        archive.extractall(output_dir)


def speaker_directory(root: str | Path, speaker: str) -> Path | None:
    """Find an extracted CMU ARCTIC speaker directory under ``root``."""

    root = Path(root)
    candidates = [
        root / f"cmu_us_{speaker}_arctic",
        root / "cmu_arctic" / f"cmu_us_{speaker}_arctic",
        root / "cmu_us" / f"{speaker}_arctic",
    ]
    for candidate in candidates:
        if (candidate / "wav").is_dir():
            return candidate
    matches = sorted(root.glob(f"**/cmu_us_{speaker}_arctic/wav"))
    if matches:
        return matches[0].parent
    return None


def prepare_speaker_archive(
    root: str | Path,
    speaker: str,
    *,
    download: bool = False,
    download_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Ensure a speaker archive is available and return factual preparation metadata."""

    root = Path(root)
    ensure_dir(root)
    archive_url = speaker_archive_url(speaker)
    archive_path = (
        Path(download_dir or root / "archives") / archive_url.rsplit("/", 1)[-1]
    )
    downloaded = False
    extracted = False

    before_dir = speaker_directory(root, speaker)
    if before_dir is None and download:
        existed = archive_path.exists() and archive_path.stat().st_size > 0
        download_file(archive_url, archive_path)
        downloaded = not existed
        _safe_extract_tar(archive_path, root)
        extracted = True

    after_dir = speaker_directory(root, speaker)
    wav_count = len(find_speaker_wavs(root, speaker)) if after_dir is not None else 0
    return {
        "archive_path": str(archive_path),
        "archive_url": archive_url,
        "downloaded": downloaded,
        "extracted": extracted,
        "speaker": speaker,
        "speaker_dir": str(after_dir) if after_dir is not None else None,
        "wav_count": wav_count,
    }


def find_speaker_wavs(root: str | Path, speaker: str) -> dict[str, Path]:
    """Return CMU ARCTIC wav files for a speaker keyed by utterance ID."""

    directory = speaker_directory(root, speaker)
    if directory is None:
        return {}
    wavs: dict[str, Path] = {}
    for path in sorted((directory / "wav").glob("arctic_*.wav")):
        wavs[path.stem] = path
    return wavs


def wav_metadata(path: str | Path) -> dict[str, Any]:
    """Read sample rate, channel count, frame count, and duration from a WAV file."""

    path = Path(path)
    with wave.open(str(path), "rb") as handle:
        sample_rate = handle.getframerate()
        frames = handle.getnframes()
        channels = handle.getnchannels()
    return {
        "channels": channels,
        "duration_sec": frames / sample_rate,
        "frames": frames,
        "sample_rate": sample_rate,
    }


def _relative_path(path: Path, base: Path | None) -> str:
    """Return ``path`` relative to ``base`` when possible."""

    if base is None:
        return path.as_posix()
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _split_utterance_ids(
    utterance_ids: list[str],
    *,
    valid_count: int,
    test_count: int,
    seed: int,
) -> dict[str, list[str]]:
    """Split utterance IDs into train, valid, and test ID lists."""

    if valid_count < 0 or test_count < 0:
        raise ValueError("valid_count and test_count must be non-negative.")
    if valid_count + test_count >= len(utterance_ids):
        raise ValueError(
            "valid_count + test_count must be smaller than available utterances."
        )
    shuffled = list(utterance_ids)
    random.Random(seed).shuffle(shuffled)
    valid = sorted(shuffled[:valid_count])
    test = sorted(shuffled[valid_count : valid_count + test_count])
    held_out = set(valid) | set(test)
    train = sorted(
        utterance_id for utterance_id in shuffled if utterance_id not in held_out
    )
    return {"train": train, "valid": valid, "test": test}


def _speaker_to_id(speakers: tuple[str, ...]) -> dict[str, int]:
    """Assign stable integer IDs from speaker order."""

    return {speaker: index for index, speaker in enumerate(speakers)}


def build_cmu_arctic_inventory_manifest(
    *,
    root: str | Path,
    manifest_path: str | Path,
    split_path: str | Path,
    summary_path: str | Path,
    speakers: tuple[str, ...] | list[str],
    prompts_path: str | Path | None = None,
    max_utterances: int = DEFAULT_MAX_UTTERANCES,
    valid_count: int = DEFAULT_VALID_COUNT,
    test_count: int = DEFAULT_TEST_COUNT,
    seed: int = DEFAULT_SEED,
    expected_sample_rate: int = 16000,
    path_base: str | Path | None = None,
) -> dict[str, Any]:
    """Build a speaker-unit CMU ARCTIC wav inventory manifest.

    Each output row describes one existing speaker utterance. The train/valid/test
    split is assigned by utterance ID over the selected utterance union and then
    reused for every speaker row that has that utterance, so parallel utterances
    do not leak across splits when the inventory is used for multi-speaker
    training. Missing speaker/utterance pairs are kept missing instead of forcing
    an all-speaker intersection.
    """

    root = Path(root)
    speaker_set = normalize_speaker_list(tuple(speakers))
    prompts_file = (
        Path(prompts_path) if prompts_path is not None else root / "cmuarctic.data"
    )
    prompts = load_cmu_arctic_prompts(prompts_file)
    wavs_by_speaker = {
        speaker: find_speaker_wavs(root, speaker) for speaker in speaker_set
    }
    missing_speakers = [
        speaker for speaker, wavs in wavs_by_speaker.items() if not wavs
    ]
    if missing_speakers:
        raise ValueError(
            f"No CMU ARCTIC wav files found for speakers: {missing_speakers}"
        )

    if max_utterances <= 0:
        raise ValueError("max_utterances must be positive.")
    selected_by_speaker = {
        speaker: sorted(set(wavs) & set(prompts))[:max_utterances]
        for speaker, wavs in wavs_by_speaker.items()
    }
    empty_speakers = [
        speaker for speaker, utterances in selected_by_speaker.items() if not utterances
    ]
    if empty_speakers:
        raise ValueError(
            f"No prompted CMU ARCTIC wav files found for speakers: {empty_speakers}"
        )
    selected_sets = {
        speaker: set(utterances) for speaker, utterances in selected_by_speaker.items()
    }
    common_utterances = sorted(set.intersection(*selected_sets.values()))
    selected = sorted(set.union(*selected_sets.values()))
    if not selected:
        raise ValueError("No CMU ARCTIC utterances matched the requested speakers.")
    splits = _split_utterance_ids(
        selected,
        valid_count=valid_count,
        test_count=test_count,
        seed=seed,
    )
    split_by_utterance = {
        utterance_id: split for split, ids in splits.items() for utterance_id in ids
    }
    speaker_ids = _speaker_to_id(speaker_set)
    base = Path(path_base) if path_base is not None else None

    rows: list[dict[str, Any]] = []
    mismatched_sample_rate = 0
    for speaker in speaker_set:
        for utterance_id in selected_by_speaker[speaker]:
            wav_path = wavs_by_speaker[speaker][utterance_id]
            metadata = wav_metadata(wav_path)
            if metadata["sample_rate"] != expected_sample_rate:
                mismatched_sample_rate += 1
            rows.append(
                {
                    "dataset": "cmu_arctic",
                    "duration_sec": round(float(metadata["duration_sec"]), 6),
                    "sample_rate": int(metadata["sample_rate"]),
                    "speaker": speaker,
                    "speaker_id": speaker_ids[speaker],
                    "split": split_by_utterance[utterance_id],
                    "text": prompts[utterance_id],
                    "utterance_id": utterance_id,
                    "wav_path": _relative_path(wav_path, base),
                }
            )

    split_document = {
        "dataset": "cmu_arctic",
        "seed": seed,
        "speaker_set": list(speaker_set),
        "speaker_to_id": speaker_ids,
        "split_kind": "utterance_id",
        "splits": splits,
        "utterance_counts_by_speaker": {
            speaker: len(utterances)
            for speaker, utterances in sorted(selected_by_speaker.items())
        },
    }
    row_split_counts = {
        split_name: sum(1 for row in rows if row["split"] == split_name)
        for split_name in ("train", "valid", "test")
    }
    summary = {
        "archive_urls": {
            speaker: speaker_archive_url(speaker) for speaker in speaker_set
        },
        "common_utterance_count": len(common_utterances),
        "dataset": "cmu_arctic",
        "expected_sample_rate": expected_sample_rate,
        "inventory_rows": len(rows),
        "manifest_path": str(manifest_path),
        "max_utterances": max_utterances,
        "max_utterances_per_speaker": max_utterances,
        "mismatched_sample_rate_rows": mismatched_sample_rate,
        "prompt_count": len(prompts),
        "raw_root": str(root),
        "row_split_counts": row_split_counts,
        "selected_utterances": len(selected),
        "speaker_set": list(speaker_set),
        "speaker_to_id": speaker_ids,
        "speaker_wav_counts": {
            speaker: len(wavs) for speaker, wavs in sorted(wavs_by_speaker.items())
        },
        "selected_utterances_per_speaker": {
            speaker: len(utterances)
            for speaker, utterances in sorted(selected_by_speaker.items())
        },
        "split_counts": {name: len(ids) for name, ids in splits.items()},
        "split_kind": "utterance_id",
        "test_count": test_count,
        "union_utterance_count": len(selected),
        "valid_count": valid_count,
    }

    write_csv_rows(
        manifest_path,
        rows,
        [
            "dataset",
            "speaker",
            "speaker_id",
            "utterance_id",
            "split",
            "wav_path",
            "text",
            "sample_rate",
            "duration_sec",
        ],
    )
    write_json(split_path, split_document)
    write_json(summary_path, summary)
    return summary


def select_vc_sample_items(
    *,
    inventory_rows: list[dict[str, Any]],
    source_speaker: str,
    target_speaker: str,
    split: str,
    utterance_ids: list[str] | tuple[str, ...] | None,
    max_items: int,
) -> list[dict[str, Any]]:
    """Select VC sample items from a speaker-unit wav inventory.

    If ``utterance_ids`` is omitted, this selects utterances available for both
    source and target speakers in the requested split.
    """

    if max_items <= 0:
        raise ValueError("max_items must be positive.")
    source_speaker = source_speaker.strip().lower()
    target_speaker = target_speaker.strip().lower()
    row_by_key = {
        (str(row.get("speaker")), str(row.get("utterance_id"))): row
        for row in inventory_rows
    }
    if utterance_ids is None:
        source_ids = {
            utterance_id
            for (speaker, utterance_id), row in row_by_key.items()
            if speaker == source_speaker
            and (split == "all" or row.get("split") == split)
        }
        target_ids = {
            utterance_id
            for (speaker, utterance_id), row in row_by_key.items()
            if speaker == target_speaker
            and (split == "all" or row.get("split") == split)
        }
        selected_ids = sorted(source_ids & target_ids)[:max_items]
    else:
        selected_ids = [
            utterance_id.strip() for utterance_id in utterance_ids if utterance_id
        ]

    if not selected_ids:
        raise ValueError(
            "No VC sample utterances matched "
            f"source={source_speaker}, target={target_speaker}, split={split}."
        )

    items: list[dict[str, Any]] = []
    for utterance_id in selected_ids:
        source_key = (source_speaker, utterance_id)
        target_key = (target_speaker, utterance_id)
        if source_key not in row_by_key:
            raise KeyError(f"source inventory row is missing: {source_key}")
        if target_key not in row_by_key:
            raise KeyError(f"target inventory row is missing: {target_key}")
        source_row = row_by_key[source_key]
        target_row = row_by_key[target_key]
        items.append(
            {
                "sample_id": f"{source_speaker}_to_{target_speaker}_{utterance_id}",
                "source_row": source_row,
                "source_speaker": source_speaker,
                "source_wav": source_row["wav_path"],
                "target_row": target_row,
                "target_speaker": target_speaker,
                "target_wav": target_row["wav_path"],
                "text": target_row.get("text", source_row.get("text")),
                "utterance_id": utterance_id,
            }
        )
    return items
