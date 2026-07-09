"""Prepare CMU ARCTIC speaker inventories."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from nf_assignment.speech.data import (
    CMU_ARCTIC_PROMPTS_URL,
    DEFAULT_MAX_UTTERANCES,
    DEFAULT_SEED,
    DEFAULT_TEST_COUNT,
    DEFAULT_VALID_COUNT,
    build_cmu_arctic_inventory_manifest,
    download_file,
    normalize_speaker_list,
    prepare_speaker_archive,
)
from nf_assignment.utils.io import load_yaml


def _nested(config: dict[str, Any], *keys: str, default=None):
    """Read a nested config value with a fallback default."""

    value = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _parse_speakers(text: str | None) -> tuple[str, ...] | None:
    """Parse an optional comma-separated speaker list."""

    if text is None:
        return None
    return normalize_speaker_list(tuple(part.strip() for part in text.split(",")))


def _config_speakers(value, *, default: tuple[str, ...]) -> tuple[str, ...]:
    """Normalize a speaker list from YAML config values."""

    if value is None:
        return default
    if isinstance(value, str):
        return normalize_speaker_list((value,))
    if isinstance(value, (list, tuple)):
        return normalize_speaker_list(tuple(str(item) for item in value))
    raise TypeError(
        f"speaker list must be a string or sequence, got {type(value).__name__}"
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for CMU ARCTIC data preparation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/speech/data.yaml")
    parser.add_argument("--root", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--split-json", default=None)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument(
        "--speakers", default=None, help="Comma-separated inventory speakers."
    )
    parser.add_argument(
        "--download-speakers",
        default=None,
        help="Comma-separated speakers to download, or 'all'. Defaults to configured speakers.",
    )
    parser.add_argument("--max-utterances", type=int, default=None)
    parser.add_argument("--valid-count", type=int, default=None)
    parser.add_argument("--test-count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--expected-sample-rate", type=int, default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--download-dir", default=None)
    return parser.parse_args()


def main() -> None:
    """Download/extract CMU ARCTIC data as requested and write inventory files."""

    args = parse_args()
    config = load_yaml(args.config)

    root = Path(args.root or config.get("root", "data/cmu_arctic"))
    manifest_path = Path(
        args.manifest
        or config.get("manifest", "data/manifests/cmu_arctic_inventory.csv")
    )
    split_path = Path(args.split_json or "data/manifests/cmu_arctic_splits.json")
    summary_path = Path(args.summary_json or "data/manifests/cmu_arctic_summary.json")
    inventory_speakers = _parse_speakers(args.speakers) or _config_speakers(
        _nested(config, "speakers", "inventory", default=None),
        default=("bdl", "slt"),
    )
    download_speakers = _parse_speakers(args.download_speakers) or _config_speakers(
        _nested(config, "speakers", "download", default=None),
        default=inventory_speakers,
    )
    max_utterances = int(
        args.max_utterances
        if args.max_utterances is not None
        else _nested(config, "subset", "max_utterances", default=DEFAULT_MAX_UTTERANCES)
    )
    valid_count = int(
        args.valid_count
        if args.valid_count is not None
        else _nested(config, "subset", "valid_count", default=DEFAULT_VALID_COUNT)
    )
    test_count = int(
        args.test_count
        if args.test_count is not None
        else _nested(config, "subset", "test_count", default=DEFAULT_TEST_COUNT)
    )
    seed = int(
        args.seed
        if args.seed is not None
        else _nested(config, "subset", "seed", default=DEFAULT_SEED)
    )
    expected_sample_rate = int(
        args.expected_sample_rate
        if args.expected_sample_rate is not None
        else config.get("sample_rate", 16000)
    )

    print(f"root={root}")
    print(f"download={args.download}")
    print(f"download_speakers={','.join(download_speakers)}")
    print(f"inventory_speakers={','.join(inventory_speakers)}")
    print(f"max_utterances={max_utterances}")
    print(f"valid_count={valid_count}")
    print(f"test_count={test_count}")
    print(f"seed={seed}")

    if args.download:
        root.mkdir(parents=True, exist_ok=True)
        prompts_path = root / "cmuarctic.data"
        download_file(CMU_ARCTIC_PROMPTS_URL, prompts_path)
        print(f"prompts_path={prompts_path}")
        for speaker in download_speakers:
            metadata = prepare_speaker_archive(
                root,
                speaker,
                download=True,
                download_dir=args.download_dir,
            )
            print(
                "speaker_prepared="
                + ",".join(
                    [
                        f"speaker:{metadata['speaker']}",
                        f"wav_count:{metadata['wav_count']}",
                        f"downloaded:{metadata['downloaded']}",
                        f"extracted:{metadata['extracted']}",
                        f"archive:{metadata['archive_path']}",
                    ]
                )
            )

    summary = build_cmu_arctic_inventory_manifest(
        root=root,
        manifest_path=manifest_path,
        split_path=split_path,
        summary_path=summary_path,
        speakers=inventory_speakers,
        max_utterances=max_utterances,
        valid_count=valid_count,
        test_count=test_count,
        seed=seed,
        expected_sample_rate=expected_sample_rate,
        path_base=Path.cwd(),
    )
    print(f"manifest_path={manifest_path}")
    print(f"split_json={split_path}")
    print(f"summary_json={summary_path}")
    print(f"selected_utterances={summary['selected_utterances']}")
    print(f"common_utterances={summary['common_utterance_count']}")
    print(
        f"selected_utterances_per_speaker={summary['selected_utterances_per_speaker']}"
    )
    print(f"inventory_rows={summary['inventory_rows']}")
    print(f"split_counts={summary['split_counts']}")
    print(f"row_split_counts={summary['row_split_counts']}")
    print(f"mismatched_sample_rate_rows={summary['mismatched_sample_rate_rows']}")


if __name__ == "__main__":
    main()
