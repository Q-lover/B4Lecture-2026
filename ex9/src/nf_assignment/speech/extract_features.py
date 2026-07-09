"""Dump WORLD and WORLD-aligned condition features from a speech manifest."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from nf_assignment.speech.data import normalize_speaker_list
from nf_assignment.speech.features.content import (
    ResampledConditionFeature,
    extract_resampled_condition_features,
    load_hubert_soft_model,
    resolve_device,
)
from nf_assignment.speech.features.world import (
    WorldFeatureConfig,
    analyze_world,
    load_mono_audio,
    world_aux_features,
)
from nf_assignment.utils.io import (
    ensure_dir,
    load_yaml,
    read_csv_rows,
    write_csv_rows,
    write_json,
    write_jsonl,
)


def _nested(config: dict[str, Any], *keys: str, default=None):
    """Read a nested config value with a fallback default."""

    value = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _parse_conditions(text: str) -> tuple[str, ...]:
    """Parse a comma-separated list of feature component names."""

    values = tuple(part.strip() for part in text.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated condition list")
    return values


def _parse_speakers(text: str) -> tuple[str, ...]:
    """Parse and normalize a comma-separated speaker list."""

    return normalize_speaker_list(tuple(part.strip() for part in text.split(",")))


def read_manifest_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read a flat CSV inventory manifest or a legacy JSONL manifest."""

    path = Path(path)
    if path.suffix == ".csv":
        return read_csv_rows(path)
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def resolve_manifest_path(path_root: str | Path, value: str) -> Path:
    """Resolve a manifest path that may be relative to ``path_root``."""

    path = Path(value)
    if path.is_absolute():
        return path
    return Path(path_root) / path


def select_manifest_rows(
    rows: list[dict[str, Any]],
    *,
    split: str,
    speakers: tuple[str, ...] | None = None,
    utterance_id: str | None,
    max_utterances: int,
) -> list[dict[str, Any]]:
    """Select inventory rows for feature dumping.

    ``max_utterances`` is applied independently to each selected speaker.
    """

    selected = (
        rows if split == "all" else [row for row in rows if row.get("split") == split]
    )
    if speakers is not None:
        speaker_set = set(speakers)
        selected = [row for row in selected if row.get("speaker") in speaker_set]
    if utterance_id is not None:
        selected = [row for row in selected if row.get("utterance_id") == utterance_id]
    if max_utterances <= 0:
        raise ValueError("max_utterances must be positive.")

    selected_per_speaker: list[dict[str, Any]] = []
    utterance_counts: dict[str, int] = {}
    for row in selected:
        speaker = str(row.get("speaker", ""))
        count = utterance_counts.get(speaker, 0)
        if count >= max_utterances:
            continue
        selected_per_speaker.append(row)
        utterance_counts[speaker] = count + 1
    selected = selected_per_speaker
    if not selected:
        raise ValueError("No manifest rows matched the requested selection.")
    return selected


def _speaker_set_from_rows(rows: list[dict[str, Any]]) -> list[str]:
    """Return sorted speaker names present in manifest rows."""

    return sorted({str(row.get("speaker")) for row in rows if row.get("speaker")})


def _speaker_to_id_from_rows(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Return speaker-to-ID metadata found in manifest rows."""

    mapping: dict[str, int] = {}
    for row in rows:
        speaker = row.get("speaker")
        speaker_id = row.get("speaker_id")
        if speaker is None or speaker_id in (None, ""):
            continue
        mapping[str(speaker)] = int(speaker_id)
    return mapping


class ChannelStats:
    """Streaming channel-wise statistics for frame features."""

    def __init__(self) -> None:
        """Initialize empty streaming accumulators."""

        self.channels: int | None = None
        self.frame_count = 0
        self.utterance_count = 0
        self.sum: np.ndarray | None = None
        self.sumsq: np.ndarray | None = None
        self.minimum: np.ndarray | None = None
        self.maximum: np.ndarray | None = None

    def update(self, features: np.ndarray) -> None:
        """Update channel statistics from one utterance feature matrix.

        Args:
            features: Array shaped ``[frames, channels]``.
        """

        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2:
            raise ValueError(
                f"expected features with shape [frames, channels], got {features.shape}"
            )
        if features.shape[0] == 0:
            raise ValueError("features must contain at least one frame.")
        if self.channels is None:
            self.channels = int(features.shape[1])
            self.sum = np.zeros(self.channels, dtype=np.float64)
            self.sumsq = np.zeros(self.channels, dtype=np.float64)
            self.minimum = np.full(self.channels, np.inf, dtype=np.float64)
            self.maximum = np.full(self.channels, -np.inf, dtype=np.float64)
        if features.shape[1] != self.channels:
            raise ValueError(
                f"expected {self.channels} channels, got {features.shape[1]}"
            )

        self.frame_count += int(features.shape[0])
        self.utterance_count += 1
        self.sum += np.sum(features, axis=0)
        self.sumsq += np.sum(np.square(features), axis=0)
        self.minimum = np.minimum(self.minimum, np.min(features, axis=0))
        self.maximum = np.maximum(self.maximum, np.max(features, axis=0))

    def to_dict(self) -> dict[str, Any]:
        """Serialize accumulated statistics for ``feature_statistics.json``."""

        if self.channels is None or self.frame_count == 0:
            return {
                "channels": 0,
                "frame_count": 0,
                "utterance_count": 0,
            }
        mean = self.sum / self.frame_count
        variance = np.maximum(self.sumsq / self.frame_count - np.square(mean), 0.0)
        return {
            "channels": self.channels,
            "frame_count": self.frame_count,
            "max": self.maximum.tolist(),
            "mean": mean.tolist(),
            "min": self.minimum.tolist(),
            "std": np.sqrt(variance).tolist(),
            "utterance_count": self.utterance_count,
            "value_count": int(self.frame_count * self.channels),
        }


class VoicedF0Stats:
    """Streaming statistics for voiced F0 values and utterance-level voiced means."""

    def __init__(self) -> None:
        """Initialize empty voiced-F0 accumulators."""

        self.frame_count = 0
        self.voiced_frame_count = 0
        self.utterance_count = 0
        self.voiced_utterance_count = 0
        self.voiced_sum = 0.0
        self.voiced_sumsq = 0.0
        self.voiced_min = np.inf
        self.voiced_max = -np.inf
        self.utterance_mean_sum = 0.0
        self.utterance_mean_sumsq = 0.0

    def update(self, f0: np.ndarray) -> float | None:
        """Update statistics from one utterance F0 contour.

        Args:
            f0: Array shaped ``[frames]`` in Hz; unvoiced frames are zero.

        Returns:
            The utterance voiced-F0 mean in Hz, or ``None`` when all frames are
            unvoiced.
        """

        f0 = np.asarray(f0, dtype=np.float64)
        self.frame_count += int(f0.shape[0])
        self.utterance_count += 1
        voiced = f0[f0 > 0.0]
        if len(voiced) == 0:
            return None

        utterance_mean = float(np.mean(voiced))
        self.voiced_frame_count += int(len(voiced))
        self.voiced_utterance_count += 1
        self.voiced_sum += float(np.sum(voiced))
        self.voiced_sumsq += float(np.sum(np.square(voiced)))
        self.voiced_min = min(self.voiced_min, float(np.min(voiced)))
        self.voiced_max = max(self.voiced_max, float(np.max(voiced)))
        self.utterance_mean_sum += utterance_mean
        self.utterance_mean_sumsq += utterance_mean * utterance_mean
        return utterance_mean

    def to_dict(self) -> dict[str, Any]:
        """Serialize voiced-F0 statistics."""

        voiced_fraction = (
            float(self.voiced_frame_count / self.frame_count)
            if self.frame_count
            else 0.0
        )
        result: dict[str, Any] = {
            "frame_count": self.frame_count,
            "utterance_count": self.utterance_count,
            "voiced_frame_count": self.voiced_frame_count,
            "voiced_fraction": voiced_fraction,
            "voiced_utterance_count": self.voiced_utterance_count,
        }
        if self.voiced_frame_count == 0:
            result.update(
                {
                    "mean_of_utterance_voiced_mean_hz": None,
                    "voiced_f0_max_hz": None,
                    "voiced_f0_mean_hz": None,
                    "voiced_f0_min_hz": None,
                    "voiced_f0_std_hz": None,
                    "voiced_utterance_mean_std_hz": None,
                }
            )
            return result

        voiced_mean = self.voiced_sum / self.voiced_frame_count
        voiced_variance = max(
            self.voiced_sumsq / self.voiced_frame_count - voiced_mean**2, 0.0
        )
        utterance_mean = self.utterance_mean_sum / self.voiced_utterance_count
        utterance_mean_variance = max(
            self.utterance_mean_sumsq / self.voiced_utterance_count - utterance_mean**2,
            0.0,
        )
        result.update(
            {
                "mean_of_utterance_voiced_mean_hz": utterance_mean,
                "voiced_f0_max_hz": self.voiced_max,
                "voiced_f0_mean_hz": voiced_mean,
                "voiced_f0_min_hz": self.voiced_min,
                "voiced_f0_std_hz": float(np.sqrt(voiced_variance)),
                "voiced_utterance_mean_std_hz": float(np.sqrt(utterance_mean_variance)),
            }
        )
        return result


def _nested_stats_dict():
    """Create the nested defaultdict value used for split/feature statistics."""

    return defaultdict(ChannelStats)


def _enabled_conditions(config: dict[str, Any]) -> tuple[str, ...]:
    """Return enabled content conditions from feature config."""

    conditions = config.get("conditions", {})
    enabled = [
        name
        for name, condition_config in conditions.items()
        if isinstance(condition_config, dict) and condition_config.get("enabled", False)
    ]
    return tuple(enabled) or ("hubert_soft", "ppg")


def _condition_rows(
    *,
    item_id: str,
    speaker: str,
    utterance_id: str,
    duration_sec: float,
    target_frames: int,
    feature_map,
    aligned_paths: dict[str, Path],
) -> list[dict[str, Any]]:
    """Build rows for ``feature_shapes.csv`` from aligned condition features.

    Feature arrays are expected to be shaped ``[frames, channels]``.
    """

    rows = []
    for name, feature in feature_map.items():
        metadata = feature.metadata or {}
        aligned_frames = int(feature.aligned.shape[0])
        rows.append(
            {
                "aligned_path": aligned_paths[name].as_posix(),
                "aligned_shape": str(list(feature.aligned.shape)),
                "alignment_method": metadata.get("alignment_method", ""),
                "channels": int(feature.aligned.shape[1]),
                "effective_frame_period_ms": 1000.0 * duration_sec / aligned_frames,
                "feature": name,
                "frames": aligned_frames,
                "input_padding_ms_each_side": metadata.get(
                    "input_padding_ms_each_side", ""
                ),
                "item_id": item_id,
                "speaker": speaker,
                "theoretical_frame_period_ms": feature.theoretical_frame_period_ms,
                "upsample_factor": metadata.get("upsample_factor", ""),
                "utterance_id": utterance_id,
                "voiced_fraction": "",
                "voiced_mean_f0_hz": "",
            }
        )
    return rows


def _condition_summary(
    feature, *, target_frames: int, duration_sec: float
) -> dict[str, Any]:
    """Summarize one aligned condition feature for ``feature_summary.json``.

    The feature's ``raw`` and ``aligned`` arrays are shaped ``[frames, channels]``.
    """

    metadata = {
        key: value
        for key, value in (feature.metadata or {}).items()
        if key != "raw_is_extracted_from_padded_waveform"
    }
    aligned_frames = int(feature.aligned.shape[0])
    return {
        "aligned_shape": list(feature.aligned.shape),
        "aligned_to_target_frame_count": aligned_frames,
        "channels": int(feature.aligned.shape[1]),
        "effective_frame_period_ms": 1000.0 * duration_sec / aligned_frames,
        "frame_difference_vs_target": int(aligned_frames - target_frames),
        "frames": aligned_frames,
        "metadata": metadata,
        "name": feature.name,
        "target_to_feature_frame_ratio": float(target_frames / aligned_frames),
        "theoretical_frame_period_ms": feature.theoretical_frame_period_ms,
    }


def _world_aux_condition(world) -> ResampledConditionFeature:
    """Wrap target WORLD auxiliary features as a cached condition component.

    Returns:
        ``ResampledConditionFeature`` whose raw/aligned arrays are shaped
        ``[frames, 2 + coded_ap_channels]``.
    """

    aux = world_aux_features(world.f0, world.coded_ap)
    return ResampledConditionFeature(
        name="world_aux",
        raw=aux,
        aligned=aux,
        theoretical_frame_period_ms=world.frame_period_ms,
        metadata={
            "alignment_method": "world_frame_native",
            "components": "log1p_f0,vuv,coded_ap",
            "coded_ap_channels": int(world.coded_ap.shape[1]),
        },
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for feature dumping."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/speech/features.yaml")
    parser.add_argument("--manifest", default="data/manifests/cmu_arctic_inventory.csv")
    parser.add_argument("--path-root", default=".")
    parser.add_argument("--output-dir", default="feature_cache/cmu_arctic")
    parser.add_argument(
        "--split", default="train", help="Manifest split to dump, or 'all'."
    )
    parser.add_argument(
        "--speakers",
        type=_parse_speakers,
        default=None,
        help="Comma-separated speaker filters, or 'all'.",
    )
    parser.add_argument("--utterance-id", default=None)
    parser.add_argument(
        "--max-utterances",
        type=int,
        default=1,
        help="Maximum utterances to dump per selected speaker.",
    )
    parser.add_argument("--conditions", type=_parse_conditions, default=None)
    parser.add_argument("--hubert-repo", default=None)
    parser.add_argument("--sample-rate", type=int, default=None)
    parser.add_argument("--world-frame-period-ms", type=float, default=None)
    parser.add_argument("--coded-sp-dim", type=int, default=None)
    parser.add_argument("--f0-floor", type=float, default=None)
    parser.add_argument("--f0-ceil", type=float, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Dump WORLD targets, aligned condition components, and statistics files."""

    args = parse_args()
    config = load_yaml(args.config)
    path_root = Path(args.path_root)
    output_dir = ensure_dir(args.output_dir)

    sample_rate = int(args.sample_rate or 16000)
    world_config = WorldFeatureConfig(
        frame_period_ms=float(
            args.world_frame_period_ms
            if args.world_frame_period_ms is not None
            else _nested(config, "world", "frame_period_ms", default=10.0)
        ),
        coded_sp_dim=int(
            args.coded_sp_dim
            if args.coded_sp_dim is not None
            else _nested(config, "world", "coded_sp_dim", default=48)
        ),
        f0_floor=float(
            args.f0_floor
            if args.f0_floor is not None
            else _nested(config, "world", "f0_floor", default=71.0)
        ),
        f0_ceil=float(
            args.f0_ceil
            if args.f0_ceil is not None
            else _nested(config, "world", "f0_ceil", default=800.0)
        ),
    )
    conditions = args.conditions or _enabled_conditions(config)
    cached_conditions = tuple([*conditions, "world_aux"])

    selected = select_manifest_rows(
        read_manifest_rows(args.manifest),
        split=args.split,
        speakers=args.speakers,
        utterance_id=args.utterance_id,
        max_utterances=args.max_utterances,
    )
    selected_speaker_set = _speaker_set_from_rows(selected)
    selected_speaker_to_id = _speaker_to_id_from_rows(selected)

    device = resolve_device(gpu=args.gpu, cpu=args.cpu)
    ppg_gpu = None if device.type == "cpu" else args.gpu
    hubert_model = None
    if "hubert_soft" in conditions:
        hubert_model = load_hubert_soft_model(
            hubert_repo=args.hubert_repo,
            device=device,
        )

    manifest_rows: list[dict[str, Any]] = []
    shape_rows: list[dict[str, Any]] = []
    alignment_checks: list[dict[str, Any]] = []
    alignment_issues: list[dict[str, Any]] = []
    feature_stats: defaultdict[str, defaultdict[str, ChannelStats]] = defaultdict(
        _nested_stats_dict
    )
    f0_stats: defaultdict[str, VoicedF0Stats] = defaultdict(VoicedF0Stats)
    summary: dict[str, Any] = {
        "cached_conditions": list(cached_conditions),
        "conditions": list(cached_conditions),
        "content_conditions": list(conditions),
        "device": str(device),
        "manifest": str(args.manifest),
        "output_dir": output_dir.as_posix(),
        "path_root": path_root.as_posix(),
        "selected_items": [
            f"{row.get('speaker', '')}:{row['utterance_id']}" for row in selected
        ],
        "speaker_filter": list(args.speakers) if args.speakers is not None else None,
        "speaker_set": selected_speaker_set,
        "speaker_to_id": selected_speaker_to_id,
        "split": args.split,
        "utterances": {},
        "world": {
            "coded_sp_dim": world_config.coded_sp_dim,
            "f0_ceil": world_config.f0_ceil,
            "f0_floor": world_config.f0_floor,
            "frame_period_ms": world_config.frame_period_ms,
        },
    }

    for item in selected:
        utterance_id = item["utterance_id"]
        speaker = str(item["speaker"])
        item_id = f"{speaker}_{utterance_id}"
        wav_path = resolve_manifest_path(path_root, item["wav_path"])
        waveform, actual_sample_rate = load_mono_audio(
            wav_path, target_sample_rate=sample_rate
        )
        duration_sec = len(waveform) / actual_sample_rate
        world = analyze_world(waveform, actual_sample_rate, world_config)
        split_name = item["split"]
        world_summary = world.summary()
        world_voiced_mean_f0_hz = world_summary["voiced_f0_mean_hz"]
        world_voiced_fraction = world_summary["voiced_fraction"]

        utterance_aligned_dir = ensure_dir(
            output_dir / "aligned" / speaker / utterance_id
        )
        world_path = utterance_aligned_dir / "world_coded_sp.npy"
        np.save(world_path, world.coded_sp)

        condition_features = extract_resampled_condition_features(
            waveform,
            actual_sample_rate,
            target_frame_count=world.frame_count,
            conditions=conditions,
            hubert_model=hubert_model,
            device=device,
            gpu=ppg_gpu,
        )
        world_aux = _world_aux_condition(world)
        condition_features["world_aux"] = world_aux
        aligned_paths: dict[str, Path] = {}
        condition_summaries: dict[str, Any] = {}
        for name, feature in condition_features.items():
            aligned_paths[name] = utterance_aligned_dir / f"{name}.npy"
            np.save(aligned_paths[name], feature.aligned)
            condition_summaries[name] = _condition_summary(
                feature,
                target_frames=world.frame_count,
                duration_sec=duration_sec,
            )

        for stats_key in ("all", split_name):
            feature_stats[stats_key]["world_coded_sp"].update(world.coded_sp)
            f0_stats[stats_key].update(world.f0)
            for name, feature in condition_features.items():
                feature_stats[stats_key][name].update(feature.aligned)

        for name, feature in condition_features.items():
            check = {
                "aligned_frames": int(feature.aligned.shape[0]),
                "feature": name,
                "ok": int(feature.aligned.shape[0]) == world.frame_count,
                "speaker": speaker,
                "split": split_name,
                "utterance_id": utterance_id,
                "world_frames": world.frame_count,
            }
            alignment_checks.append(check)
            if not check["ok"]:
                alignment_issues.append(check)

        shape_rows.append(
            {
                "aligned_path": world_path.as_posix(),
                "aligned_shape": str(list(world.coded_sp.shape)),
                "alignment_method": "",
                "channels": int(world.coded_sp.shape[1]),
                "effective_frame_period_ms": 1000.0 * duration_sec / world.frame_count,
                "feature": "world_coded_sp",
                "frames": world.frame_count,
                "input_padding_ms_each_side": "",
                "item_id": item_id,
                "speaker": speaker,
                "theoretical_frame_period_ms": world_config.frame_period_ms,
                "upsample_factor": "",
                "utterance_id": utterance_id,
                "voiced_fraction": world_voiced_fraction,
                "voiced_mean_f0_hz": world_voiced_mean_f0_hz,
            }
        )
        shape_rows.extend(
            _condition_rows(
                item_id=item_id,
                speaker=speaker,
                utterance_id=utterance_id,
                duration_sec=duration_sec,
                target_frames=world.frame_count,
                feature_map=condition_features,
                aligned_paths=aligned_paths,
            )
        )

        manifest_row = {
            "aligned_condition_paths": {
                name: path.as_posix() for name, path in sorted(aligned_paths.items())
            },
            "conditions": sorted(condition_features),
            "duration_sec": duration_sec,
            "item_id": item_id,
            "sample_rate": actual_sample_rate,
            "speaker": speaker,
            "speaker_id": (
                int(item["speaker_id"])
                if item.get("speaker_id") not in (None, "")
                else None
            ),
            "speaker_set": selected_speaker_set,
            "speaker_to_id": selected_speaker_to_id,
            "split": item["split"],
            "utterance_id": utterance_id,
            "voiced_fraction": world_voiced_fraction,
            "voiced_mean_f0_hz": world_voiced_mean_f0_hz,
            "wav_path": item["wav_path"],
            "world_coded_sp_path": world_path.as_posix(),
            "world_frame_count": world.frame_count,
            "world_frame_period_ms": world_config.frame_period_ms,
        }
        manifest_rows.append(manifest_row)
        summary["utterances"][item_id] = {
            "conditions": condition_summaries,
            "duration_sec": duration_sec,
            "manifest_row": manifest_row,
            "world": world_summary,
        }

        print(
            "utterance="
            + ",".join(
                [
                    f"id:{utterance_id}",
                    f"speaker:{speaker}",
                    f"world:{tuple(world.coded_sp.shape)}",
                    *[
                        f"{name}:{tuple(feature.aligned.shape)}"
                        for name, feature in sorted(condition_features.items())
                    ],
                ]
            )
        )

    shape_fieldnames = [
        "item_id",
        "speaker",
        "utterance_id",
        "feature",
        "aligned_shape",
        "alignment_method",
        "frames",
        "channels",
        "theoretical_frame_period_ms",
        "effective_frame_period_ms",
        "input_padding_ms_each_side",
        "upsample_factor",
        "aligned_path",
        "voiced_fraction",
        "voiced_mean_f0_hz",
    ]
    statistics = {
        split_name: {
            "features": {
                feature_name: stats.to_dict()
                for feature_name, stats in sorted(split_stats.items())
            },
            "world_f0": f0_stats[split_name].to_dict(),
        }
        for split_name, split_stats in sorted(feature_stats.items())
    }
    alignment_report = {
        "check_count": len(alignment_checks),
        "checked_condition_features": list(cached_conditions),
        "checked_utterance_count": len(selected),
        "checks": alignment_checks,
        "issue_count": len(alignment_issues),
        "issues": alignment_issues,
    }
    summary["statistics_path"] = (output_dir / "feature_statistics.json").as_posix()
    summary["frame_alignment_report_path"] = (
        output_dir / "frame_alignment_report.json"
    ).as_posix()
    write_jsonl(output_dir / "feature_manifest.jsonl", manifest_rows)
    write_csv_rows(output_dir / "feature_shapes.csv", shape_rows, shape_fieldnames)
    write_json(output_dir / "feature_summary.json", summary)
    write_json(output_dir / "feature_statistics.json", statistics)
    write_json(output_dir / "frame_alignment_report.json", alignment_report)

    print(f"output_dir={output_dir}")
    print(f"feature_manifest={output_dir / 'feature_manifest.jsonl'}")
    print(f"feature_summary={output_dir / 'feature_summary.json'}")
    print(f"feature_shapes={output_dir / 'feature_shapes.csv'}")
    print(f"feature_statistics={output_dir / 'feature_statistics.json'}")
    print(f"frame_alignment_report={output_dir / 'frame_alignment_report.json'}")


if __name__ == "__main__":
    main()
