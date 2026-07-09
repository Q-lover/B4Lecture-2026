"""Generate WORLD coded spectral envelopes and synthesize speech."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from nf_assignment.speech.conditions import WORLD_AUX_CONDITION, parse_condition_spec
from nf_assignment.speech.data import select_vc_sample_items
from nf_assignment.speech.features.content import load_hubert_soft_model
from nf_assignment.speech.features.world import (
    WorldFeatureBundle,
    WorldFeatureConfig,
    analyze_world,
    load_mono_audio,
    voiced_f0_mean,
)
from nf_assignment.speech.model import build_speech_flow
from nf_assignment.speech.normalization import load_feature_normalizers
from nf_assignment.speech.sample import (
    extract_vc_condition,
    generate_coded_sp,
    synthesize_generated_world,
    synthesize_target_world,
)
from nf_assignment.speech.train import resolve_device
from nf_assignment.training.checkpoints import load_checkpoint
from nf_assignment.utils.io import (
    ensure_dir,
    load_yaml,
    read_csv_rows,
    write_csv_rows,
    write_json,
)
from nf_assignment.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for speech generation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-config", default="configs/speech/sample.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--inventory-manifest", default=None)
    parser.add_argument("--path-root", default=None)
    parser.add_argument("--statistics-path", default=None)
    parser.add_argument("--statistics-split", default=None)
    parser.add_argument("--source-speaker", default=None)
    parser.add_argument("--target-speaker", default=None)
    parser.add_argument(
        "--utterance-ids",
        default=None,
        help="Comma-separated utterance IDs to sample. If omitted, rows are selected by split.",
    )
    parser.add_argument(
        "--condition",
        default=None,
        help=(
            "Condition component name or comma/plus separated component list, "
            "for example hubert_soft or hubert_soft,world_aux."
        ),
    )
    parser.add_argument("--split", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-utterances", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--latent-scale", type=float, default=None)
    parser.add_argument("--hubert-repo", default=None)
    parser.add_argument("--no-synthesis", action="store_true")
    return parser.parse_args()


def _parse_utterance_ids(
    value: str | list[str] | tuple[str, ...] | None,
) -> list[str] | None:
    """Normalize optional utterance-ID selections from config or CLI."""

    if value is None:
        return None
    if isinstance(value, str):
        items = value.split(",")
    else:
        items = [str(item) for item in value]
    parsed = [item.strip() for item in items if item and item.strip()]
    return parsed or None


def _read_inventory_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read source/target wav inventory rows from CSV or JSONL."""

    path = Path(path)
    if path.suffix == ".csv":
        return read_csv_rows(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _resolve_existing_path(value: str | Path, roots: list[Path]) -> Path:
    """Resolve a path by checking absolute value and candidate roots."""

    path = Path(value)
    if path.is_absolute():
        return path
    for root in roots:
        candidate = root / path
        if candidate.exists():
            return candidate
    return roots[0] / path


def _statistics_path(
    args: argparse.Namespace,
    sample_config: dict[str, Any],
    metadata: dict[str, Any],
) -> Path:
    """Choose the feature-statistics file used for sample-time normalization."""

    configured = args.statistics_path or sample_config.get("statistics_path")
    if configured:
        return Path(configured)
    metadata_stats = metadata.get("statistics_path")
    if metadata_stats:
        return Path(metadata_stats)
    return Path("feature_cache/cmu_arctic_full/feature_statistics.json")


def _world_config_from_metadata(
    metadata: dict[str, Any],
    sample_config: dict[str, Any],
) -> WorldFeatureConfig:
    """Build WORLD settings compatible with the checkpoint channel count."""

    world_config = sample_config.get("world", {}) or {}
    model_kwargs = metadata.get("model_kwargs", {})
    model_channels = int(model_kwargs.get("coded_sp_channels", 48))
    coded_sp_dim = int(world_config.get("coded_sp_dim", model_channels))
    if coded_sp_dim != model_channels:
        raise ValueError(
            f"sample WORLD coded_sp_dim={coded_sp_dim} does not match "
            f"checkpoint coded_sp_channels={model_channels}."
        )
    return WorldFeatureConfig(
        frame_period_ms=float(world_config.get("frame_period_ms", 10.0)),
        coded_sp_dim=coded_sp_dim,
        f0_floor=float(world_config.get("f0_floor", 71.0)),
        f0_ceil=float(world_config.get("f0_ceil", 800.0)),
    )


def _analyze_item_world(
    row: dict[str, Any],
    *,
    wav_key: str,
    world_config: WorldFeatureConfig,
    target_sample_rate: int | None,
    path_roots: list[Path],
) -> tuple[str, np.ndarray, WorldFeatureBundle]:
    """Load and analyze one source or target wav item with WORLD.

    Returns:
        A tuple of resolved wav path, waveform shaped ``[samples]``, and WORLD
        feature bundle.
    """

    if not row.get(wav_key):
        utterance_id = row.get("utterance_id")
        raise KeyError(f"{wav_key} is required for VC sampling: {utterance_id}")
    wav_path = _resolve_existing_path(row[wav_key], path_roots)
    waveform, sample_rate = load_mono_audio(
        wav_path, target_sample_rate=target_sample_rate
    )
    world = analyze_world(waveform, sample_rate, world_config)
    return wav_path.as_posix(), waveform, world


def _synthesis_features_from_world(
    world: WorldFeatureBundle,
    *,
    utterance_id: str,
    wav_path: str,
) -> dict[str, Any]:
    """Convert a WORLD bundle into the synthesis-feature dictionary format."""

    return {
        "aperiodicity": world.aperiodicity,
        "f0": world.f0,
        "fft_size": world.fft_size,
        "frame_period_ms": world.frame_period_ms,
        "sample_rate": world.sample_rate,
        "time_axis": world.time_axis,
        "utterance_id": utterance_id,
        "wav_path": wav_path,
    }


def _sample_manifest_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    """Return stable CSV field order for the speech sample manifest."""

    preferred = [
        "sample_id",
        "utterance_id",
        "source_speaker",
        "target_speaker",
        "condition",
        "condition_components",
        "condition_role",
        "base_condition",
        "uses_world_aux",
        "frames",
        "target_frames",
        "source_world_frames",
        "source_condition_shape",
        "source_wav_path",
        "original_target_wav_path",
        "source_world_wav_path",
        "generated_coded_sp_path",
        "target_coded_sp_path",
        "source_condition_path",
        "source_world_aux_path",
        "generated_sp_path",
        "target_sp_path",
        "shifted_f0_path",
        "target_f0_path",
        "generated_wav_path",
        "target_wav_path",
        "wav_path",
        "generated_min",
        "generated_max",
        "generated_std",
    ]
    keys = {key for row in rows for key in row}
    return [key for key in preferred if key in keys] + sorted(keys - set(preferred))


def _csv_ready_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert nested values to JSON strings before CSV writing."""

    converted = []
    for row in rows:
        converted_row = {}
        for key, value in row.items():
            if isinstance(value, (list, dict)):
                converted_row[key] = json.dumps(value, sort_keys=True)
            else:
                converted_row[key] = value
        converted.append(converted_row)
    return converted


def main() -> None:
    """Run VC-style speech sampling and optional WORLD synthesis."""

    args = parse_args()
    sample_config = load_yaml(args.sample_config)
    checkpoint_path = args.checkpoint or str(
        sample_config.get("checkpoint", "runs/speech_world_flow/checkpoint.pt")
    )
    payload = load_checkpoint(checkpoint_path, map_location="cpu")
    metadata = payload.get("metadata", {})

    seed = args.seed if args.seed is not None else int(sample_config.get("seed", 0))
    set_seed(seed)
    device = resolve_device(args.device)
    config_condition = sample_config.get("condition")
    if args.condition is not None:
        condition_value = args.condition
    elif config_condition is not None:
        condition_value = config_condition
    else:
        condition_value = metadata.get(
            "condition_components",
            metadata.get("condition", "hubert_soft"),
        )
    condition_spec = parse_condition_spec(condition_value)
    condition = condition_spec.name
    base_condition = condition_spec.single_content_condition()
    uses_world_aux = condition_spec.uses_world_aux
    inventory_manifest = args.inventory_manifest or str(
        sample_config.get(
            "inventory_manifest", "data/manifests/cmu_arctic_inventory.csv"
        )
    )
    path_root = Path(args.path_root or sample_config.get("path_root", "."))
    path_roots = [path_root, Path.cwd()]
    inventory_manifest_path = _resolve_existing_path(inventory_manifest, path_roots)
    statistics_path = _resolve_existing_path(
        _statistics_path(args, sample_config, metadata),
        path_roots,
    )
    statistics_split = str(
        args.statistics_split
        or sample_config.get("statistics_split")
        or metadata.get("statistics_split", "train")
    )
    source_speaker = str(
        args.source_speaker or sample_config.get("source_speaker", "bdl")
    )
    target_speaker = str(
        args.target_speaker or sample_config.get("target_speaker", "slt")
    )
    utterance_ids = _parse_utterance_ids(
        args.utterance_ids
        if args.utterance_ids is not None
        else sample_config.get("utterance_ids")
    )
    split = args.split or str(
        sample_config.get("split", metadata.get("valid_split", "valid"))
    )
    latent_scale = (
        float(args.latent_scale)
        if args.latent_scale is not None
        else float(sample_config.get("latent_scale", 1.0))
    )
    num_utterances = args.num_utterances or int(sample_config.get("num_utterances", 2))
    output_dir = ensure_dir(
        args.output_dir
        or str(sample_config.get("output_dir", "outputs/speech_world_flow"))
    )

    normalizers = load_feature_normalizers(statistics_path, split=statistics_split)
    needed_normalizers = {"world_coded_sp", *condition_spec.components}
    missing_normalizers = sorted(needed_normalizers - set(normalizers))
    if missing_normalizers:
        raise KeyError(
            f"normalization statistics are missing from {statistics_path}: "
            f"{missing_normalizers}"
        )
    normalizers = {name: normalizers[name] for name in needed_normalizers}
    sample_items = select_vc_sample_items(
        inventory_rows=_read_inventory_rows(inventory_manifest_path),
        source_speaker=source_speaker,
        target_speaker=target_speaker,
        split=split,
        utterance_ids=utterance_ids,
        max_items=num_utterances,
    )
    world_config = _world_config_from_metadata(metadata, sample_config)
    model = build_speech_flow(**metadata["model_kwargs"]).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    expected_condition_channels = int(metadata["model_kwargs"]["condition_channels"])
    ppg_gpu = None if device.type == "cpu" else int(device.index or 0)
    hubert_model = None
    if base_condition == "hubert_soft":
        hubert_model = load_hubert_soft_model(
            hubert_repo=args.hubert_repo or sample_config.get("hubert_repo"),
            device=device,
        )

    started = time.time()
    rows: list[dict[str, Any]] = []
    count = len(sample_items)
    path_roots.extend(
        [inventory_manifest_path.parent, inventory_manifest_path.parent.parent]
    )
    for item in sample_items:
        sample_rate = sample_config.get("sample_rate")
        source_wav_path, source_waveform, source_world = _analyze_item_world(
            item,
            wav_key="source_wav",
            world_config=world_config,
            target_sample_rate=sample_rate,
            path_roots=path_roots,
        )
        target_wav_path, _, target_world = _analyze_item_world(
            item,
            wav_key="target_wav",
            world_config=world_config,
            target_sample_rate=sample_rate,
            path_roots=path_roots,
        )
        target_voiced_mean_f0_hz = voiced_f0_mean(target_world.f0)
        source_condition = extract_vc_condition(
            source_waveform,
            source_world.sample_rate,
            condition=condition_spec.components,
            frame_count=source_world.frame_count,
            normalizers=normalizers,
            device=device,
            hubert_model=hubert_model,
            gpu=ppg_gpu,
            source_world=source_world,
            target_voiced_mean_f0_hz=target_voiced_mean_f0_hz,
        )
        condition_tensor = source_condition["condition"]
        if int(condition_tensor.shape[1]) != expected_condition_channels:
            raise ValueError(
                f"source condition channel mismatch: expected {expected_condition_channels}, "
                f"got {int(condition_tensor.shape[1])}"
            )
        with torch.no_grad():
            generated_norm = generate_coded_sp(
                model,
                condition=condition_tensor,
                mask=source_condition["mask"],
                latent_scale=latent_scale,
            )
        length = int(source_condition["length"])
        generated_norm_np = generated_norm[0, :, :length].detach().cpu().numpy().T
        target_coded_sp = target_world.coded_sp
        target_length = int(target_coded_sp.shape[0])
        if int(target_coded_sp.shape[1]) != int(
            metadata["model_kwargs"]["coded_sp_channels"]
        ):
            raise ValueError(
                f"target coded_sp channels={target_coded_sp.shape[1]} does not match "
                f"checkpoint coded_sp_channels={metadata['model_kwargs']['coded_sp_channels']}."
            )
        normalizer = normalizers["world_coded_sp"]
        generated_coded_sp = normalizer.denormalize(generated_norm_np)

        utterance_id = str(item["utterance_id"])
        sample_id = str(item.get("sample_id", utterance_id))
        utterance_dir = ensure_dir(output_dir / sample_id)
        generated_path = utterance_dir / "generated_coded_sp.npy"
        target_path = utterance_dir / "target_coded_sp.npy"
        source_condition_path = utterance_dir / f"source_{condition}.npy"
        np.save(generated_path, generated_coded_sp.astype(np.float32))
        np.save(target_path, target_coded_sp.astype(np.float32))
        np.save(source_condition_path, source_condition["aligned"].astype(np.float32))
        source_world_aux_path = None
        if uses_world_aux:
            source_world_aux_path = utterance_dir / "source_world_aux.npy"
            np.save(
                source_world_aux_path,
                source_condition["components"][WORLD_AUX_CONDITION].astype(np.float32),
            )

        row: dict[str, Any] = {
            "base_condition": base_condition,
            "condition": condition,
            "condition_components": list(condition_spec.components),
            "condition_role": "source",
            "frames": length,
            "generated_coded_sp_path": generated_path.as_posix(),
            "generated_max": float(np.max(generated_coded_sp)),
            "generated_min": float(np.min(generated_coded_sp)),
            "generated_std": float(np.std(generated_coded_sp)),
            "source_condition_path": source_condition_path.as_posix(),
            "source_condition_shape": list(source_condition["aligned"].shape),
            "source_speaker": item.get("source_speaker"),
            "source_world_frames": source_world.frame_count,
            "source_world_wav_path": source_wav_path,
            "sample_id": sample_id,
            "target_coded_sp_path": target_path.as_posix(),
            "target_frames": target_length,
            "target_speaker": item.get("target_speaker"),
            "utterance_id": utterance_id,
            "uses_world_aux": uses_world_aux,
        }
        if source_world_aux_path is not None:
            row["source_world_aux_path"] = source_world_aux_path.as_posix()
        row["original_target_wav_path"] = target_wav_path
        row["source_wav_path"] = source_wav_path
        if not args.no_synthesis:
            synthesis_features = _synthesis_features_from_world(
                source_world,
                utterance_id=utterance_id,
                wav_path=source_wav_path,
            )
            synthesis = synthesize_generated_world(
                generated_coded_sp,
                synthesis_features=synthesis_features,
                target_voiced_mean_f0_hz=target_voiced_mean_f0_hz,
            )
            target_synthesis_features = _synthesis_features_from_world(
                target_world,
                utterance_id=utterance_id,
                wav_path=target_wav_path,
            )
            target_synthesis = synthesize_target_world(
                target_coded_sp,
                synthesis_features=target_synthesis_features,
            )
            generated_sp_path = utterance_dir / "generated_sp.npy"
            target_sp_path = utterance_dir / "target_sp.npy"
            shifted_f0_path = utterance_dir / "shifted_source_f0.npy"
            target_f0_path = utterance_dir / "target_f0.npy"
            waveform_path = utterance_dir / "generated.wav"
            target_waveform_path = utterance_dir / "target.wav"
            np.save(generated_sp_path, synthesis["spectral_envelope"])
            target_sp = target_synthesis["spectral_envelope"]
            np.save(target_sp_path, target_sp)
            np.save(shifted_f0_path, synthesis["f0"])
            np.save(target_f0_path, target_synthesis["f0"])
            import soundfile as sf

            sf.write(
                waveform_path,
                synthesis["waveform"],
                int(synthesis_features["sample_rate"]),
            )
            sf.write(
                target_waveform_path,
                target_synthesis["waveform"],
                int(target_synthesis_features["sample_rate"]),
            )
            row.update(
                {
                    "generated_sp_path": generated_sp_path.as_posix(),
                    "generated_wav_path": waveform_path.as_posix(),
                    "shifted_f0_path": shifted_f0_path.as_posix(),
                    "target_f0_path": target_f0_path.as_posix(),
                    "target_sp_path": target_sp_path.as_posix(),
                    "target_wav_path": target_waveform_path.as_posix(),
                    "wav_path": waveform_path.as_posix(),
                }
            )
        rows.append(row)

    elapsed_sec = time.time() - started
    write_csv_rows(
        output_dir / "sample_manifest.csv",
        _csv_ready_rows(rows),
        _sample_manifest_fieldnames(rows),
    )
    metrics = {
        "base_condition": base_condition,
        "checkpoint": str(checkpoint_path),
        "condition": condition,
        "condition_components": list(condition_spec.components),
        "device": str(device),
        "elapsed_sec": elapsed_sec,
        "inference_condition_role": "source",
        "inventory_manifest": inventory_manifest_path.as_posix(),
        "latent_scale": latent_scale,
        "num_utterances": count,
        "output_dir": str(output_dir),
        "source_speaker": source_speaker,
        "split": split,
        "statistics_path": statistics_path.as_posix(),
        "statistics_split": statistics_split,
        "synthesis_world_role": "source",
        "synthesis": not args.no_synthesis,
        "target_speaker": target_speaker,
        "torch_version": torch.__version__,
        "utterance_ids": utterance_ids,
        "uses_world_aux": uses_world_aux,
    }
    if device.type == "cuda":
        metrics["cuda_device"] = torch.cuda.get_device_name(device)
    write_json(output_dir / "sample_metrics.json", metrics)

    print(f"output_dir={output_dir}")
    print(f"checkpoint={checkpoint_path}")
    print(f"condition={condition}")
    print(f"device={device}")
    print(f"num_utterances={count}")
    print(f"elapsed_sec={elapsed_sec:.3f}")


if __name__ == "__main__":
    main()
