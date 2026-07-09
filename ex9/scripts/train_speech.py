"""Train WORLD coded spectral envelope conditional flows."""

from __future__ import annotations

import argparse
import csv
import sys
import time

import torch
from nf_assignment.speech.conditions import parse_condition_spec
from nf_assignment.speech.data import normalize_speaker_list
from nf_assignment.speech.dataset import SpeechFeatureDataset, collate_speech_features
from nf_assignment.speech.model import build_speech_flow
from nf_assignment.speech.train import (
    build_optimizer,
    move_batch_to_device,
    resolve_device,
    speech_nll_loss,
    train_speech_flow,
)
from nf_assignment.speech.visualize import plot_loss_curve
from nf_assignment.training.checkpoints import load_checkpoint, save_checkpoint
from nf_assignment.utils.io import ensure_dir, load_yaml, write_csv_rows, write_json
from nf_assignment.utils.seed import set_seed
from torch.utils.data import DataLoader
from tqdm import tqdm

DEFAULT_DATA_CONFIG = "configs/speech/data.yaml"
DEFAULT_MODEL_CONFIG = "configs/speech/model.yaml"
DEFAULT_TRAIN_CONFIG = "configs/speech/train.yaml"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for speech-flow training."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--model-config", default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--train-config", default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument(
        "--condition",
        default=None,
        help=(
            "Condition component name or comma/plus separated component list, "
            "for example hubert_soft or hubert_soft,world_aux."
        ),
    )
    parser.add_argument("--feature-manifest", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--speakers",
        default=None,
        help="Comma-separated speaker filter for cached feature rows.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help=(
            "Resume model and optimizer state from this checkpoint. "
            "When the default train config is used, checkpoint metadata supplies defaults for "
            "condition, feature manifest, batch size, segment frames, splits, and optimizer."
        ),
    )
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--segment-frames",
        type=int,
        default=None,
        help="Crop training batches to this many frames; 0 or a negative value disables cropping.",
    )
    return parser.parse_args()


def _resume_train_config(args: argparse.Namespace, resume_metadata: dict) -> dict:
    """Return train-config metadata when the default config is being resumed."""

    if not resume_metadata or args.train_config != DEFAULT_TRAIN_CONFIG:
        return {}
    train_config = resume_metadata.get("train_config", {})
    return train_config if isinstance(train_config, dict) else {}


def _merge_train_config(train_config: dict, resume_train_config: dict) -> dict:
    """Overlay resume training defaults onto the current training config."""

    if not resume_train_config:
        return train_config
    merged = dict(train_config)
    merged.update(resume_train_config)
    return merged


def _normalize_speakers(value) -> tuple[str, ...] | None:
    """Normalize an optional speaker filter from config or CLI."""

    if value is None:
        return None
    if isinstance(value, str):
        return normalize_speaker_list(tuple(part.strip() for part in value.split(",")))
    if isinstance(value, (list, tuple)):
        return normalize_speaker_list(tuple(str(part).strip() for part in value))
    raise TypeError(
        "speakers must be a comma-separated string or a list of speaker names."
    )


def _resume_config(
    args: argparse.Namespace,
    resume_metadata: dict,
    *,
    arg_value: str,
    default_arg_value: str,
    metadata_key: str,
    current_config: dict,
) -> dict:
    """Return resume metadata config only when the CLI argument stayed at default."""

    if resume_metadata and arg_value == default_arg_value:
        config = resume_metadata.get(metadata_key, {})
        if isinstance(config, dict):
            return config
    return current_config


def _model_kwargs(
    model_config: dict, *, target_channels: int, condition_channels: int
) -> dict:
    """Build ``build_speech_flow`` keyword arguments from config and data shapes."""

    return {
        "coded_sp_channels": target_channels,
        "condition_channels": condition_channels,
        "data_dep_init": bool(model_config.get("data_dep_init", False)),
        "dilation_rate": int(model_config.get("dilation_rate", 2)),
        "dropout": float(model_config.get("dropout", 0.0)),
        "hidden_channels": int(model_config.get("hidden_channels", 128)),
        "kernel_size": int(model_config.get("kernel_size", 5)),
        "n_split": int(model_config.get("n_split", 4)),
        "num_blocks": int(model_config.get("num_blocks", 6)),
        "num_layers_per_block": int(model_config.get("num_conditioner_layers", 4)),
    }


def main() -> None:
    """Train or resume a speech flow and write checkpoint/metrics artifacts."""

    args = parse_args()
    data_config = load_yaml(args.data_config)
    model_config = load_yaml(args.model_config)
    train_config = load_yaml(args.train_config)
    resume_payload = (
        load_checkpoint(args.resume, map_location="cpu") if args.resume else None
    )
    resume_metadata = resume_payload.get("metadata", {}) if resume_payload else {}
    if resume_payload is not None and not isinstance(resume_metadata, dict):
        raise ValueError("resume checkpoint metadata must be a dictionary.")
    resume_train_config = _resume_train_config(args, resume_metadata)
    effective_train_config = _merge_train_config(train_config, resume_train_config)
    metadata_data_config = _resume_config(
        args,
        resume_metadata,
        arg_value=args.data_config,
        default_arg_value=DEFAULT_DATA_CONFIG,
        metadata_key="data_config",
        current_config=data_config,
    )
    metadata_model_config = _resume_config(
        args,
        resume_metadata,
        arg_value=args.model_config,
        default_arg_value=DEFAULT_MODEL_CONFIG,
        metadata_key="model_config",
        current_config=model_config,
    )

    seed = int(effective_train_config.get("seed", resume_metadata.get("seed", 0)))
    set_seed(seed)
    device = resolve_device(
        args.device or str(effective_train_config.get("device", "auto"))
    )
    condition_value = (
        args.condition
        if args.condition is not None
        else resume_metadata.get(
            "condition_components",
            resume_metadata.get(
                "condition",
                effective_train_config.get("condition", "hubert_soft"),
            ),
        )
    )
    condition_spec = parse_condition_spec(condition_value)
    condition = condition_spec.name
    feature_manifest = args.feature_manifest or str(
        resume_metadata.get(
            "feature_manifest",
            effective_train_config.get(
                "feature_manifest",
                (
                    f"{data_config.get('feature_cache', 'feature_cache/cmu_arctic')}"
                    "/feature_manifest.jsonl"
                ),
            ),
        )
    )
    batch_size = (
        int(args.batch_size)
        if args.batch_size is not None
        else int(
            resume_metadata.get(
                "batch_size", effective_train_config.get("batch_size", 8)
            )
        )
    )
    segment_frames = (
        int(args.segment_frames)
        if args.segment_frames is not None
        else int(
            resume_metadata.get(
                "segment_frames",
                effective_train_config.get("segment_frames", 0),
            )
        )
    )
    num_steps = args.num_steps or int(effective_train_config.get("num_steps", 1000))
    log_every = int(effective_train_config.get("log_every", 10))
    train_split = str(
        resume_metadata.get(
            "train_split",
            effective_train_config.get(
                "train_split",
                data_config.get("splits", {}).get("train", "train"),
            ),
        )
    )
    valid_split = str(
        resume_metadata.get(
            "valid_split",
            effective_train_config.get(
                "valid_split",
                data_config.get("splits", {}).get("valid", "valid"),
            ),
        )
    )
    speakers = _normalize_speakers(
        args.speakers
        if args.speakers is not None
        else resume_metadata.get("speakers", effective_train_config.get("speakers"))
    )

    output_dir = ensure_dir(
        args.output_dir
        or str(effective_train_config.get("output_dir", "runs/speech_world_flow"))
    )
    statistics_split = str(
        resume_metadata.get(
            "statistics_split",
            effective_train_config.get("statistics_split", train_split),
        )
    )
    normalize = bool(effective_train_config.get("normalize", True))
    train_dataset = SpeechFeatureDataset(
        feature_manifest,
        condition=condition_spec.components,
        split=train_split,
        speakers=speakers,
        normalize=normalize,
        statistics_split=statistics_split,
    )
    valid_dataset = SpeechFeatureDataset(
        feature_manifest,
        condition=condition_spec.components,
        split=valid_split,
        speakers=speakers,
        normalize=normalize,
        statistics_split=statistics_split,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_speech_features,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_speech_features,
    )

    first_sample = train_dataset[0]
    if resume_payload is not None:
        kwargs = dict(resume_metadata.get("model_kwargs") or {})
        if not kwargs:
            raise ValueError("resume checkpoint metadata must include model_kwargs.")
        expected_channels = {
            "coded_sp_channels": int(first_sample["coded_sp"].shape[0]),
            "condition_channels": int(first_sample["condition"].shape[0]),
        }
        for key, expected in expected_channels.items():
            actual = int(kwargs.get(key, -1))
            if actual != expected:
                raise ValueError(
                    f"resume checkpoint {key}={actual} does not match dataset {expected}."
                )
        if "model_state_dict" not in resume_payload:
            raise ValueError("resume checkpoint must include model_state_dict.")
        if "optimizer_state_dict" not in resume_payload:
            raise ValueError("resume checkpoint must include optimizer_state_dict.")
    else:
        kwargs = _model_kwargs(
            model_config,
            target_channels=int(first_sample["coded_sp"].shape[0]),
            condition_channels=int(first_sample["condition"].shape[0]),
        )
    model = build_speech_flow(**kwargs).to(device)
    optimizer = build_optimizer(model, effective_train_config)
    if resume_payload is not None:
        model.load_state_dict(resume_payload["model_state_dict"])
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])

    resume_previous_steps = 0
    if resume_payload is not None:
        resume_previous_steps = int(
            resume_metadata.get("total_steps", resume_metadata.get("num_steps", 0))
        )
    total_steps = resume_previous_steps + num_steps

    started = time.time()
    loss_csv_path = output_dir / "loss.csv"
    progress_file = loss_csv_path.open("w", newline="", encoding="utf-8")
    progress_writer = csv.DictWriter(progress_file, fieldnames=["step", "loss"])
    progress_writer.writeheader()
    progress_file.flush()
    last_progress_step = 0

    def report_progress(entry: dict[str, float | int]) -> None:
        """Write one progress row and advance the progress bar."""

        nonlocal last_progress_step
        step = int(entry["step"])
        loss = float(entry["loss"])
        progress_writer.writerow({"step": step, "loss": loss})
        progress_file.flush()
        progress_bar.set_postfix(loss=f"{loss:.6f}", refresh=False)
        progress_bar.update(step - last_progress_step)
        last_progress_step = step

    with tqdm(
        total=num_steps,
        desc="train_speech",
        unit="step",
        dynamic_ncols=True,
        file=sys.stdout,
    ) as progress_bar:
        history = train_speech_flow(
            model,
            optimizer,
            train_loader,
            device=device,
            num_steps=num_steps,
            segment_frames=segment_frames,
            log_every=log_every,
            seed=seed,
            progress_callback=report_progress,
        )
    progress_file.close()
    elapsed_sec = time.time() - started

    model.eval()
    with torch.no_grad():
        valid_batch = move_batch_to_device(next(iter(valid_loader)), device)
        valid_loss = float(speech_nll_loss(model, valid_batch).detach().cpu())

    write_csv_rows(output_dir / "loss.csv", history, fieldnames=["step", "loss"])
    plot_loss_curve(history, output_dir / "loss_curve.png")
    checkpoint_path = output_dir / "checkpoint.pt"
    metadata = {
        "batch_size": batch_size,
        "condition": condition,
        "condition_components": list(condition_spec.components),
        "data_config": metadata_data_config,
        "device": str(device),
        "feature_manifest": feature_manifest,
        "model_config": metadata_model_config,
        "model_kwargs": kwargs,
        "num_steps": num_steps,
        "seed": seed,
        "segment_frames": segment_frames,
        "speakers": list(speakers) if speakers is not None else None,
        "statistics_split": statistics_split,
        "total_steps": total_steps,
        "train_config": effective_train_config,
        "train_split": train_split,
        "valid_split": valid_split,
    }
    if args.resume:
        metadata["resume_checkpoint"] = str(args.resume)
        metadata["resume_previous_steps"] = resume_previous_steps
    save_checkpoint(
        checkpoint_path, model=model, optimizer=optimizer, metadata=metadata
    )

    metrics = {
        "batch_size": batch_size,
        "checkpoint": str(checkpoint_path),
        "condition": condition,
        "condition_components": list(condition_spec.components),
        "device": str(device),
        "elapsed_sec": elapsed_sec,
        "feature_manifest": feature_manifest,
        "final_logged_loss": float(history[-1]["loss"]),
        "initial_logged_loss": float(history[0]["loss"]),
        "num_steps": num_steps,
        "segment_frames": segment_frames,
        "speakers": list(speakers) if speakers is not None else None,
        "torch_version": torch.__version__,
        "total_steps": total_steps,
        "train_utterances": len(train_dataset),
        "valid_loss": valid_loss,
        "valid_utterances": len(valid_dataset),
    }
    if args.resume:
        metrics["resume_checkpoint"] = str(args.resume)
        metrics["resume_previous_steps"] = resume_previous_steps
    if device.type == "cuda":
        metrics["cuda_device"] = torch.cuda.get_device_name(device)
    write_json(output_dir / "metrics.json", metrics)

    print(f"output_dir={output_dir}")
    print(f"checkpoint={checkpoint_path}")
    print(f"condition={condition}")
    print(f"device={device}")
    if speakers is not None:
        print(f"speakers={','.join(speakers)}")
    print(f"steps={num_steps}")
    if args.resume:
        print(f"resume_checkpoint={args.resume}")
        print(f"resume_previous_steps={resume_previous_steps}")
        print(f"total_steps={total_steps}")
    print(f"final_logged_loss={metrics['final_logged_loss']}")
    print(f"valid_loss={valid_loss}")
    print(f"elapsed_sec={elapsed_sec:.3f}")


if __name__ == "__main__":
    main()
