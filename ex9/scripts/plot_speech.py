"""Plot speech spectrograms, waveforms, and loss curves."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from nf_assignment.speech.visualize import (
    plot_coded_sp_comparison,
    plot_loss_curve,
    plot_mel_spectrogram_comparison,
    plot_spectral_envelope_comparison,
)
from nf_assignment.utils.io import ensure_dir, read_csv_rows, write_json


def _read_loss_history(path: str | Path) -> list[dict[str, float | int]]:
    """Read ``loss.csv`` rows for speech loss plotting."""

    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        return [{"loss": float(row["loss"]), "step": int(row["step"])} for row in rows]


def _read_sample_manifest(sample_output_dir: Path) -> list[dict]:
    """Read the speech sample manifest CSV from an output directory."""

    return read_csv_rows(sample_output_dir / "sample_manifest.csv")


def _resolve_existing_path(
    value: str | None, *, sample_output_dir: Path
) -> Path | None:
    """Resolve an optional manifest path relative to cwd or the sample directory."""

    if not value:
        return None
    path = Path(value)
    candidates = (
        [path] if path.is_absolute() else [Path.cwd() / path, sample_output_dir / path]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _read_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Read audio as a mono waveform array shaped ``[samples]``."""

    import soundfile as sf

    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if waveform.ndim == 2:
        waveform = np.mean(waveform, axis=1)
    return np.asarray(waveform, dtype=np.float32), int(sample_rate)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for speech plotting."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-output-dir", default="runs/speech_world_flow")
    parser.add_argument("--sample-output-dir", default="outputs/speech_world_flow")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-plots", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    """Create speech loss and sample-review plots."""

    args = parse_args()
    train_output_dir = Path(args.train_output_dir)
    sample_output_dir = Path(args.sample_output_dir)
    output_dir = ensure_dir(args.output_dir or sample_output_dir)

    started = time.time()
    history = _read_loss_history(train_output_dir / "loss.csv")
    plot_loss_curve(history, output_dir / "loss_curve.png")

    rows = _read_sample_manifest(sample_output_dir)
    plotted = 0
    plotted_mel = 0
    plotted_sp = 0
    for row in rows[: args.max_plots]:
        plot_id = row.get("sample_id", row["utterance_id"])
        target = np.load(row["target_coded_sp_path"])
        generated = np.load(row["generated_coded_sp_path"])
        plot_coded_sp_comparison(
            target,
            generated,
            output_dir / f"{plot_id}_coded_sp.png",
        )
        plotted += 1
        target_sp_path = _resolve_existing_path(
            row.get("target_sp_path"),
            sample_output_dir=sample_output_dir,
        )
        generated_sp_path = _resolve_existing_path(
            row.get("generated_sp_path"),
            sample_output_dir=sample_output_dir,
        )
        if target_sp_path is not None and generated_sp_path is not None:
            plot_spectral_envelope_comparison(
                np.load(target_sp_path),
                np.load(generated_sp_path),
                output_dir / f"{plot_id}_sp.png",
            )
            plotted_sp += 1
        target_wav_path = _resolve_existing_path(
            row.get("target_wav_path"),
            sample_output_dir=sample_output_dir,
        )
        generated_wav_path = _resolve_existing_path(
            row.get("generated_wav_path") or row.get("wav_path"),
            sample_output_dir=sample_output_dir,
        )
        if target_wav_path is not None and generated_wav_path is not None:
            target_waveform, target_sample_rate = _read_audio(target_wav_path)
            generated_waveform, generated_sample_rate = _read_audio(generated_wav_path)
            plot_mel_spectrogram_comparison(
                target_waveform,
                target_sample_rate,
                generated_waveform,
                generated_sample_rate,
                output_dir / f"{plot_id}_mel.png",
            )
            plotted_mel += 1

    elapsed_sec = time.time() - started
    write_json(
        output_dir / "plot_metrics.json",
        {
            "elapsed_sec": elapsed_sec,
            "loss_curve": str(output_dir / "loss_curve.png"),
            "plotted_mel_utterances": plotted_mel,
            "plotted_sp_utterances": plotted_sp,
            "plotted_utterances": plotted,
            "sample_output_dir": str(sample_output_dir),
            "train_output_dir": str(train_output_dir),
        },
    )

    print(f"output_dir={output_dir}")
    print(f"plotted_utterances={plotted}")
    print(f"elapsed_sec={elapsed_sec:.3f}")


if __name__ == "__main__":
    main()
