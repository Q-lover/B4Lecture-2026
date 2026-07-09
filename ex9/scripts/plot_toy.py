"""Plot toy target/generated samples and loss curves."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from nf_assignment.toy.data import make_toy_distribution
from nf_assignment.toy.model import build_realnvp_2d
from nf_assignment.toy.visualize import (
    plot_density_heatmaps,
    plot_loss_curve,
    plot_sample_comparison,
    plot_warped_base_grid,
)
from nf_assignment.training.checkpoints import load_checkpoint
from nf_assignment.utils.io import ensure_dir, load_yaml, write_json


def _resolve_device(requested: str) -> torch.device:
    """Resolve ``auto`` or an explicit PyTorch device string."""

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device was requested, but torch.cuda.is_available() is False."
        )
    return device


def _read_loss_history(path: str | Path) -> list[dict[str, float | int]]:
    """Read ``loss.csv`` rows for plotting."""

    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        return [{"step": int(row["step"]), "loss": float(row["loss"])} for row in rows]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for toy plotting."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", default="configs/toy/data.yaml")
    parser.add_argument("--model-config", default="configs/toy/model.yaml")
    parser.add_argument("--train-output-dir", default="runs/toy_realnvp")
    parser.add_argument("--sample-output-dir", default="outputs/toy_realnvp")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--density-grid-size", type=int, default=180)
    parser.add_argument("--base-grid-size", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    """Create toy loss, sample, density, and warped-grid plots."""

    args = parse_args()
    data_config = load_yaml(args.data_config)
    model_config = load_yaml(args.model_config)

    train_output_dir = Path(args.train_output_dir)
    sample_output_dir = Path(args.sample_output_dir)
    output_dir = ensure_dir(args.output_dir or sample_output_dir)
    checkpoint_path = (
        Path(args.checkpoint) if args.checkpoint else train_output_dir / "checkpoint.pt"
    )
    device = _resolve_device(args.device)

    started = time.time()
    history = _read_loss_history(train_output_dir / "loss.csv")
    target_samples = np.load(sample_output_dir / "target_samples.npy")
    generated_samples = np.load(sample_output_dir / "generated_samples.npy")

    noise = data_config.get("noise")
    target = make_toy_distribution(
        str(data_config.get("dataset", "moons")),
        noise=None if noise is None else float(noise),
    ).to(device)
    if not hasattr(target, "log_prob"):
        raise ValueError("Density plots require a target distribution with log_prob.")

    hidden_dims = tuple(int(v) for v in model_config.get("hidden_dims", [64, 64]))
    model = build_realnvp_2d(
        num_layers=int(model_config.get("num_layers", 16)),
        hidden_dims=hidden_dims,
        init_zeros=True,
        target=target,
    ).to(device)
    payload = load_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    plot_loss_curve(history, output_dir / "loss_curve.png")
    plot_sample_comparison(
        target_samples, generated_samples, output_dir / "samples.png"
    )
    plot_density_heatmaps(
        target.log_prob,
        model.log_prob,
        output_dir / "density_heatmap.png",
        device=device,
        grid_size=args.density_grid_size,
    )
    plot_warped_base_grid(
        model.base.log_prob,
        model.log_prob,
        model.forward,
        output_dir / "warped_grid.png",
        device=device,
        density_grid_size=args.density_grid_size,
        base_grid_size=args.base_grid_size,
    )

    elapsed_sec = time.time() - started
    metrics = {
        "base_grid_size": args.base_grid_size,
        "checkpoint": str(checkpoint_path),
        "density_grid_size": args.density_grid_size,
        "device": str(device),
        "elapsed_sec": elapsed_sec,
        "generated_samples": int(len(generated_samples)),
        "output_dir": str(output_dir),
        "target_samples": int(len(target_samples)),
        "torch_version": torch.__version__,
    }
    if device.type == "cuda":
        metrics["cuda_device"] = torch.cuda.get_device_name(device)
    write_json(output_dir / "plot_metrics.json", metrics)

    print(f"output_dir={output_dir}")
    print(f"checkpoint={checkpoint_path}")
    print(f"device={device}")
    print(f"density_grid_size={args.density_grid_size}")
    print(f"base_grid_size={args.base_grid_size}")
    print(f"elapsed_sec={elapsed_sec:.3f}")


if __name__ == "__main__":
    main()
