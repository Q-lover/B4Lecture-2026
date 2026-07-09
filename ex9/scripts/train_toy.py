"""Train the required toy normalizing-flow model."""

from __future__ import annotations

import argparse
import sys
import time

import torch
from nf_assignment.toy.data import TwoMoons, make_toy_distribution
from nf_assignment.toy.model import build_realnvp_2d
from nf_assignment.toy.train import train_forward_kld
from nf_assignment.toy.visualize import plot_loss_curve
from nf_assignment.training.checkpoints import save_checkpoint
from nf_assignment.utils.io import ensure_dir, load_yaml, write_csv_rows, write_json
from nf_assignment.utils.seed import set_seed
from tqdm import tqdm


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


def _build_optimizer(
    model: torch.nn.Module, train_config: dict
) -> torch.optim.Optimizer:
    """Build the toy optimizer from YAML training config."""

    optimizer_config = train_config.get("optimizer", {})
    name = str(optimizer_config.get("name", "adam")).lower()
    lr = float(optimizer_config.get("lr", 5e-4))
    weight_decay = float(optimizer_config.get("weight_decay", 0.0))
    if name != "adam":
        raise ValueError(f"Unsupported optimizer: {name}")
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for toy training."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", default="configs/toy/data.yaml")
    parser.add_argument("--model-config", default="configs/toy/model.yaml")
    parser.add_argument("--train-config", default="configs/toy/train.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """Train the toy flow and write loss, checkpoint, and metrics artifacts."""

    args = parse_args()
    data_config = load_yaml(args.data_config)
    model_config = load_yaml(args.model_config)
    train_config = load_yaml(args.train_config)

    seed = int(train_config.get("seed", data_config.get("seed", 0)))
    set_seed(seed)

    requested_device = args.device or str(train_config.get("device", "auto"))
    device = _resolve_device(requested_device)
    output_dir = ensure_dir(
        args.output_dir or train_config.get("output_dir", "runs/toy_realnvp")
    )

    noise = data_config.get("noise")
    target = make_toy_distribution(
        str(data_config.get("dataset", "moons")),
        noise=None if noise is None else float(noise),
    ).to(device)
    if not isinstance(target, TwoMoons):
        raise ValueError(
            "Stage06 validation currently expects a TwoMoons-compatible target."
        )

    hidden_dims = tuple(int(v) for v in model_config.get("hidden_dims", [64, 64]))
    model = build_realnvp_2d(
        num_layers=int(model_config.get("num_layers", 16)),
        hidden_dims=hidden_dims,
        init_zeros=True,
        target=target,
    ).to(device)
    optimizer = _build_optimizer(model, train_config)

    num_steps = args.num_steps or int(train_config.get("num_steps", 1200))
    batch_size = args.batch_size or int(train_config.get("batch_size", 512))
    log_every = int(train_config.get("log_every", 10))

    started = time.time()
    last_progress_step = 0

    def report_progress(entry: dict[str, float | int]) -> None:
        """Advance the progress bar when a training loss is logged."""

        nonlocal last_progress_step
        step = int(entry["step"])
        loss = float(entry["loss"])
        progress_bar.set_postfix(loss=f"{loss:.6f}", refresh=False)
        progress_bar.update(step - last_progress_step)
        last_progress_step = step

    with tqdm(
        total=num_steps,
        desc="train_toy",
        unit="step",
        dynamic_ncols=True,
        file=sys.stdout,
    ) as progress_bar:
        history = train_forward_kld(
            model,
            optimizer,
            target,
            batch_size=batch_size,
            num_steps=num_steps,
            log_every=log_every,
            progress_callback=report_progress,
        )
    elapsed_sec = time.time() - started

    write_csv_rows(output_dir / "loss.csv", history, fieldnames=["step", "loss"])
    plot_loss_curve(history, output_dir / "loss_curve.png", ylabel="forward KLD")

    checkpoint_path = output_dir / "checkpoint.pt"
    metadata = {
        "batch_size": batch_size,
        "data_config": data_config,
        "device": str(device),
        "model_config": model_config,
        "num_steps": num_steps,
        "seed": seed,
        "train_config": train_config,
    }
    save_checkpoint(
        checkpoint_path, model=model, optimizer=optimizer, metadata=metadata
    )

    metrics = {
        "batch_size": batch_size,
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "elapsed_sec": elapsed_sec,
        "final_logged_loss": float(history[-1]["loss"]),
        "hidden_dims": list(hidden_dims),
        "initial_logged_loss": float(history[0]["loss"]),
        "loss_delta": float(history[-1]["loss"]) - float(history[0]["loss"]),
        "num_layers": int(model_config.get("num_layers", 16)),
        "seed": seed,
        "steps": num_steps,
        "torch_version": torch.__version__,
    }
    if device.type == "cuda":
        metrics["cuda_device"] = torch.cuda.get_device_name(device)
    write_json(output_dir / "metrics.json", metrics)

    print(f"output_dir={output_dir}")
    print(f"device={device}")
    print(f"steps={num_steps}")
    print(f"batch_size={batch_size}")
    print(f"initial_logged_loss={metrics['initial_logged_loss']}")
    print(f"final_logged_loss={metrics['final_logged_loss']}")
    print(f"elapsed_sec={elapsed_sec:.3f}")


if __name__ == "__main__":
    main()
