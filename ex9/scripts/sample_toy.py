"""Generate samples from a trained toy normalizing-flow model."""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from nf_assignment.toy.data import make_toy_distribution
from nf_assignment.toy.model import build_realnvp_2d
from nf_assignment.toy.sample import sample_model
from nf_assignment.training.checkpoints import load_checkpoint
from nf_assignment.utils.io import ensure_dir, load_yaml, write_json
from nf_assignment.utils.seed import set_seed


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


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for toy sampling."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", default="configs/toy/data.yaml")
    parser.add_argument("--model-config", default="configs/toy/model.yaml")
    parser.add_argument("--sample-config", default="configs/toy/sample.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """Load a toy checkpoint and write target/generated sample arrays."""

    args = parse_args()
    data_config = load_yaml(args.data_config)
    model_config = load_yaml(args.model_config)
    sample_config = load_yaml(args.sample_config)

    seed = args.seed if args.seed is not None else int(sample_config.get("seed", 0))
    set_seed(seed)

    device = _resolve_device(args.device)
    output_dir = ensure_dir(
        args.output_dir or sample_config.get("output_dir", "outputs/toy_realnvp")
    )
    checkpoint_path = args.checkpoint or sample_config.get(
        "checkpoint", "runs/toy_realnvp/checkpoint.pt"
    )

    noise = data_config.get("noise")
    target = make_toy_distribution(
        str(data_config.get("dataset", "moons")),
        noise=None if noise is None else float(noise),
    ).to(device)
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

    num_samples = args.num_samples or int(sample_config.get("num_samples", 4096))
    started = time.time()
    with torch.no_grad():
        target_samples = target.sample(num_samples).detach().cpu().numpy()
        generated_samples = sample_model(model, num_samples).detach().cpu().numpy()
    elapsed_sec = time.time() - started

    np.save(output_dir / "target_samples.npy", target_samples)
    np.save(output_dir / "generated_samples.npy", generated_samples)

    metrics = {
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "elapsed_sec": elapsed_sec,
        "generated_samples": int(len(generated_samples)),
        "num_samples": num_samples,
        "output_dir": str(output_dir),
        "seed": seed,
        "target_samples": int(len(target_samples)),
        "torch_version": torch.__version__,
    }
    if device.type == "cuda":
        metrics["cuda_device"] = torch.cuda.get_device_name(device)
    write_json(output_dir / "sample_metrics.json", metrics)

    print(f"output_dir={output_dir}")
    print(f"checkpoint={checkpoint_path}")
    print(f"device={device}")
    print(f"num_samples={num_samples}")
    print(f"elapsed_sec={elapsed_sec:.3f}")


if __name__ == "__main__":
    main()
