"""Seeding helpers for reproducible training and data loading."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs and set deterministic flags.

    Parameters
    ----------
    seed : int
        Seed value for all RNGs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_seed_worker(seed: int):
    """Create a worker_init_fn that seeds NumPy and random per worker.

    Parameters
    ----------
    seed : int
        Base seed used for worker-specific seeds.
    """

    def _seed_worker(worker_id: int) -> None:
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    return _seed_worker


def make_torch_generator(seed: int) -> torch.Generator:
    """Create a torch.Generator seeded for deterministic shuffling.

    Parameters
    ----------
    seed : int
        Seed value for the generator.

    Returns
    -------
    generator : torch.Generator
        Seeded generator instance.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
