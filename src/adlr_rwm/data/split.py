"""Assignment of a recording's episodes to train/val/test."""

from pathlib import Path

import h5py
import numpy as np


def make_split(recording: Path, val_fraction: float, test_fraction: float, seed: int) -> dict[str, list[str]]:
    """Randomly assign every episode of a single HDF5 recording to train/val/test."""
    with h5py.File(recording, "r") as handle:
        demos = sorted(name for name in handle["data"] if name.startswith("demo_"))
    episode_ids = [f"{recording.name}:{demo}" for demo in demos]  # same id format as RWMDataset(included_episodes)

    # Sorted ids + fixed seed: the split only depends on which episodes exist, not on the order they are read in
    shuffled = [episode_ids[i] for i in np.random.default_rng(seed).permutation(len(episode_ids))]
    n_val, n_test = round(len(shuffled) * val_fraction), round(len(shuffled) * test_fraction)
    split = {
        "train": sorted(shuffled[n_val + n_test :]),
        "val": sorted(shuffled[:n_val]),
        "test": sorted(shuffled[n_val : n_val + n_test]),
    }

    return split
