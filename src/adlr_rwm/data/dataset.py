"""Windowed trajectory dataset for the robotic world model."""

from __future__ import annotations

import re
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

# Robot joint kinematics + object SE(3) pose/velocity.
DEFAULT_STATE_FIELDS: tuple[str, ...] = (
    "states/articulation/robot/joint_position",
    "states/articulation/robot/joint_velocity",
    "states/rigid_object/object/root_pose",
    "states/rigid_object/object/root_velocity",
)

DEFAULT_ACTION_FIELD = "actions"

# Isaac Lab stores quaternions as (x, y, z, w). Since q and -q represent the
# same rotation, signs are aligned per episode to keep targets continuous.
DEFAULT_QUATERNION_SLICES: Mapping[str, slice] = {
    "actions": slice(3, 7),
    "processed_actions": slice(3, 7),
    "states/articulation/robot/root_pose": slice(3, 7),
    "states/rigid_object/object/root_pose": slice(3, 7),
}

_DEMO_RE = re.compile(r"demo_(\d+)$")


@dataclass(frozen=True, slots=True)
class _Episode:
    """Metadata for a single valid demonstration episode."""

    file_idx: int
    demo: str
    length: int
    success: bool


class RWMDataset(Dataset):
    """Map-style dataset yielding fixed-length ``(state, action)`` windows."""

    def __init__(
        self,
        data_dir: str | Path,
        history_horizon: int = 8,
        forecast_horizon: int = 3,
        *,
        state_fields: Sequence[str] = DEFAULT_STATE_FIELDS,
        action_field: str = DEFAULT_ACTION_FIELD,
        stride: int = 1,
        only_successful: bool = False,
        max_cached_episodes: int = 128,
        return_meta: bool = False,
        quaternion_slices: Mapping[str, slice] | None = DEFAULT_QUATERNION_SLICES,
    ):
        """Validate config, discover HDF5 recordings, and build the window index."""
        if history_horizon <= 0 or forecast_horizon <= 0:
            raise ValueError("history_horizon and forecast_horizon must be positive")
        if stride <= 0:
            raise ValueError("stride must be positive")

        self.data_dir = Path(data_dir)
        self.history_horizon = history_horizon
        self.forecast_horizon = forecast_horizon
        self.sequence_length = history_horizon + forecast_horizon
        self.state_fields = tuple(state_fields)
        self.action_field = action_field
        self.stride = stride
        self.only_successful = only_successful
        self.max_cached_episodes = max_cached_episodes
        self.return_meta = return_meta
        self.quaternion_slices = dict(quaternion_slices or {})

        self._files = self._discover_files()
        self.reset_worker()

        self._episodes, self._windows = self._build_index()
        if not self._windows:
            raise ValueError(
                f"No windows of length {self.sequence_length} found in {self.data_dir}. "
                "Collect longer episodes or reduce history/forecast horizons."
            )

    def _discover_files(self) -> list[Path]:
        """Return ``[data_dir]`` if given a file, or all ``.hdf5``/``.h5`` files under it."""
        if self.data_dir.is_file():
            return [self.data_dir]

        files = sorted(p for p in self.data_dir.rglob("*") if p.suffix in {".hdf5", ".h5"})
        if not files:
            raise FileNotFoundError(f"No .hdf5/.h5 recordings found under {self.data_dir}")
        return files

    def _build_index(self) -> tuple[list[_Episode], list[tuple[int, int]]]:
        """Inspect HDF5 headers without loading trajectories to build ``(episodes, windows)``."""
        episodes: list[_Episode] = []
        windows: list[tuple[int, int]] = []
        state_dim: int | None = None
        action_dim: int | None = None
        required_fields = (*self.state_fields, self.action_field)

        for file_idx, path in enumerate(self._files):
            with h5py.File(path, "r") as handle:
                root = handle.get("data", handle)
                demos = sorted(
                    (name for name in root if _DEMO_RE.match(name)),
                    key=lambda name: int(_DEMO_RE.match(name).group(1)),  # type: ignore[union-attr]
                )

                for demo in demos:
                    group = root[demo]
                    success = bool(group.attrs.get("success", False))
                    if self.only_successful and not success:
                        continue

                    missing = [f for f in required_fields if f not in group]
                    if missing:
                        raise KeyError(
                            f"{path}:{demo} is missing {missing}. Available datasets: {sorted(group.keys())}"
                        )

                    num_samples = int(group.attrs.get("num_samples", group[self.action_field].shape[0]))
                    lengths = {f: int(group[f].shape[0]) for f in required_fields}
                    if any(n != num_samples for n in lengths.values()):
                        raise ValueError(
                            f"{path}:{demo} has inconsistent lengths: num_samples={num_samples}, fields={lengths}"
                        )
                    if num_samples < self.sequence_length:
                        continue

                    ep_state_dim = sum(int(group[f].shape[-1]) for f in self.state_fields)
                    ep_action_dim = int(group[self.action_field].shape[-1])
                    state_dim = ep_state_dim if state_dim is None else state_dim
                    action_dim = ep_action_dim if action_dim is None else action_dim
                    if (ep_state_dim, ep_action_dim) != (state_dim, action_dim):
                        raise ValueError(
                            f"{path}:{demo} has state/action dims ({ep_state_dim}, {ep_action_dim}), "
                            f"expected ({state_dim}, {action_dim}). Mixing incompatible recordings is not supported."
                        )

                    ep_idx = len(episodes)
                    episodes.append(_Episode(file_idx, demo, num_samples, success))
                    windows.extend(
                        (ep_idx, start) for start in range(0, num_samples - self.sequence_length + 1, self.stride)
                    )

        self.state_dim = state_dim or 0
        self.action_dim = action_dim or 0
        return episodes, windows

    def _read_field(self, group: h5py.Group, field: str) -> Tensor:
        """Read one dataset field as float32 and keep consecutive quaternions sign-consistent."""
        array = np.asarray(group[field][:], dtype=np.float32)
        quat_slice = self.quaternion_slices.get(field)
        if quat_slice is not None and array.shape[-1] >= quat_slice.stop and len(array) >= 2:
            quat = array[:, quat_slice]
            flips = np.cumprod(np.where(np.sum(quat[1:] * quat[:-1], axis=-1) < 0.0, -1.0, 1.0))
            quat[1:] *= flips[:, None]
        return torch.from_numpy(array)

    def _load_episode(self, ep_idx: int) -> tuple[Tensor, Tensor]:
        """Fetch concatenated ``(state, action)`` tensors for ``ep_idx`` from LRU cache or disk."""
        cached = self._episode_cache.get(ep_idx)
        # Caches episode for faster access without fs read
        if cached is not None:
            self._episode_cache.move_to_end(ep_idx)
            return cached

        ep = self._episodes[ep_idx]
        handle = self._handles.get(ep.file_idx)
        # Caches handle for faster access without lookup
        if handle is None:
            handle = h5py.File(self._files[ep.file_idx], "r")
            self._handles[ep.file_idx] = handle

        group = handle.get("data", handle)[ep.demo]
        state = torch.cat([self._read_field(group, f) for f in self.state_fields], dim=-1)
        action = self._read_field(group, self.action_field)

        pair = (state, action)
        # LRU cache logic
        self._episode_cache[ep_idx] = pair
        if self.max_cached_episodes is not None and len(self._episode_cache) > self.max_cached_episodes:
            self._episode_cache.popitem(last=False)
        return pair

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Slice a fixed-length ``(state, action)`` window (and optional metadata) by window index."""
        ep_idx, start = self._windows[idx]
        end = start + self.sequence_length
        state, action = self._load_episode(ep_idx)

        sample = {
            "state": state[start:end],
            "action": action[start:end],
        }
        if self.return_meta:
            ep = self._episodes[ep_idx]
            sample["episode_index"] = torch.tensor(ep_idx, dtype=torch.long)
            sample["window_start"] = torch.tensor(start, dtype=torch.long)
            sample["success"] = torch.tensor(ep.success, dtype=torch.bool)
            sample["episode_length"] = torch.tensor(ep.length, dtype=torch.long)
        return sample

    def reset_worker(self) -> None:
        """Reset per-process HDF5 handles and LRU cache (called on init and worker spawn/fork)."""
        self._handles: dict[int, h5py.File] = {}
        self._episode_cache: OrderedDict[int, tuple[Tensor, Tensor]] = OrderedDict()

    def close(self) -> None:
        """Close all open HDF5 file handles held by the current process."""
        for handle in self._handles.values():
            with suppress(OSError, ValueError):
                handle.close()
        self.reset_worker()

    def __getstate__(self) -> dict:
        """Strip unpicklable HDF5 handles and tensor cache before sending to spawned workers."""
        state = self.__dict__.copy()
        state.pop("_handles", None)
        state.pop("_episode_cache", None)
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore dataset state in a spawned worker with fresh handles and cache."""
        self.__dict__.update(state)
        self.reset_worker()

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()
