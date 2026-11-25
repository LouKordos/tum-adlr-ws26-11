"""Windowed trajectory dataset for the robotic world model."""

from __future__ import annotations

import re
from collections import OrderedDict
from collections.abc import Collection, Mapping, Sequence
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
        included_episodes: Collection[str] | None = None,
        preload: bool = True,
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
        # Episode ids ("<file name>:<demo>") to index, e.g. one split's episodes. None = all episodes.
        self.included_episodes = None if included_episodes is None else set(included_episodes)

        self._files = self._discover_files()
        self.reset_worker()

        # The first element in each window tuple is an index into self._episodes (not the HDF5 file index)
        self._episodes, self._windows = self._build_index()
        if not self._windows:
            raise ValueError(
                f"No windows of length {self.sequence_length} found in {self.data_dir}. "
                "Collect longer episodes or reduce history/forecast horizons."
            )

        # preload=True reads every indexed episode into RAM once, so __getitem__ never touches the disk. Without it,
        # random access re-reads (almost) a whole episode per sample, because the LRU cache rarely hits.
        # Layout: (states [N, state_dim], actions [N, action_dim], start row of each episode in those tensors).
        self._preloaded: tuple[Tensor, Tensor, list[int]] | None = None
        if preload:
            self._preload()

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
                    # The id includes the file name because every recording has its own demo_0, demo_1, ...
                    if self.included_episodes is not None and f"{path.name}:{demo}" not in self.included_episodes:
                        continue

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

    def _read_episode(self, ep_idx: int) -> tuple[Tensor, Tensor]:
        """Read concatenated ``(state, action)`` tensors for ``ep_idx`` from disk."""
        ep = self._episodes[ep_idx]
        handle = self._handles.get(ep.file_idx)
        # Caches handle for faster access without lookup
        if handle is None:
            handle = h5py.File(self._files[ep.file_idx], "r")
            self._handles[ep.file_idx] = handle

        group = handle.get("data", handle)[ep.demo]
        state = torch.cat([self._read_field(group, f) for f in self.state_fields], dim=-1)
        action = self._read_field(group, self.action_field)
        return state, action

    def _preload(self) -> None:
        """Copy all indexed episodes into two preallocated tensors (forked DataLoader workers share them)."""
        offsets = [0]
        for ep in self._episodes:
            offsets.append(offsets[-1] + ep.length)
        states = torch.empty(offsets[-1], self.state_dim)
        actions = torch.empty(offsets[-1], self.action_dim)
        for ep_idx in range(len(self._episodes)):
            states[offsets[ep_idx] : offsets[ep_idx + 1]], actions[offsets[ep_idx] : offsets[ep_idx + 1]] = (
                self._read_episode(ep_idx)
            )
        self.close()  # HDF5 handles are no longer needed
        self._preloaded = (states, actions, offsets)

    def _load_episode(self, ep_idx: int) -> tuple[Tensor, Tensor]:
        """Fetch concatenated ``(state, action)`` tensors for ``ep_idx`` from RAM, LRU cache or disk."""
        if self._preloaded is not None:
            states, actions, offsets = self._preloaded
            return states[offsets[ep_idx] : offsets[ep_idx + 1]], actions[offsets[ep_idx] : offsets[ep_idx + 1]]

        cached = self._episode_cache.get(ep_idx)
        # Caches episode for faster access without fs read
        if cached is not None:
            self._episode_cache.move_to_end(ep_idx)
            return cached

        pair = self._read_episode(ep_idx)
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
            # No shift here, because Isaac Lab stores each action in the same row as the state it results in
            # (actions[k] -> states[k]), but SystemDynamicsEnsemble does the shift itself
            "action": action[start:end],
        }
        if self.return_meta:
            ep = self._episodes[ep_idx]
            sample["episode_index"] = torch.tensor(ep_idx, dtype=torch.long)
            sample["window_start"] = torch.tensor(start, dtype=torch.long)
            sample["success"] = torch.tensor(ep.success, dtype=torch.bool)
            sample["episode_length"] = torch.tensor(ep.length, dtype=torch.long)
        return sample

    def compute_statistics(self) -> dict[str, tuple[Tensor, Tensor]]:
        """Per-dimension mean/std of states and actions over all timesteps of the indexed episodes.

        Computed per timestep (not per window), so overlapping windows don't over-weight the middle of episodes.
        Use the train split only, so no val/test information leaks into the normalization.
        """
        if self._preloaded is not None:
            states, actions, _ = self._preloaded
        else:
            states, actions = (torch.cat(x) for x in zip(*(self._load_episode(i) for i in range(len(self._episodes)))))
        return {"state": (states.mean(dim=0), states.std(dim=0)), "action": (actions.mean(dim=0), actions.std(dim=0))}

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
