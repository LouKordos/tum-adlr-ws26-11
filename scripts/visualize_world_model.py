"""Play back recorded episodes in Isaac Lab and overlay the world model's predictions as a transparent "ghost".

What the video shows
--------------------
* Solid robot + cube: the recorded episode (ground truth), exactly as it was recorded.
* Transparent, tinted robot + cube (the "ghost"): what the world model predicts for the same time step.
  If the model were perfect, the ghost would stay hidden inside the solid robot and cube the whole time.
  --no-ghost turns it off and only plays back the recording.

How it works
------------
1. The checkpoint (a .pt file written by scripts/train_world_model.py) contains the model weights, the normalization
   statistics and the full training config. Its run directory also contains split.json (the train/val/test episode
   lists). The config names the recording (.hdf5) the model was trained on, and the recording names the Isaac Lab task
   it was recorded in. That is why --checkpoint is the only required argument.
2. For every selected episode, the world model predicts the states from the recorded actions (see "Predictions").
3. We create the task's environment with 2 copies ("envs"): env 0 shows the recorded states, env 1 the predicted
   states. Nothing is simulated: every frame we write both states directly into the simulator (like replaying motion
   capture) and render one image. Physics would only get in the way here, because the prediction should be shown
   exactly as predicted, even if it is physically impossible (e.g. a floating cube).
4. The Newton GL visualizer (OpenGL, so it works without an RTX GPU) renders the frames, and ffmpeg turns them into
   one mp4 per episode.

Predictions
-----------
H = history_horizon, F = forecast_horizon of the checkpoint (both are read from the checkpoint's config).
The ghost shows what the model is trained and evaluated on. Every F steps the ghost is reset to the recording: the
model gets the last H recorded steps as history and predicts the next F steps on its own. You see the error grow
within each chunk and jump back to zero at the start of the next one.
The first H steps of an episode have no prediction (they are the history), so the ghost equals the recording there.

Examples
--------
    # 4 random test episodes with the ghost
    uv run python scripts/visualize_world_model.py --checkpoint outputs/<date>/<time>/best.pt

    # Two specific episodes, only the recording
    uv run python scripts/visualize_world_model.py --checkpoint outputs/<date>/<time>/best.pt \
        --no-ghost --episodes demo_12 demo_345

    # Watch it live in the viewer window while the videos are written
    uv run python scripts/visualize_world_model.py --checkpoint outputs/<date>/<time>/best.pt --window

Videos are written to <run dir>/videos/<checkpoint>_ghost/ or <checkpoint>_recording/ (inside outputs/, which git
ignores).
Recordings made before collect.py stored the task name need --task, e.g. --task ADLR-Lift-Cube-Franka-IK-Abs.

This script is deliberately self-contained: its settings are the command line flags below, not the Hydra configs in
src/adlr_rwm/configs, and it only reads the training code (dataset + model), it doesn't change it.
"""

import argparse
from pathlib import Path

# Isaac Lab's launcher has to be imported before most other Isaac Lab modules (same pattern as collect.py).
from isaaclab.app import add_launcher_args, launch_simulation

REPO_ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description="Overlay world model predictions on recorded episodes in Isaac Lab.")
parser.add_argument(
    "--checkpoint", type=Path, required=True, help="World model checkpoint (.pt) written by train_world_model.py."
)
parser.add_argument(
    "--split",
    choices=("train", "val", "test"),
    default="test",
    help="Split to pick random episodes from (default: test, i.e. episodes the model has never seen).",
)
parser.add_argument(
    "--episodes",
    nargs="+",
    default=None,
    help="Episodes to render, e.g. demo_12 demo_345. Default: --num_episodes random episodes of --split.",
)
parser.add_argument("--num_episodes", type=int, default=4, help="Number of random episodes (default: 4).")
parser.add_argument("--seed", type=int, default=0, help="Seed for picking the random episodes (default: 0).")
parser.add_argument(
    "--ghost",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Show the model's prediction as a ghost (default). --no-ghost only plays back the recording.",
)
parser.add_argument("--ghost_opacity", type=float, default=0.5, help="Ghost opacity (0-1).")
parser.add_argument(
    "--ghost_color", type=float, nargs=3, default=(0.15, 0.55, 1.0), help="Ghost RGB tint in [0, 1] (default: blue)."
)
parser.add_argument(
    "--fps", type=float, default=None, help="Video frame rate. Default: real time (one frame per recorded step)."
)
parser.add_argument(
    "--task", default=None, help="Isaac Lab task to play back in. Default: the task name stored in the recording."
)
parser.add_argument("--output_dir", type=Path, default=None, help="Where to write the mp4s (default: see docstring).")
parser.add_argument("--width", type=int, default=1920, help="Video width in pixels (default: 1920).")
parser.add_argument("--height", type=int, default=1080, help="Video height in pixels (default: 1080).")
parser.add_argument("--eye", type=float, nargs=3, default=None, help="Camera position [m]. Default: CAMERA below.")
parser.add_argument("--lookat", type=float, nargs=3, default=None, help="Camera target [m]. Default: CAMERA below.")
parser.add_argument(
    "--window",
    action="store_true",
    help="Open the viewer window to watch while rendering (don't resize it, that breaks the video). "
    "Default: render off-screen, only the mp4s are written.",
)
# Adds Isaac Lab's own flags, e.g. --device (default cuda:0)
add_launcher_args(parser)
args_cli = parser.parse_args()

import os

if not args_cli.window:
    # Without a display, the Newton viewer renders off-screen (EGL) instead of opening a window, and logs a warning
    # that says so. This has to happen before pyglet (the viewer's window library) is imported by the imports below.
    os.environ.pop("DISPLAY", None)
    os.environ.pop("WAYLAND_DISPLAY", None)

import json
import random
import re
import subprocess

import gymnasium as gym
import h5py
import isaaclab_tasks  # noqa: F401  (registers Isaac Lab's tasks with gym)
import numpy as np
import torch
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab_visualizers.newton import NewtonGLVisualizerCfg
from omegaconf import OmegaConf

import adlr_rwm.tasks  # noqa: F401  (registers our tasks, e.g. ADLR-Lift-Cube-Franka-IK-Abs, with gym)
from adlr_rwm.data import RWMDataset
from adlr_rwm.models import SystemDynamicsEnsemble

# Index of the env that shows the recording and of the env that shows the prediction ("ghost")
RECORDED_ENV = 0
GHOST_ENV = 1

# Camera: eye and lookat in [m], focal length of the lens in [mm] (Isaac Lab's default of 12 mm is very wide-angle and
# makes the robot tiny). The robot base is at the env origin and the robot faces +x; the table top is at z = 0 and the
# cube starts around x = 0.5 on it.
CAMERA = {"eye": (1.9, -1.5, 1.0), "lookat": (0.35, 0.0, 0.35), "focal_length": 24.0}


# --------------------------------------------------------------------------------------------------------------------
# Loading the checkpoint, the recording and the episodes
# --------------------------------------------------------------------------------------------------------------------


def load_world_model(checkpoint: dict, device: str) -> tuple[SystemDynamicsEnsemble, dict[str, tuple]]:
    """Rebuild the world model from a checkpoint and return it with its normalization statistics.

    The checkpoint stores the complete training config, so the model is built with exactly the same architecture,
    horizons and state/action dimensions it was trained with.
    """
    wm_cfg = OmegaConf.create(checkpoint["config"]["world_model"])
    wm_cfg.device = device  # the saved config says "cuda"; follow --device instead
    model = SystemDynamicsEnsemble(wm_cfg)
    model.load_state_dict(checkpoint["model"])
    model.eval()  # no dropout/batchnorm in the model today, but this is the correct mode for inference anyway

    # The model only understands normalized inputs: (x - mean) / std with the train split's statistics.
    norm = checkpoint["normalization"]
    if "state" in norm:
        # Current format of train_world_model.py: {"state": (mean, std), "action": (mean, std)}
        stats = {key: (mean, std) for key, (mean, std) in norm.items()}
    else:
        # Older checkpoints stored flat keys: {"state_mean": ..., "state_std": ..., "action_mean": ..., ...}
        stats = {key: (norm[f"{key}_mean"], norm[f"{key}_std"]) for key in ("state", "action")}
    stats = {key: (mean.to(device), std.to(device)) for key, (mean, std) in stats.items()}
    return model, stats


def find_recording(checkpoint: dict) -> Path:
    """Path of the .hdf5 recording the checkpoint was trained on (same lookup as train_world_model.py)."""
    training_cfg = checkpoint["config"]["world_model"]["training"]
    data_dir = Path(training_cfg["dataset"]["data_dir"])
    data_dir = data_dir if data_dir.is_absolute() else REPO_ROOT / data_dir
    # data_dir is either the recording itself or a directory; then the split config names the file it used
    return data_dir if data_dir.is_file() else data_dir / training_cfg["split"]["recording"]


def read_task_name(recording: Path) -> str | None:
    """Isaac Lab task name stored by the recorder in the file's attributes (empty in recordings made before
    collect.py set env_cfg.env_name)."""
    with h5py.File(recording, "r") as handle:
        return json.loads(handle["data"].attrs["env_args"]).get("env_name") or None


def choose_episodes(split: dict[str, list[str]], recording: Path) -> list[str]:
    """Episode ids ("<recording file name>:demo_<n>", the format of split.json) to render."""
    if args_cli.episodes:
        return [f"{recording.name}:{demo}" for demo in args_cli.episodes]
    candidates = split[args_cli.split]
    return random.Random(args_cli.seed).sample(candidates, k=min(args_cli.num_episodes, len(candidates)))


def read_recorded_states(episode: h5py.Group) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
    """Full recorded scene state of one episode: {asset type: {asset name: {quantity: tensor [T, dim]}}}.

    Isaac Lab's recorder stores scene.get_state(is_relative=True) after every step under
    states/<asset type>/<asset name>/<quantity>, e.g. states/articulation/robot/joint_position. Reading it back
    into the same nesting gives exactly what InteractiveScene.reset_to() expects. It contains everything in the
    scene, also quantities the world model doesn't predict (e.g. the robot base pose).
    """
    return {
        asset_type: {
            asset_name: {quantity: torch.from_numpy(data[:]) for quantity, data in asset.items()}
            for asset_name, asset in assets.items()
        }
        for asset_type, assets in episode["states"].items()
    }


# --------------------------------------------------------------------------------------------------------------------
# World model predictions
# --------------------------------------------------------------------------------------------------------------------


@torch.no_grad()
def rollout(model: SystemDynamicsEnsemble, states: torch.Tensor, actions: torch.Tensor, steps: int) -> torch.Tensor:
    """Open-loop prediction of `steps` states after a history of H recorded states, for a batch of windows.

    This is SystemDynamicsEnsemble.forecast() with a free number of steps instead of always forecast_horizon (a copy,
    so the training code stays untouched). Same indexing, so the predictions are exactly what evaluation measures:
    * states:  [B, H, state_dim] normalized, the recorded history
    * actions: [B, H + steps, action_dim] normalized. actions[k] is the action that leads to states[k], which is how
      Isaac Lab records them, so action H + i is the one that produces predicted step i.
    * returns: [B, steps, state_dim] normalized predictions (ensemble mean), i.e. rows H .. H + steps - 1 of the
      window. Each prediction is fed back as the input for the next step ("autoregressive").
    """
    H = model.history_horizon
    recurrent = model.base_cfg.type in ("rnn", "rssm")
    model.reset()  # clear the RNN memory, otherwise it leaks in from the previous call
    x_state = states
    predictions = []
    for i in range(steps):
        if recurrent and i > 0:
            # The RNN remembers the history in its hidden state, so after the first step it only needs the newest
            # state (x_state, the previous prediction) and the action that leads to the next state
            x_action = actions[:, H + i : H + i + 1]
        else:
            # First step of the RNN, or every step of the MLP: the whole H-step window. Each state is paired with the
            # action that follows it (states[k] with actions[k + 1]), hence the offset of 1.
            x_action = actions[:, i + 1 : H + i + 1]
        mean, *_ = model(x_state, x_action)  # forward() returns (ensemble mean, aleatoric, epistemic, aux...)
        predictions.append(mean)
        if recurrent:
            x_state = mean.unsqueeze(1)
        else:
            # The MLP's window slides: drop the oldest state, append the prediction
            x_state = torch.cat([x_state[:, 1:], mean.unsqueeze(1)], dim=1)
    model.reset()
    return torch.stack(predictions, dim=1)


def predict_episode(
    model: SystemDynamicsEnsemble, stats: dict, states: torch.Tensor, actions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predicted state for every row of one episode, in physical units (not normalized).

    Args:
        states: [T, state_dim] recorded states of the episode (as the dataset gives them to the model in training).
        actions: [T, action_dim] recorded actions.

    Returns:
        predicted: [T, state_dim]. Rows without a prediction (the first H rows) are copies of the recorded states.
        has_prediction: [T] bool, which rows were predicted.
    """
    H, F = model.history_horizon, model.forecast_horizon
    T = len(states)
    device = stats["state"][0].device

    # Normalize the inputs exactly like train_world_model.py does
    states_n = (states.to(device) - stats["state"][0]) / stats["state"][1]
    actions_n = (actions.to(device) - stats["action"][0]) / stats["action"][1]

    # Every prediction window starts after H recorded steps of history: a new window every F steps, each predicts
    # F steps (see "Predictions" in the module docstring)
    starts, steps = list(range(H, T, F)), F

    # The last chunk can reach past the end of the episode, but the model wants `steps` actions for every window.
    # Repeat the last action as padding: a prediction only depends on actions up to its own row, so the padding only
    # affects predictions past the episode end, which are thrown away below.
    actions_n = torch.cat([actions_n, actions_n[-1:].expand(steps, -1)])

    # Batch all windows of the episode into one call: [num_windows, H, state_dim] and [num_windows, H + steps, ...]
    state_windows = torch.stack([states_n[start - H : start] for start in starts])
    action_windows = torch.stack([actions_n[start - H : start + steps] for start in starts])
    predictions = rollout(model, state_windows, action_windows, steps)

    # Undo the normalization and write every window's predictions into its rows of the episode
    predictions = predictions * stats["state"][1] + stats["state"][0]
    predicted = states.clone()
    has_prediction = torch.zeros(T, dtype=torch.bool)
    for start, window in zip(starts, predictions.cpu()):
        n = min(steps, T - start)  # drop the predictions past the episode end
        predicted[start : start + n] = window[:n]
        has_prediction[start : start + n] = True
    return predicted, has_prediction


def ghost_states(
    recorded: dict, predicted: torch.Tensor, state_fields: tuple[str, ...], dataset: RWMDataset
) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
    """Scene state for the ghost env: the recorded scene state with every predicted quantity replaced.

    The world model's state vector is the concatenation of the dataset's state fields, e.g.
    states/articulation/robot/joint_position (9) | .../joint_velocity (9) | states/rigid_object/object/root_pose (7) | ...
    Each field name is also its place in the scene state dict, so we cut the vector back into fields and put each one
    where it came from. This works for any environment, as long as the state fields are recorded scene states.
    Everything the model doesn't predict (e.g. the robot base pose) stays as recorded.
    """
    # New nested dicts (the tensors are replaced, never modified, so the recorded states stay untouched)
    ghost = {kind: {name: dict(asset) for name, asset in assets.items()} for kind, assets in recorded.items()}
    offset = 0
    for field in state_fields:
        prefix, asset_type, asset_name, quantity = field.split("/")
        assert prefix == "states", f"Can only show recorded scene states, got state field {field}"
        dim = recorded[asset_type][asset_name][quantity].shape[-1]
        value = predicted[:, offset : offset + dim].clone()
        offset += dim
        # A predicted quaternion is not exactly unit length; the simulator wants unit quaternions
        quat_slice = dataset.quaternion_slices.get(field)
        if quat_slice is not None:
            value[:, quat_slice] = torch.nn.functional.normalize(value[:, quat_slice], dim=-1)
        ghost[asset_type][asset_name][quantity] = value
    return ghost


def prediction_errors(
    predicted: torch.Tensor, states: torch.Tensor, has_prediction: torch.Tensor, recorded: dict, state_fields
) -> str:
    """Root mean squared error per state field over the predicted rows, in physical units (rad, m, ...).

    Printed per episode to help pick out interesting videos; the proper metrics are the ones logged during training.
    """
    errors = (predicted - states)[has_prediction]
    parts, offset = [], 0
    for field in state_fields:
        _, asset_type, asset_name, quantity = field.split("/")
        dim = recorded[asset_type][asset_name][quantity].shape[-1]  # size of this field in the state vector
        parts.append(f"{asset_name}/{quantity} {errors[:, offset : offset + dim].square().mean().sqrt():.4f}")
        offset += dim
    return ", ".join(parts)


# --------------------------------------------------------------------------------------------------------------------
# Scene states over time
# --------------------------------------------------------------------------------------------------------------------


def stack_envs(*env_states: dict) -> dict:
    """Combine the states of several envs into one state for all of them: tensors [T, num_envs, dim], in the order
    given (RECORDED_ENV first, then GHOST_ENV), so one reset_to() call writes all envs at once."""
    first = env_states[0]
    return {
        kind: {
            name: {q: torch.stack([states[kind][name][q] for states in env_states], dim=1) for q in asset}
            for name, asset in assets.items()
        }
        for kind, assets in first.items()
    }


def state_at(states: dict, row: int) -> dict:
    """Scene state at one recorded step (row) of the episode: tensors [T, num_envs, dim] -> [num_envs, dim]."""
    return {
        kind: {name: {q: v[row] for q, v in asset.items()} for name, asset in assets.items()}
        for kind, assets in states.items()
    }


# --------------------------------------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------------------------------------


def ghost_asset_matcher(env, ghost_assets: set[str]):
    """Function that tells whether a prim path belongs to one of the predicted assets (in any env).

    Each scene asset's prim path is a regex that matches the asset in every env, e.g. the robot's is
    "/World/envs/env_[^/]+/Robot". A prim belongs to the asset if it is that prim or below it, e.g.
    "/World/envs/env_1/Robot/Geometry/panda_link0/link0_2/link0_2".
    """
    patterns = [re.compile(env.scene[name].cfg.prim_path + "(/.*)?") for name in sorted(ghost_assets)]
    return lambda path: any(pattern.fullmatch(path) for pattern in patterns)


def style_ghost_env(visualizer, env, ghost_assets: set[str]) -> None:
    """Make the predicted assets of the ghost env tinted and transparent, and hide the rest of the ghost env (e.g. its
    table): it would be drawn exactly on top of the recorded env's copy and only cause flickering. These changes only
    affect drawing, physics never sees them.

    The Newton visualizer keeps its own Newton model of the scene (built from the USD stage) with one entry per
    rendered shape. Each shape knows which env ("world") it belongs to (model.shape_world), its prim path
    (model.shape_label), color and opacity. The viewer re-reads color and opacity from the model every frame, so
    changing them once here is enough.
    """
    is_ghost_asset = ghost_asset_matcher(env, ghost_assets)
    model = visualizer.backend.model
    worlds = model.shape_world.numpy()
    opacities = model.shape_opacity.numpy()
    colors = model.shape_color.numpy()
    for shape, (label, world) in enumerate(zip(model.shape_label, worlds)):
        if world != GHOST_ENV:
            continue  # shapes of the recorded env, and shared ones like the ground plane (world -1), stay as they are
        if is_ghost_asset(label):
            colors[shape] = args_cli.ghost_color
            opacities[shape] = args_cli.ghost_opacity
        else:
            opacities[shape] = 0.0
    model.shape_opacity.assign(opacities)
    model.shape_color.assign(colors)


def render_frame(env, visualizer, env_states: dict, row: int) -> np.ndarray:
    """Show the episode at `row` in all envs and return the rendered image [height, width, 3]."""
    # Write the envs' states into the simulator. is_relative: the recorder stored positions relative to each env's
    # origin. This sets poses and joint positions directly, no physics.
    env.scene.reset_to(state_at(env_states, row), is_relative=True)
    env.sim.forward()  # compute the robot's link poses from the new joint positions
    env.sim.render()  # let the visualizer draw them
    return visualizer.render_rgb_array()


def disable_debug_markers(env_cfg) -> None:
    """Turn off the debug drawings of sensors and commands (e.g. the end-effector frame axes, the goal pose marker).

    Their poses are only updated inside env.step(), which playback never calls, so they would show stale poses that
    have nothing to do with the episode. Works for any task: every config with a debug_vis flag is switched off.
    """
    for group in (env_cfg.scene, getattr(env_cfg, "commands", None)):
        if group is None:  # e.g. a task without commands
            continue
        for term_cfg in vars(group).values():
            if hasattr(term_cfg, "debug_vis"):
                term_cfg.debug_vis = False


class Mp4Writer:
    """Writes RGB frames to an mp4 by piping the raw pixels into the ffmpeg command line tool.

    Isaac Lab's own VideoRecorder needs the moviepy package, which is not installed in this venv. ffmpeg is a
    system package (/usr/bin/ffmpeg), so this way no new Python dependency is needed.
    """

    def __init__(self, path: Path, fps: float):
        self.path, self.fps = path, fps
        self.process: subprocess.Popen | None = None
        self.frame_shape: tuple[int, ...] | None = None

    def write(self, frame: np.ndarray) -> None:
        if self.process is None:
            # Start ffmpeg with the first frame, because that's when we know the image size
            self.frame_shape = frame.shape
            height, width = frame.shape[:2]
            # fmt: off
            self.process = subprocess.Popen(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    # Input: raw RGB bytes from stdin ("-"), one frame after the other
                    "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(self.fps), "-i", "-",
                    # H.264 needs an even width and height; yuv420p is the pixel format every video player supports
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                    str(self.path),
                ],
                stdin=subprocess.PIPE,
            )
            # fmt: on
        if frame.shape != self.frame_shape:
            # ffmpeg reads fixed-size frames from the pipe, so a different size would scramble the rest of the video
            raise RuntimeError(f"Frame size changed from {self.frame_shape} to {frame.shape}. Was the window resized?")
        self.process.stdin.write(np.ascontiguousarray(frame[..., :3], dtype=np.uint8).tobytes())

    def close(self) -> None:
        if self.process is not None:
            self.process.stdin.close()  # end of input: ffmpeg finishes the file and exits
            self.process.wait()


# --------------------------------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------------------------------


def main() -> None:
    # ---- What to show: checkpoint -> model, recording, split -> episodes, task ----
    checkpoint_path = args_cli.checkpoint.resolve()
    run_dir = checkpoint_path.parent  # train_world_model.py writes checkpoints, config.yaml and split.json here
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model, stats = load_world_model(checkpoint, args_cli.device)
    H, F = model.history_horizon, model.forecast_horizon

    recording = find_recording(checkpoint)
    split = json.loads((run_dir / "split.json").read_text())
    episode_ids = choose_episodes(split, recording)

    task = args_cli.task or read_task_name(recording)
    if task is None:
        raise SystemExit(
            f"{recording.name} doesn't store its task name (recorded before collect.py saved it). "
            "Pass it with --task, e.g. --task ADLR-Lift-Cube-Franka-IK-Abs"
        )
    print(f"[INFO] checkpoint {checkpoint_path} (H={H}, F={F}), recording {recording.name}, task {task}")

    # Load the episodes with the training dataset class, so the model gets exactly the inputs it was trained on
    # (same state fields in the same order, same quaternion sign fix). H and F only matter for its window index here.
    dataset = RWMDataset(recording, H, F, included_episodes=episode_ids, preload=True)
    state_fields = dataset.state_fields
    # Which assets the model predicts, e.g. {"robot", "object"} (the asset name is the 3rd part of each field)
    ghost_assets = {field.split("/")[2] for field in state_fields}

    # ---- The environment: the recording's task, played back without physics ----
    env_cfg = parse_env_cfg(task, device=args_cli.device, num_envs=2 if args_cli.ghost else 1)
    env_cfg.scene.env_spacing = 0.0  # all envs at the same place, so the ghost is drawn on top of the recording
    env_cfg.recorders = {}  # playback must not record a new dataset
    eye, lookat = tuple(args_cli.eye or CAMERA["eye"]), tuple(args_cli.lookat or CAMERA["lookat"])
    env_cfg.sim.visualizer_cfgs = NewtonGLVisualizerCfg(
        eye=eye,
        lookat=lookat,
        focal_length=CAMERA["focal_length"],
        window_width=args_cli.width,  # also the size of the recorded frames
        window_height=args_cli.height,
        enable_picking=False,  # dragging bodies with the mouse makes no sense without physics
    )

    with launch_simulation(env_cfg, args_cli):
        # Only after the launch: inspecting the configs triggers Isaac Lab's lazy imports, which load USD (pxr), and
        # USD must not be loaded before Isaac Sim has started (it then fails with "undefined symbol" errors)
        disable_debug_markers(env_cfg)
        env = gym.make(task, cfg=env_cfg).unwrapped
        env.reset()
        env.sim.render()  # first render, so that the visualizer exists and has built its view of the scene
        visualizer = next(v for v in env.sim.visualizers if hasattr(v, "render_rgb_array"))
        # The task's own viewer settings may override the camera of the config above, so set it once more
        visualizer.set_camera_view(eye, lookat)
        if args_cli.ghost:
            style_ghost_env(visualizer, env, ghost_assets)

        fps = args_cli.fps or 1.0 / env.step_dt  # step_dt = sim dt x decimation = time between recorded rows
        name = f"{checkpoint_path.stem}_" + ("ghost" if args_cli.ghost else "recording")
        output_dir = args_cli.output_dir or run_dir / "videos" / name
        output_dir.mkdir(parents=True, exist_ok=True)
        episode_index = {f"{recording.name}:{ep.demo}": i for i, ep in enumerate(dataset._episodes)}

        with h5py.File(recording, "r") as handle:
            for episode_id in episode_ids:
                if episode_id not in episode_index:
                    print(f"[WARN] {episode_id} skipped: not in the recording or shorter than H + F = {H + F} steps")
                    continue
                demo = episode_id.split(":")[1]
                info = dataset._episodes[episode_index[episode_id]]
                in_split = next((name for name, ids in split.items() if episode_id in ids), "unknown split")
                outcome = "success" if info.success else "failure"
                path = output_dir / f"{in_split}_{demo}_{outcome}.mp4"

                recorded = read_recorded_states(handle["data"][demo])
                if args_cli.ghost:
                    # Model inputs, preprocessed like in training (_load_episode is internal to RWMDataset, but it
                    # is exactly the code path training uses, so the inputs can't silently differ)
                    states, actions = dataset._load_episode(episode_index[episode_id])
                    predicted, has_prediction = predict_episode(model, stats, states, actions)
                    env_states = stack_envs(recorded, ghost_states(recorded, predicted, state_fields, dataset))
                    errors = prediction_errors(predicted, states, has_prediction, recorded, state_fields)
                    print(f"[INFO] {in_split} {demo} ({outcome}, {info.length} steps) | prediction RMSE: {errors}")
                else:
                    env_states = stack_envs(recorded)
                    print(f"[INFO] {in_split} {demo} ({outcome}, {info.length} steps)")

                writer = Mp4Writer(path, fps)
                try:
                    for row in range(info.length):
                        writer.write(render_frame(env, visualizer, env_states, row))
                        if not visualizer.is_running():  # viewer window was closed
                            raise KeyboardInterrupt
                except KeyboardInterrupt:
                    print("[INFO] Viewer closed, stopping.")
                    break
                finally:
                    writer.close()
                print(f"[INFO] wrote {path}")

        env.close()


if __name__ == "__main__":
    main()
