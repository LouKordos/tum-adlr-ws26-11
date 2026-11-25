"""Play back recorded episodes in Isaac Lab and write one video per episode.

What the video shows
--------------------
* The recorded episode (ground truth), exactly as it was recorded.

How it works
------------
1. The checkpoint (a .pt file written by scripts/train_world_model.py) contains the model weights, the normalization
   statistics and the full training config. Its run directory also contains split.json (the train/val/test episode
   lists). The config names the recording (.hdf5) the model was trained on, and the recording names the Isaac Lab task
   it was recorded in. That is why --checkpoint is the only required argument.
2. We create the task's environment. Nothing is simulated: every frame we write the recorded state directly into the
   simulator (like replaying motion capture) and render one image.
3. The Newton GL visualizer (OpenGL, so it works without an RTX GPU) renders the frames, and ffmpeg turns them into
   one mp4 per episode.

Examples
--------
    # 4 random test episodes
    uv run python scripts/visualize_world_model.py --checkpoint outputs/<date>/<time>/best.pt

    # Two specific episodes
    uv run python scripts/visualize_world_model.py --checkpoint outputs/<date>/<time>/best.pt --episodes demo_12 demo_345

    # Watch it live in the viewer window while the videos are written
    uv run python scripts/visualize_world_model.py --checkpoint outputs/<date>/<time>/best.pt --window

Videos are written to <run dir>/videos/<checkpoint>_recording/ (inside outputs/, which git ignores).
Recordings made before collect.py stored the task name need --task, e.g. --task ADLR-Lift-Cube-Franka-IK-Abs.

This script is deliberately self-contained: its settings are the command line flags below, not the Hydra configs in
src/adlr_rwm/configs, and it only reads the training code (dataset + model), it doesn't change it.
"""

import argparse
from pathlib import Path

# Isaac Lab's launcher has to be imported before most other Isaac Lab modules (same pattern as collect.py).
from isaaclab.app import add_launcher_args, launch_simulation

REPO_ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description="Play back recorded episodes in Isaac Lab and write videos.")
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
import subprocess

import gymnasium as gym
import h5py
import isaaclab_tasks  # noqa: F401  (registers Isaac Lab's tasks with gym)
import numpy as np
import torch
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

import adlr_rwm.tasks  # noqa: F401  (registers our tasks, e.g. ADLR-Lift-Cube-Franka-IK-Abs, with gym)

# Camera: eye and lookat in [m], focal length of the lens in [mm] (Isaac Lab's default of 12 mm is very wide-angle and
# makes the robot tiny). The robot base is at the env origin and the robot faces +x; the table top is at z = 0 and the
# cube starts around x = 0.5 on it.
CAMERA = {"eye": (1.9, -1.5, 1.0), "lookat": (0.35, 0.0, 0.35), "focal_length": 24.0}


# --------------------------------------------------------------------------------------------------------------------
# Loading the checkpoint, the recording and the episodes
# --------------------------------------------------------------------------------------------------------------------


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
# Scene states over time
# --------------------------------------------------------------------------------------------------------------------


def stack_envs(*env_states: dict) -> dict:
    """Combine the states of several envs into one state for all of them: tensors [T, num_envs, dim], in the order
    given (env 0 first), so one reset_to() call writes all envs at once."""
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
    # ---- What to show: checkpoint -> recording, split -> episodes, task ----
    checkpoint_path = args_cli.checkpoint.resolve()
    run_dir = checkpoint_path.parent  # train_world_model.py writes checkpoints, config.yaml and split.json here
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    recording = find_recording(checkpoint)
    split = json.loads((run_dir / "split.json").read_text())
    episode_ids = choose_episodes(split, recording)

    task = args_cli.task or read_task_name(recording)
    if task is None:
        raise SystemExit(
            f"{recording.name} doesn't store its task name (recorded before collect.py saved it). "
            "Pass it with --task, e.g. --task ADLR-Lift-Cube-Franka-IK-Abs"
        )
    print(f"[INFO] checkpoint {checkpoint_path}, recording {recording.name}, task {task}")

    # ---- The environment: the recording's task, played back without physics ----
    env_cfg = parse_env_cfg(task, device=args_cli.device, num_envs=1)
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

        fps = args_cli.fps or 1.0 / env.step_dt  # step_dt = sim dt x decimation = time between recorded rows
        output_dir = args_cli.output_dir or run_dir / "videos" / f"{checkpoint_path.stem}_recording"
        output_dir.mkdir(parents=True, exist_ok=True)

        with h5py.File(recording, "r") as handle:
            for episode_id in episode_ids:
                demo = episode_id.split(":")[1]
                if demo not in handle["data"]:
                    print(f"[WARN] {episode_id} skipped: not in the recording")
                    continue
                episode = handle["data"][demo]
                length = int(episode.attrs["num_samples"])
                in_split = next((name for name, ids in split.items() if episode_id in ids), "unknown split")
                outcome = "success" if episode.attrs.get("success", False) else "failure"
                path = output_dir / f"{in_split}_{demo}_{outcome}.mp4"

                env_states = stack_envs(read_recorded_states(episode))
                print(f"[INFO] {in_split} {demo} ({outcome}, {length} steps)")

                writer = Mp4Writer(path, fps)
                try:
                    for row in range(length):
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
