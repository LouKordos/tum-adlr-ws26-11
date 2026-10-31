from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.envs.mdp.terminations import time_out
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lift_success(
    env: ManagerBasedRLEnv,
    position_threshold: float = 0.05,
) -> torch.Tensor:
    """True when the cube is within the goal tolerance at the time limit."""
    robot = env.scene["robot"]
    obj = env.scene["object"]
    command = env.command_manager.get_command("object_pose")

    # The command is relative to the robot root; compare in world coordinates.
    goal_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w.torch,
        robot.data.root_quat_w.torch,
        command[:, :3],
    )
    distance = torch.linalg.vector_norm(
        obj.data.root_pos_w.torch - goal_pos_w,
        dim=-1,
    )

    return time_out(env) & (distance < position_threshold)
