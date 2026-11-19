from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def approach_velocity_penalty(
    env: ManagerBasedRLEnv,
    d0: float = 0.20,
    k: float = 30.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_body_name: str = "panda_hand",
) -> torch.Tensor:

    robot = env.scene[asset_cfg.name]
    obj = env.scene[object_cfg.name]
    ee_frame = env.scene["ee_frame"]

    # End-effector position and linear velocity in world frame
    ee_pos = ee_frame.data.target_pos_w[:, 0, :]
    body_ids, _ = robot.find_bodies(ee_body_name)
    ee_vel = robot.data.body_lin_vel_w[:, body_ids[0], :]

    # Object position and linear velocity
    obj_pos = obj.data.root_pos_w
    obj_vel = obj.data.root_lin_vel_w

    # Relative distance and velocity
    distance = torch.linalg.vector_norm(
        ee_pos - obj_pos, dim=-1
    )

    relative_velocity = ee_vel - obj_vel
    velocity_squared = torch.sum(
        relative_velocity ** 2, dim=-1
    )

    # Smooth proximity weighting
    proximity = torch.sigmoid(k * (d0 - distance))

    return proximity * velocity_squared
