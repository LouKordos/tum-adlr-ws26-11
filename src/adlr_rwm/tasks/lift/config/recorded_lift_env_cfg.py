from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.envs.mdp.recorders import ActionStateRecorderManagerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers.recorder_manager import DatasetExportMode
from isaaclab.utils import configclass
from isaaclab_tasks.contrib.lift.config.franka.ik_abs_env_cfg import FrankaCubeLiftEnvCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_reached_goal_position(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, asset_cfg: SceneEntityCfg
):
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_w = command[:, :3]
    curr_pos_w = asset.data.body_pos_w.torch[:, asset_cfg.body_ids][:, 0] # copied from mdp.position_command_error; indexing probably shouldn't be changed

    pos_error = torch.linalg.norm(curr_pos_w - des_pos_w, dim=1)

    return pos_error < threshold


@configclass
class RecordedLiftEnvCfg(FrankaCubeLiftEnvCfg):
    # Is reporting type error, I think because I am assigning a manager to the recorder field?
    recorders: object = ActionStateRecorderManagerCfg(
        dataset_compression=True,
        dataset_export_mode=DatasetExportMode.EXPORT_ALL,
    )

    def __post_init__(self):
        super().__post_init__()

        self.terminations.object_reached_goal = DoneTerm(
            func=object_reached_goal_position,
            params={"command_name": "object_pose", "asset_cfg": SceneEntityCfg("object"), "threshold": 0.05},
        )
