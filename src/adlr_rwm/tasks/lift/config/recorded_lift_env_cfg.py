from isaaclab.envs.mdp.recorders import ActionStateRecorderManagerCfg
from isaaclab.managers import TerminationTermCfg, RewardTermCfg, CurriculumTermCfg
from isaaclab.managers.recorder_manager import DatasetExportMode
from isaaclab.utils import configclass
from isaaclab_tasks.contrib.lift import lift_env_cfg
from isaaclab_tasks.contrib.lift.config.franka.ik_abs_env_cfg import FrankaCubeLiftEnvCfg
from isaaclab_tasks.contrib.lift.mdp import modify_reward_weight

from adlr_rwm.tasks.lift.mdp.success import lift_success
from adlr_rwm.tasks.lift.mdp.rewards import approach_velocity_penalty


@configclass
class RecordedLiftTerminationsCfg(lift_env_cfg.TerminationsCfg):
    # Needed so that RecorderManager automatically labels successful and failed trajectories
    # We call it success because that is how it's parsed by the recorder manager, and set time_out = True
    # because that makes it set truncated instead of terminated. This makes sense because it is only True on the last
    # timestep, so that the lifted cube actually has to *stay* lifted.
    success = TerminationTermCfg(func=lift_success, time_out=True)


@configclass
class RecordedLiftRewardsConfig(lift_env_cfg.RewardsCfg):
    # The function returns a non-negative squared relative velocity, so the reward
    # weight must be negative for this to act as a penalty.
    velocity_penalty = RewardTermCfg(func=approach_velocity_penalty, weight=-0.1)


@configclass
class RecordedLiftCurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurriculumTermCfg(
        func=modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 15000}
    )

    joint_vel = CurriculumTermCfg(
        func=modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 15000}
    )


@configclass
class RecordedLiftEnvCfg(FrankaCubeLiftEnvCfg):
    # Is reporting type error, I think because I am assigning a manager to the recorder field?
    # recorders: object = ActionStateRecorderManagerCfg(
    #     dataset_compression=True,
    #     dataset_export_mode=DatasetExportMode.EXPORT_ALL,
    # )

    terminations: object = RecordedLiftTerminationsCfg()
    rewards: object = RecordedLiftRewardsConfig()
    curriculum: object = None

    def __post_init__(self):
        super().__post_init__()

        self.rewards.object_goal_tracking.weight = 35
        self.rewards.object_goal_tracking_fine_grained.weight = 25
        self.rewards.action_rate.weight = -1e-3
        self.rewards.joint_vel.weight = -1e-3
