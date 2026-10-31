from isaaclab.envs.mdp.recorders import ActionStateRecorderManagerCfg
from isaaclab.managers import TerminationTermCfg
from isaaclab.managers.recorder_manager import DatasetExportMode
from isaaclab.utils import configclass
from isaaclab_tasks.contrib.lift import lift_env_cfg
from isaaclab_tasks.contrib.lift.config.franka.ik_abs_env_cfg import FrankaCubeLiftEnvCfg

from adlr_rwm.tasks.lift.mdp.success import lift_success


@configclass
class RecordedLiftTerminationsCfg(lift_env_cfg.TerminationsCfg):
    # Needed so that RecorderManager automatically labels successful and failed trajectories
    # We call it success because that is how it's parsed by the recorder manager, and set time_out = True
    # because that makes it set truncated instead of terminated. This makes sense because it is only True on the last
    # timestep, so that the lifted cube actually has to *stay* lifted.
    success = TerminationTermCfg(func=lift_success, time_out=True)


@configclass
class RecordedLiftEnvCfg(FrankaCubeLiftEnvCfg):
    # Is reporting type error, I think because I am assigning a manager to the recorder field?
    recorders: object = ActionStateRecorderManagerCfg(
        dataset_compression=True,
        dataset_export_mode=DatasetExportMode.EXPORT_ALL,
    )

    terminations: object = RecordedLiftTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
