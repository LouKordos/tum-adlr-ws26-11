from isaaclab.envs.mdp.recorders import ActionStateRecorderManagerCfg
from isaaclab.managers.recorder_manager import DatasetExportMode
from isaaclab.utils import configclass
from isaaclab_tasks.contrib.lift.config.franka.ik_abs_env_cfg import FrankaCubeLiftEnvCfg


@configclass
class RecordedLiftEnvCfg(FrankaCubeLiftEnvCfg):
    # Is reporting type error, I think because I am assigning a manager to the recorder field?
    recorders: object = ActionStateRecorderManagerCfg(
        dataset_export_dir_path="data/recordings",
        dataset_compression=True,
        dataset_filename="franka_contrib_lift",
        dataset_export_mode=DatasetExportMode.EXPORT_ALL,
    )

    def __post_init__(self):
        super().__post_init__()
