"""Train the SystemDynamicsEnsemble world model on recorded HDF5 episodes.

Work in progress: the train/val/test split and the training loop are still TODO.

    python scripts/train_world_model.py
    python scripts/train_world_model.py device=cpu world_model.training.logging.wandb.enabled=true
"""

from pathlib import Path

import hydra
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict

from adlr_rwm.data import RWMDataset
from adlr_rwm.models import SystemDynamicsEnsemble

REPO_ROOT = Path(__file__).resolve().parents[1]


@hydra.main(config_path="../src/adlr_rwm/configs", config_name="config")
def main(cfg: DictConfig) -> None:
    wm = cfg.world_model
    tr = wm.training

    data_dir = Path(tr.dataset.data_dir)
    data_dir = data_dir if data_dir.is_absolute() else REPO_ROOT / data_dir
    dataset_kwargs = {key: value for key, value in tr.dataset.items() if key != "data_dir"}
    # TODO (split): one dataset per split, each with included_episodes=<that split's "<file name>:<demo>" ids>
    train_set = RWMDataset(data_dir, wm.history_horizon, wm.forecast_horizon, **dataset_kwargs)

    # state_dim/action_dim are not in the YAML because they depend on the recorded fields. Add them to the config
    # *before* it is saved or sent to wandb, so every run records them like any other parameter.
    with open_dict(wm):  # Hydra configs are "struct": adding new keys needs open_dict
        wm.state_dim = train_set.state_dim
        wm.action_dim = train_set.action_dim

    # Hydra writes .hydra/config.yaml *before* main() runs, so that copy lacks the inferred values.
    # This one is the complete config of the run.
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    OmegaConf.save(cfg, run_dir / "config.yaml", resolve=True)

    wb = tr.logging.wandb
    run = wandb.init(
        project=wb.project,
        entity=wb.entity,
        name=wb.run_name,
        tags=list(wb.tags),
        dir=run_dir,
        config=OmegaConf.to_container(cfg, resolve=True),
        # None: online, unless the WANDB_MODE env var says otherwise (e.g. offline on the cluster)
        mode=None if wb.enabled else "disabled",
    )

    model = SystemDynamicsEnsemble(wm)
    print(f"state_dim={wm.state_dim} action_dim={wm.action_dim} windows={len(train_set)} run_dir={run_dir}")

    # TODO: dataloaders, optimizer, training loop, validation, checkpoints (save cfg in each checkpoint too)
    del model

    run.finish()


if __name__ == "__main__":
    main()
