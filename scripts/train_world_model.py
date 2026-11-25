from pathlib import Path

import h5py
import hydra
import numpy as np
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict

from adlr_rwm.data import RWMDataset
from adlr_rwm.models import SystemDynamicsEnsemble

REPO_ROOT = Path(__file__).resolve().parents[1]


def make_split(recording: Path, val_fraction: float, test_fraction: float, seed: int) -> dict[str, list[str]]:
    """Randomly assign every episode of a single HDF5 recording to train/val/test."""
    with h5py.File(recording, "r") as handle:
        demos = sorted(name for name in handle["data"] if name.startswith("demo_"))
    episode_ids = [f"{recording.name}:{demo}" for demo in demos]  # same id format as RWMDataset(included_episodes)

    # Sorted ids + fixed seed: the split only depends on which episodes exist, not on the order they are read in
    shuffled = [episode_ids[i] for i in np.random.default_rng(seed).permutation(len(episode_ids))]
    n_val, n_test = round(len(shuffled) * val_fraction), round(len(shuffled) * test_fraction)
    split = {
        "train": sorted(shuffled[n_val + n_test :]),
        "val": sorted(shuffled[:n_val]),
        "test": sorted(shuffled[n_val : n_val + n_test]),
    }

    return split


@hydra.main(config_path="../src/adlr_rwm/configs", config_name="config")
def main(cfg: DictConfig) -> None:
    wm = cfg.world_model
    tr = wm.training

    data_dir = Path(tr.dataset.data_dir)
    data_dir = data_dir if data_dir.is_absolute() else REPO_ROOT / data_dir
    recordings = [data_dir] if data_dir.is_file() else sorted(data_dir.glob("*.hdf5"))
    recording = recordings[0]

    split = make_split(recording, tr.split.val_fraction, tr.split.test_fraction, tr.split.seed)
    dataset_kwargs = {key: value for key, value in tr.dataset.items() if key != "data_dir"}
    datasets = {
        name: RWMDataset(recording, wm.history_horizon, wm.forecast_horizon, included_episodes=ids, **dataset_kwargs)
        for name, ids in split.items()
    }
    train_set = datasets["train"]

    # Values that only exist at runtime are added to the config before it is saved or sent to wandb
    with open_dict(cfg):
        # state_dim/action_dim are not in the YAML because they depend on the recorded fields
        wm.state_dim = train_set.state_dim
        wm.action_dim = train_set.action_dim

    # Hydra writes .hydra/config.yaml before main() runs, so that copy lacks the inferred values.
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
    print(f"state_dim={wm.state_dim} action_dim={wm.action_dim} windows={ {n: len(d) for n, d in datasets.items()} }")
    print(f"run_dir={run_dir}")

    # TODO: dataloaders (only train shuffles), optimizer, training loop, validation, test, checkpoints
    #  (save cfg in each checkpoint too)
    del model

    run.finish()


if __name__ == "__main__":
    main()
