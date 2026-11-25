# Run with `uv run just <recipe>` (just is a dev dependency), list recipes with `uv run just`.
# Extra arguments are passed to the script unchanged, e.g. `uv run just train world_model.training.loss.type=mse`.

set positional-arguments

# Seeds for train-seeds, override with `uv run just seeds=0,1,2 train-seeds`
seeds := "0,1,2,3,4"
# wandb group of a train-seeds sweep (wandb shows mean/std per group), override with `uv run just group=nll train-seeds`
group := "seeds_" + datetime("%Y-%m-%d_%H-%M-%S")

default:
    @just --list

# Record 20k state machine episodes from 512 parallel envs into data/recordings
collect *args:
    uv run python scripts/environments/state_machine/collect.py --num_envs 512 --num_episodes 20000 --disable_fabric --viz none "$@"

# Train the world model once
train *overrides:
    uv run python scripts/train_world_model.py "$@"

# Train once per seed (Hydra multirun, one after another), all runs in one wandb group
train-seeds *overrides:
    uv run python scripts/train_world_model.py --multirun seed={{seeds}} world_model.training.logging.wandb.group={{group}} "$@"
