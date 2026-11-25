# Run with `uv run just <recipe>` (just is a dev dependency), list recipes with `uv run just`.
# Extra arguments are passed to the script unchanged, e.g. `uv run just train world_model.training.loss.type=mse`.

set positional-arguments

# Seeds that `train` runs, e.g. `uv run just seeds=0 train` for a single run
seeds := "0,1,2,3,4"
# How many seeds train at the same time, e.g. `uv run just parallel=3 train` (each needs its own GPU memory and RAM)
parallel := "1"
# wandb group of the runs (wandb shows mean/std per group), also the output folder outputs/<group>/seed_<seed>
group := "seeds_" + datetime("%Y-%m-%d_%H-%M-%S")

default:
    @just --list

# Record 20k state machine episodes from 512 parallel envs into data/recordings
collect *args:
    uv run python scripts/environments/state_machine/collect.py --num_envs 512 --num_episodes 20000 --disable_fabric --viz none "$@"

# Train the world model once per seed, `parallel` seeds at a time
train *overrides:
    echo {{seeds}} | tr , '\n' | xargs -P {{parallel}} -I SEED uv run python scripts/train_world_model.py \
        seed=SEED hydra.run.dir=outputs/{{group}}/seed_SEED world_model.training.logging.wandb.group={{group}} "$@"
