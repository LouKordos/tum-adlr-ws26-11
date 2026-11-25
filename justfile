# Run with `uv run just <recipe>` (just is a dev dependency), list recipes with `uv run just`.
# Extra arguments are passed to the script unchanged, e.g. `uv run just train world_model.training.num_epochs=20`.

set positional-arguments

# `train` runs every combination of these, e.g. `uv run just seeds=0 losses=gaussian_nll train` for a single run
seeds := "0,1,2"
losses := "gaussian_nll,mse"
# How many runs train at the same time, e.g. `uv run just parallel=3 train` (each needs its own GPU memory and RAM)
parallel := "1"
# wandb group per loss is <sweep>_<loss> (wandb shows mean/std over the seeds), outputs go to outputs/<sweep>/
sweep := "sweep_" + datetime("%Y-%m-%d_%H-%M-%S")

default:
    @just --list

# Record 20k state machine episodes from 512 parallel envs into data/recordings
collect *args:
    uv run python scripts/environments/state_machine/collect.py --num_envs 512 --num_episodes 20000 --disable_fabric --viz none "$@"

# Train every seed with every loss type, `parallel` runs at a time
train *overrides:
    #!/usr/bin/env bash
    # One line "<loss> <seed>" per run. xargs starts `parallel` runs at a time and exits with 123 if any run failed.
    for loss in {{replace(losses, ",", " ")}}; do for seed in {{replace(seeds, ",", " ")}}; do echo "$loss $seed"; done; done |
        xargs -P {{parallel}} -I {} bash -c 'read -r loss seed <<< "$1"; shift; exec uv run python scripts/train_world_model.py \
            seed=$seed world_model.training.loss.type=$loss hydra.run.dir=outputs/{{sweep}}/${loss}_seed_$seed \
            world_model.training.logging.wandb.group={{sweep}}_$loss "$@"' run {} "$@"
