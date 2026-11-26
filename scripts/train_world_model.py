import hashlib
import json
import logging
import math
import random
import time
from collections import defaultdict
from datetime import timedelta
from itertools import islice
from pathlib import Path

import h5py
import hydra
import matplotlib

matplotlib.use("Agg")  # figures are only rendered into files for wandb, also on headless machines
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR, SequentialLR, StepLR
from torch.utils.data import Subset

from adlr_rwm.data import RWMDataset, make_dataloader
from adlr_rwm.models import SystemDynamicsEnsemble

REPO_ROOT = Path(__file__).resolve().parents[1]
LOSS_TERMS = ("state", "sequence", "bound", "kl", "extension", "contact", "termination")  # order of compute_loss
log = logging.getLogger(__name__)


def hms(seconds: float) -> timedelta:
    return timedelta(seconds=int(seconds))


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


def state_field_slices(recording: Path, state_fields) -> dict[str, slice]:
    """Position of each recorded field in the concatenated state, e.g. {"robot_joint_position": slice(0, 9)}."""
    with h5py.File(recording, "r") as handle:
        demo = next(name for name in handle["data"] if name.startswith("demo_"))
        bounds = np.cumsum([0, *(handle["data"][demo][field].shape[-1] for field in state_fields)])
    return {"_".join(f.split("/")[-2:]): slice(int(a), int(b)) for f, a, b in zip(state_fields, bounds, bounds[1:])}


@torch.no_grad()
def logstd_bounds(model, field_slices: dict[str, slice]) -> dict[str, float]:
    """Learned soft logstd bounds of the state heads per state field, mean over heads and dims (none without std)."""
    heads = [head for head in model.state_heads if head.output_std]
    if not heads:
        return {}
    bounds = {
        "min": torch.cat([head.state_min_logstd for head in heads]).cpu(),  # [heads, state_dim]
        "max": torch.cat([head.state_max_logstd for head in heads]).cpu(),
    }
    return {
        f"logstd/{kind}_{name}": values[:, dims].mean().item()
        for kind, values in bounds.items()
        for name, dims in field_slices.items()
    }


@torch.no_grad()
def evaluate(
    model, batches, normalize, loss_type: str, loss_weights, field_slices: dict[str, slice]
) -> tuple[dict[str, float], dict[str, list[float]]]:
    """Mean metrics over batches, and curves over the forecast horizon.

    Metrics: *_loss: configured loss. mse*: deterministic forecast error in normalized units.
    Curves (one value per forecast step, mean over windows and state dims, normalized units): mse, and the predicted
    variance split into aleatoric_var + epistemic_var. With calibrated uncertainty, mse ≈ total predicted variance.
    """
    model.eval()
    sums, sq_err, n = defaultdict(float), 0.0, 0
    window_err, aleatoric, epistemic = [], [], []
    var_sums = {"aleatoric_var": 0.0, "epistemic_var": 0.0}  # [forecast step], summed over windows
    # compute_loss samples from the predicted std, a fixed seed makes losses comparable between evaluations
    with torch.random.fork_rng(devices=[model.device] if "cuda" in str(model.device) else []):
        torch.manual_seed(0)
        for batch in batches:
            state, action = normalize(batch)
            terms = dict(zip(LOSS_TERMS, model.compute_loss(state, action, None, None, None, loss_type=loss_type)))
            sums["loss"] += sum(weight * terms[name] for name, weight in loss_weights.items()) * len(state)
            for name in loss_weights:
                sums[f"{name}_loss"] += terms[name] * len(state)
            forecast = model.forecast(state, action)
            err = (forecast.mean - state[:, model.history_horizon :]).square()  # [batch, forecast step, state dim]
            sq_err = sq_err + err.sum(dim=0)
            window_err.append(err.mean(dim=(1, 2)))
            aleatoric.append(forecast.aleatoric.mean(dim=1))
            epistemic.append(forecast.epistemic.mean(dim=1))
            for name, value in var_sums.items():
                var_sums[name] = value + getattr(forecast, name).mean(dim=2).sum(dim=0)
            n += len(state)
    model.train()

    mse = sq_err / n  # [forecast step, state dim]
    metrics = {name: float(value) / n for name, value in sums.items()}
    metrics["mse"] = mse.mean().item()  # headline number: all steps and dims, comparable across loss types
    metrics |= {f"mse_step_{k + 1}": value.item() for k, value in enumerate(mse.mean(dim=1))}
    metrics |= {f"mse_{name}": mse[:, dims].mean().item() for name, dims in field_slices.items()}
    # How much the error grows from the first to the last forecast step (compounding errors)
    metrics["mse_growth"] = metrics[f"mse_step_{model.forecast_horizon}"] / metrics["mse_step_1"]
    window_err = torch.cat(window_err)
    for name, values in (("aleatoric", torch.cat(aleatoric)), ("epistemic", torch.cat(epistemic))):
        metrics[name] = values.mean().item()
        # Does the model know when it is wrong? Correlation of per-window uncertainty with per-window error
        metrics[f"{name}_error_corr"] = torch.corrcoef(torch.stack([values, window_err]))[0, 1].item()
    curves = {"mse": mse.mean(dim=1).tolist()} | {name: (value / n).tolist() for name, value in var_sums.items()}
    return metrics, curves


def pick_rollout_windows(batches, field_slices: dict[str, slice], num_windows: int, history_horizon: int):
    """Example windows for the rollout plots: those where the object moves most during the forecast.

    Returns raw (unnormalized) state and action windows. Neighbouring windows overlap (same episode, shifted by one
    step), so picked windows are at least a sequence length apart in the loader order.
    """
    state, action = (torch.cat(values) for values in zip(*((batch["state"], batch["action"]) for batch in batches)))
    position = state[:, history_horizon - 1 :, field_slices["object_root_pose"]][..., :3]  # last history step onwards
    motion = (position - position[:, :1]).norm(dim=-1).amax(dim=1)  # largest distance from the start of the forecast
    picked = []
    for index in motion.argsort(descending=True).tolist():
        if all(abs(index - other) > state.shape[1] for other in picked):
            picked.append(index)
        if len(picked) == num_windows:
            break
    return state[picked], action[picked]


@torch.no_grad()
def rollout_figure(model, state, action, normalize, stats, field_slices: dict[str, slice], dims):
    """Ground truth vs forecast with ±1σ/±2σ bands of the total predictive std, one row per window, one column per dim.

    state/action: raw windows [windows, H + F, dim]. dims: (state field, index within the field, axis label).
    Plotted in physical units.
    """
    model.eval()
    forecast = model.forecast(*normalize({"state": state, "action": action}))
    model.train()
    state_mean, state_std = stats["state"]
    mean = (forecast.mean * state_std + state_mean).cpu().numpy()
    std = ((forecast.aleatoric_var + forecast.epistemic_var).sqrt() * state_std).cpu().numpy()
    state = state.cpu().numpy()

    H, F = model.history_horizon, model.forecast_horizon
    t_history, t_forecast = np.arange(-H + 1, 1), np.arange(0, F + 1)  # step 0: last history step
    fig, axes = plt.subplots(len(state), len(dims), figsize=(4 * len(dims), 2.4 * len(state)), squeeze=False)
    for row in range(len(state)):
        for col, (field, index, label) in enumerate(dims):
            d = field_slices[field].start + index
            ax = axes[row, col]
            last = state[row, H - 1, d]  # the forecast starts from the last history state, with zero uncertainty
            mu, sigma = np.r_[last, mean[row, :, d]], np.r_[0.0, std[row, :, d]]
            ax.fill_between(t_forecast, mu - 2 * sigma, mu + 2 * sigma, color="tab:blue", alpha=0.15, lw=0, label="±2σ")
            ax.fill_between(t_forecast, mu - sigma, mu + sigma, color="tab:blue", alpha=0.3, lw=0, label="±1σ")
            ax.plot(t_history, state[row, :H, d], color="0.6", lw=1.2, label="history")
            ax.plot(t_forecast, state[row, H - 1 :, d], color="black", lw=1.2, label="ground truth")
            ax.plot(t_forecast, mu, color="tab:blue", lw=1.2, label="prediction")
            ax.axvline(0, color="0.8", lw=0.8, zorder=0)
            ax.tick_params(labelsize=7)
            if row == 0:
                ax.set_title(label, fontsize=9)
            if row == len(state) - 1:
                ax.set_xlabel("step (0 = last history step)", fontsize=8)
    axes[0, 0].legend(fontsize=6, loc="best")
    fig.tight_layout()
    return fig


def wandb_keys(split: str, metrics: dict[str, float]) -> dict[str, float]:
    """Prefix evaluation metrics with the split. The per-step MSEs get their own section, e.g. val_per_step/mse_step_1."""
    return {
        (f"{split}_per_step/{key}" if key.startswith("mse_step_") else f"{split}/{key}"): value
        for key, value in metrics.items()
    }


@hydra.main(config_path="../src/adlr_rwm/configs", config_name="config")
def main(cfg: DictConfig) -> None:
    wm = cfg.world_model
    tr = wm.training
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device(wm.device)

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
    for name, dataset in datasets.items():
        log.info("%-5s: %6d episodes, %8d windows", name, len(split[name]), len(dataset))

    # Values that only exist at runtime are added to the config before it is saved or sent to wandb
    with open_dict(cfg):
        # state_dim/action_dim are not in the YAML because they depend on the recorded fields
        wm.state_dim = train_set.state_dim
        wm.action_dim = train_set.action_dim
        tr.split.recording = recording.name
        tr.split.num_episodes = {name: len(ids) for name, ids in split.items()}
        tr.split.num_windows = {name: len(dataset) for name, dataset in datasets.items()}
        tr.split.fingerprint = hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest()[:12]
    model = SystemDynamicsEnsemble(wm)
    with open_dict(cfg):
        wm.num_parameters = sum(p.numel() for p in model.parameters())

    # Hydra writes .hydra/config.yaml before main() runs, so that copy lacks the inferred values.
    # This one is the complete config of the run.
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    OmegaConf.save(cfg, run_dir / "config.yaml", resolve=True)
    split_file = run_dir / "split.json"
    split_file.write_text(json.dumps(split, indent=1) + "\n")

    wb = tr.logging.wandb
    run = wandb.init(
        project=wb.project,
        entity=wb.entity,
        name=wb.run_name,
        group=wb.group,
        tags=list(wb.tags),
        dir=run_dir,
        config=OmegaConf.to_container(cfg, resolve=True),
        # None: online, unless the WANDB_MODE env var says otherwise (e.g. offline on the cluster)
        mode=None if wb.enabled else "disabled",
    )
    # Versioned file in wandb: runs with the same split share one artifact version (wandb dedupes by content)
    split_artifact = wandb.Artifact(
        f"split-{recording.stem}", type="split", metadata={"fingerprint": tr.split.fingerprint}
    )
    split_artifact.add_file(str(split_file))
    run.log_artifact(split_artifact)
    if tr.logging.watch_interval:
        # Gradients only: parameter histograms need model(...) calls, but training only calls model.compute_loss
        run.watch(model, log="gradients", log_freq=tr.logging.watch_interval)

    # Normalization statistics from the train split only, so nothing leaks from val/test
    stats = {
        key: (mean.to(device), std.clamp_min(tr.normalization.min_std).to(device))
        for key, (mean, std) in train_set.compute_statistics().items()
    }

    def normalize(batch):
        state, action = ((batch[key].to(device, non_blocking=True) - stats[key][0]) / stats[key][1] for key in stats)
        return state, action

    # Evaluations during training use a fixed random subset of the val windows: cheap enough to run often, and the same
    # windows every time so evaluations stay comparable. Sorted, so episodes are still read in order.
    eval_sets = dict(datasets)
    num_eval_windows = tr.logging.eval_windows
    if num_eval_windows and num_eval_windows < len(datasets["val"]):
        generator = torch.Generator().manual_seed(tr.split.seed)
        indices = torch.randperm(len(datasets["val"]), generator=generator)[:num_eval_windows].sort().values
        eval_sets["val"] = Subset(datasets["val"], indices.tolist())
        log.info("val evaluations use %d of %d windows", num_eval_windows, len(datasets["val"]))
    # Only the train loader shuffles; evaluation never drops or reorders data
    loaders = {
        name: make_dataloader(
            dataset, **{**tr.dataloader, **({} if name == "train" else {"shuffle": False, "drop_last": False})}
        )
        for name, dataset in eval_sets.items()
    }
    field_slices = state_field_slices(recording, train_set.state_fields)
    # The same example windows are plotted at every evaluation, picked from the first val batches
    plots = tr.logging.rollout_plots
    rollout_windows = (
        pick_rollout_windows(islice(loaders["val"], 4), field_slices, plots.num_windows, wm.history_horizon)
        if plots.num_windows
        else None
    )

    optimizers = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}
    optimizer = optimizers[tr.optimizer.type](
        model.parameters(), lr=tr.optimizer.lr, weight_decay=tr.optimizer.weight_decay, betas=tuple(tr.optimizer.betas)
    )
    sched = tr.lr_scheduler  # stepped once per epoch
    if sched.type == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer, T_max=max(1, tr.num_epochs - sched.warmup_epochs), eta_min=sched.min_lr
        )
    elif sched.type == "step":
        scheduler = StepLR(optimizer, step_size=sched.step_size, gamma=sched.gamma)
    else:
        assert sched.type == "none", sched.type
        scheduler = ConstantLR(optimizer, factor=1.0)
    if sched.warmup_epochs:  # linear warmup from lr / (warmup + 1) to lr, then the schedule above
        warmup = LinearLR(optimizer, start_factor=1 / (sched.warmup_epochs + 1), total_iters=sched.warmup_epochs)
        scheduler = SequentialLR(optimizer, [warmup, scheduler], milestones=[sched.warmup_epochs])
    max_grad_norm = tr.max_grad_norm or math.inf
    # Auxiliary terms (extension/contact/termination) are always 0 while their head is disabled (dim 0): left out
    loss_weights = {name: weight for name, weight in tr.loss.weights.items() if getattr(model, f"{name}_dim", 1) > 0}
    steps_per_epoch = min(len(loaders["train"]), tr.limit_batches or math.inf)
    step, epoch, best, start = 0, 0, math.inf, time.perf_counter()
    log.info(f"{run.name}: {wm.num_parameters / 1e6:.2f}M parameters, {steps_per_epoch} steps/epoch, run dir {run_dir}")

    def save_checkpoint(name: str, **extra) -> None:
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}
        # the model only understands normalized data
        normalization = {key: (mean.cpu(), std.cpu()) for key, (mean, std) in stats.items()}
        config = OmegaConf.to_container(cfg, resolve=True)
        torch.save(
            {**state, "normalization": normalization, "config": config, "epoch": epoch, "step": step, **extra},
            run_dir / name,
        )

    last_eval_step = None
    mse_curves = []  # (training step, val MSE per forecast step) of every evaluation, for the horizon chart

    def validate() -> float:
        """Evaluate on the val windows, log the metrics and keep the best checkpoint. Returns the time it took."""
        nonlocal best, last_eval_step
        eval_start = time.perf_counter()
        val_batches = islice(loaders["val"], tr.limit_batches)
        val, curves = evaluate(model, val_batches, normalize, tr.loss.type, loss_weights, field_slices)
        # Charts over the forecast horizon. wandb shows the latest version of each, so the MSE chart is logged with
        # the curves of all evaluations so far.
        mse_curves.append((step, curves["mse"]))
        horizon = list(range(1, wm.forecast_horizon + 1))
        predicted_var = [a + e for a, e in zip(curves["aleatoric_var"], curves["epistemic_var"], strict=True)]
        logged = wandb_keys("val", val) | {
            "epoch": epoch,
            "val_horizon/mse_per_eval": wandb.plot.line_series(
                horizon,
                [curve for _, curve in mse_curves],
                keys=[f"step {eval_step}" for eval_step, _ in mse_curves],
                title="Val MSE over the forecast horizon, one line per evaluation",
                xname="forecast step",
            ),
            "val_horizon/mse_vs_variance": wandb.plot.line_series(
                horizon,
                [curves["mse"], predicted_var, curves["aleatoric_var"], curves["epistemic_var"]],
                keys=["MSE", "predicted variance", "aleatoric variance", "epistemic variance"],
                title=f"Val MSE vs predicted variance over the forecast horizon (step {step})",
                xname="forecast step",
            ),
        }
        if rollout_windows is not None:
            fig = rollout_figure(model, *rollout_windows, normalize, stats, field_slices, plots.dims)
            caption = f"step {step}. σ: predictive std of the deterministic forecast (single-step, not accumulated)"
            logged["val_rollouts/examples"] = wandb.Image(fig, caption=caption)
            plt.close(fig)
        run.log(logged, step=step)
        is_best = val[tr.checkpoint.best_metric] < best
        if is_best:
            best = val[tr.checkpoint.best_metric]
            run.summary.update(
                {"best/epoch": epoch, "best/step": step, **{f"best/val_{key}": value for key, value in val.items()}}
            )
            if tr.checkpoint.keep_best:
                save_checkpoint("best.pt", val=val)
        steps = " ".join(f"{val[f'mse_step_{k + 1}']:.3g}" for k in range(wm.forecast_horizon))
        fields = " ".join(f"{name} {val[f'mse_{name}']:.3g}" for name in field_slices)
        log.info(
            f"{tr.loss.type} seed {cfg.seed} | epoch {epoch} step {step} val | loss {val['loss']:.4g} | mse {val['mse']:.4g} (per step: {steps}) | {fields} | "
            f"epistemic-error corr {val['epistemic_error_corr']:.2f} | best {tr.checkpoint.best_metric} "
            f"{best:.4g}{' (new)' if is_best else ''} | eval {hms(time.perf_counter() - eval_start)}, "
            f"total {hms(time.perf_counter() - start)}"
        )
        last_eval_step = step
        return time.perf_counter() - eval_start

    window, window_steps, data_time, window_start = defaultdict(float), 0, 0.0, time.perf_counter()
    for epoch in range(1, tr.num_epochs + 1):
        model.train()
        batch_end = time.perf_counter()
        for i, batch in enumerate(islice(loaders["train"], tr.limit_batches), start=1):
            data_time += time.perf_counter() - batch_end  # time spent waiting for the DataLoader
            state, action = normalize(batch)
            # Autoregressive rollout over the forecast horizon for every ensemble head happens inside compute_loss
            out = model.compute_loss(state, action, None, None, None, bootstrap=tr.bootstrap, loss_type=tr.loss.type)
            terms = dict(zip(LOSS_TERMS, out, strict=True))
            loss = sum(weight * terms[name] for name, weight in loss_weights.items())
            optimizer.zero_grad(set_to_none=True)  # gradients accumulate otherwise
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # norm before clipping
            optimizer.step()
            step += 1

            clipped = (grad_norm > max_grad_norm).float()  # averaged over the log window = share of clipped steps
            logged = {"loss": loss, "grad_norm": grad_norm, "clip_fraction": clipped}
            logged |= {f"{name}_loss": terms[name] for name in loss_weights}
            for name, value in logged.items():
                window[name] += value.detach()  # stays on the GPU: no CPU/GPU sync per step
            window_steps += 1
            if step % tr.logging.log_interval == 0:
                elapsed = time.perf_counter() - window_start
                metrics = {name: float(value) / window_steps for name, value in window.items()}
                if not math.isfinite(metrics["loss"]):
                    raise FloatingPointError(f"Training loss is {metrics['loss']} at step {step}")
                rate = window_steps / elapsed
                metrics |= {
                    "lr": optimizer.param_groups[0]["lr"],
                    "steps_per_s": rate,
                    "samples_per_s": rate * len(state),
                    "data_wait_fraction": data_time / elapsed,
                }
                train_metrics = {f"train/{key}": value for key, value in metrics.items()}
                run.log(train_metrics | logstd_bounds(model, field_slices) | {"epoch": epoch}, step=step)
                remaining = (tr.num_epochs - epoch) * steps_per_epoch + steps_per_epoch - i
                terms = " ".join(f"{k[:-5]} {v:.4g}" for k, v in metrics.items() if k.endswith("_loss"))
                log.info(
                    f"{tr.loss.type} seed {cfg.seed} | epoch {epoch}/{tr.num_epochs} [{i}/{steps_per_epoch}] step {step} | loss {metrics['loss']:.4g} "
                    f"({terms}) | grad {metrics['grad_norm']:.3g} ({metrics['clip_fraction']:.0%} clipped) | lr {metrics['lr']:.2e} | {rate:.1f} it/s, "
                    f"data wait {metrics['data_wait_fraction']:.0%} | eta epoch {hms((steps_per_epoch - i) / rate)}, "
                    f"total {hms(remaining / rate)}"
                )
                window, window_steps, data_time, window_start = defaultdict(float), 0, 0.0, time.perf_counter()
            if tr.logging.eval_interval and step % tr.logging.eval_interval == 0:
                window_start += validate()  # evaluation doesn't count towards the training speed
            batch_end = time.perf_counter()
        scheduler.step()
        pause_start = time.perf_counter()  # evaluation and checkpointing don't count towards the training speed

        if epoch == tr.num_epochs and last_eval_step != step:  # the final model is always evaluated
            validate()
        save_checkpoint("last.pt")
        if epoch % tr.checkpoint.interval == 0:
            save_checkpoint(f"epoch_{epoch:04d}.pt")
        window_start += time.perf_counter() - pause_start

    # Test once, with the selected checkpoint. Never use it for decisions.
    checkpoint = run_dir / ("best.pt" if (run_dir / "best.pt").exists() else "last.pt")
    model.load_state_dict(torch.load(checkpoint, map_location=device)["model"])
    test, _ = evaluate(
        model, islice(loaders["test"], tr.limit_batches), normalize, tr.loss.type, loss_weights, field_slices
    )
    run.log(wandb_keys("test", test), step=step)
    run.summary.update(wandb_keys("test", test))
    log.info(
        f"{tr.loss.type} seed {cfg.seed} | test ({checkpoint.name}) | loss {test['loss']:.4g} | mse {test['mse']:.4g}"
    )
    model_artifact = wandb.Artifact(f"world-model-{run.id}", type="model", metadata={"test": test})
    model_artifact.add_file(str(checkpoint))
    run.log_artifact(model_artifact)

    run.finish()


if __name__ == "__main__":
    main()
