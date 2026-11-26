"""Figures and wandb charts of world model forecasts."""

import numpy as np
import torch
import wandb
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .metrics import GAUSSIAN_COVERAGE, fixed_rng


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
def rollout_figure(
    model, state, action, normalize, stats, field_slices: dict[str, slice], dims, num_particles, num_trajectories
):
    """Ground truth vs Monte Carlo forecast (sample_forecast), one row per window, one column per dim.

    Prediction: particle mean. Bands: ±1σ/±2σ of the particles, so they include the uncertainty propagated over the
    rollout. Also num_trajectories randomly drawn particles. With a fixed seed, so the same random numbers at every
    evaluation: changes between evaluations come from the model.
    state/action: raw windows [windows, H + F, dim]. dims: (state field, index within the field, axis label).
    Plotted in physical units. A plain Figure (not pyplot), so nothing has to be closed and no backend is needed.
    """
    model.eval()
    with fixed_rng(model.device):
        forecast = model.sample_forecast(*normalize({"state": state, "action": action}), num_particles)
        drawn = torch.stack([torch.randperm(num_particles)[:num_trajectories] for _ in range(len(state))])
    model.train()
    state_mean, state_std = stats["state"]
    mean = (forecast.mean * state_std + state_mean).cpu().numpy()
    std = (forecast.var.sqrt() * state_std).cpu().numpy()
    particles = forecast.particles[torch.arange(len(state))[:, None], drawn.to(forecast.particles.device)]
    particles = (particles * state_std + state_mean).cpu().numpy()  # [windows, num_trajectories, F, state_dim]
    state = state.cpu().numpy()

    H, F = model.history_horizon, model.forecast_horizon
    t_history, t_forecast = np.arange(-H + 1, 1), np.arange(0, F + 1)  # step 0: last history step
    fig = Figure(figsize=(4 * len(dims), 2.4 * len(state)), layout="constrained")
    axes = fig.subplots(len(state), len(dims), squeeze=False)
    for row in range(len(state)):
        for col, (field, index, label) in enumerate(dims):
            d = field_slices[field].start + index
            ax = axes[row, col]
            last = state[row, H - 1, d]  # the forecast starts from the last history state, with zero uncertainty
            mu, sigma = np.r_[last, mean[row, :, d]], np.r_[0.0, std[row, :, d]]
            ax.fill_between(t_forecast, mu - 2 * sigma, mu + 2 * sigma, color="tab:blue", alpha=0.15, lw=0, label="±2σ")
            ax.fill_between(t_forecast, mu - sigma, mu + sigma, color="tab:blue", alpha=0.3, lw=0, label="±1σ")
            for j in range(num_trajectories):
                legend = f"{num_trajectories} of {num_particles} particles" if j == 0 else None
                trajectory = np.r_[last, particles[row, j, :, d]]
                ax.plot(t_forecast, trajectory, color="tab:orange", lw=0.6, alpha=0.8, label=legend)
            ax.plot(t_history, state[row, :H, d], color="0.6", lw=1.2, label="history")
            ax.plot(t_forecast, state[row, H - 1 :, d], color="black", lw=1.2, label="ground truth")
            ax.plot(t_forecast, mu, color="tab:blue", lw=1.2, label="particle mean")
            ax.axvline(0, color="0.8", lw=0.8, zorder=0)
            ax.tick_params(labelsize=7)
            if row == 0:
                ax.set_title(label, fontsize=9)
            if row == len(state) - 1:
                ax.set_xlabel("step (0 = last history step)", fontsize=8)
    axes[0, 0].legend(fontsize=6, loc="best")
    return fig


def figure_to_array(fig: Figure) -> np.ndarray:
    """RGBA pixels of a figure, for wandb.Image: wandb 0.31 fails on Figure objects unless pyplot was imported."""
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())


def mse_and_variance_over_forecast_steps(
    mse_curves: list[tuple[int, list[float]]], curves: dict[str, dict[str, list[float]]], step: int, source: str
) -> dict:
    """wandb charts over the forecast horizon. Keys go after the split, e.g. "val_".

    horizon/mse_per_eval: the MSE (mean over all dims) of every evaluation so far, one line each (mse_curves:
        (training step, MSE per forecast step)). wandb shows the latest version of a chart, so it has to contain all
        evaluations.
    horizon/mse_vs_variance: MSE and predicted variance of the latest evaluation, mean over all dims. They match if
        the uncertainty is calibrated. From the curves of evaluate_sampled() (or evaluate(), whose variance is that of
        a single step), source says which, for the titles.
    horizon_per_field/<field>: the same for the dims of each state field, whose errors differ by orders of magnitude.
    """
    horizon = list(range(1, len(mse_curves[-1][1]) + 1))

    def mse_vs_variance(curve: dict[str, list[float]], dims: str):
        return wandb.plot.line_series(
            horizon,
            [curve["mse"], curve["variance"], curve["aleatoric_var"], curve["epistemic_var"]],
            keys=["MSE", "predicted variance", "aleatoric variance", "epistemic variance"],
            title=f"Val MSE vs predicted variance over the forecast horizon, {dims} (step {step}, {source})",
            xname="forecast step",
        )

    charts = {
        "horizon/mse_per_eval": wandb.plot.line_series(
            horizon,
            [curve for _, curve in mse_curves],
            keys=[f"step {eval_step}" for eval_step, _ in mse_curves],
            title="Val MSE over the forecast horizon, one line per evaluation",
            xname="forecast step",
        ),
        "horizon/mse_vs_variance": mse_vs_variance(curves["all"], "all dims"),
    }
    charts |= {f"horizon_per_field/{group}": mse_vs_variance(c, group) for group, c in curves.items() if group != "all"}
    return charts


def coverage_over_forecast_steps(curves: dict[str, dict[str, list[float]]], step: int, source: str) -> dict:
    """wandb charts of the coverage over the forecast horizon, from the curves of evaluate_sampled().

    horizon/coverage_{1sigma,2sigma}: share of true states within ±1σ/±2σ of the predicted mean, over all dims and per
    state field, with the share for a Gaussian as reference. Keys go after the split, e.g. "val_".
    """
    horizon = list(range(1, len(curves["all"]["mse"]) + 1))
    return {
        f"horizon/coverage_{z}": wandb.plot.line_series(
            horizon,
            [curve[f"coverage_{z}"] for curve in curves.values()] + [[share] * len(horizon)],
            keys=[*(f"{group} dims" if group == "all" else group for group in curves), "Gaussian"],
            title=f"Val coverage ±{z[0]}σ over the forecast horizon (step {step}, {source})",
            xname="forecast step",
        )
        for z, share in GAUSSIAN_COVERAGE.items()
    }
