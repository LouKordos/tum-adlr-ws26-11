"""Figures and wandb charts of world model forecasts."""

import numpy as np
import torch
import wandb
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


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
    Plotted in physical units. A plain Figure (not pyplot), so nothing has to be closed and no backend is needed.
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
    return fig


def figure_to_array(fig: Figure) -> np.ndarray:
    """RGBA pixels of a figure, for wandb.Image: wandb 0.31 fails on Figure objects unless pyplot was imported."""
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())


def mse_and_variance_over_forecast_steps(
    mse_curves: list[tuple[int, list[float]]], curves: dict[str, list[float]], step: int
) -> dict:
    """wandb charts over the forecast horizon, from the curves of evaluate().

    mse_per_eval: the MSE of every evaluation so far, one line each (mse_curves: (training step, MSE per forecast
    step)). wandb shows the latest version of a chart, so it has to contain all evaluations.
    mse_vs_variance: MSE and predicted variance of the latest evaluation. They match if the uncertainty is calibrated.
    """
    horizon = list(range(1, len(curves["mse"]) + 1))
    predicted_var = [a + e for a, e in zip(curves["aleatoric_var"], curves["epistemic_var"], strict=True)]
    return {
        "mse_per_eval": wandb.plot.line_series(
            horizon,
            [curve for _, curve in mse_curves],
            keys=[f"step {eval_step}" for eval_step, _ in mse_curves],
            title="Val MSE over the forecast horizon, one line per evaluation",
            xname="forecast step",
        ),
        "mse_vs_variance": wandb.plot.line_series(
            horizon,
            [curves["mse"], predicted_var, curves["aleatoric_var"], curves["epistemic_var"]],
            keys=["MSE", "predicted variance", "aleatoric variance", "epistemic variance"],
            title=f"Val MSE vs predicted variance over the forecast horizon (step {step})",
            xname="forecast step",
        ),
    }
