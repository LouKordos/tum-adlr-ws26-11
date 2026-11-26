"""Evaluation metrics of the world model, computed on normalized windows."""

from collections import defaultdict
from contextlib import contextmanager

import torch

from adlr_rwm.models import LOSS_TERMS

# Share of a Gaussian within ±1σ and ±2σ of its mean
GAUSSIAN_COVERAGE = {"1sigma": 0.6827, "2sigma": 0.9545}
# Windows per sample_forecast call are chosen so that windows * particles stays below this (GPU memory)
MAX_PARTICLE_ROWS = 2**15


@contextmanager
def fixed_rng(device):
    """Context in which torch's random numbers start from seed 0, without changing them outside of it.

    For evaluations that sample: the same random numbers every time keep them comparable over training.
    """
    with torch.random.fork_rng(devices=[device] if "cuda" in str(device) else []):
        torch.manual_seed(0)
        yield


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
) -> tuple[dict[str, float], dict[str, dict[str, list[float]]]]:
    """Mean metrics over batches, and curves over the forecast horizon.

    Metrics: *_loss: configured loss. mse*: deterministic forecast error in normalized units.
    Curves: {"all" or state field: {"mse", "variance", "aleatoric_var", "epistemic_var": one value per forecast step}},
    mean over windows and over all state dims or the dims of that field, in normalized units. variance = aleatoric_var
    + epistemic_var is the predicted variance of a single step: forecast() feeds back the mean, so it doesn't include
    the uncertainty accumulated over the rollout (that is evaluate_sampled()).
    """
    model.eval()
    sums, sq_err, n = defaultdict(float), 0.0, 0
    window_err, aleatoric, epistemic = [], [], []
    var_sums = {"aleatoric_var": 0.0, "epistemic_var": 0.0}  # [forecast step, state dim], summed over windows
    # compute_loss samples from the predicted std, a fixed seed makes losses comparable between evaluations
    with fixed_rng(model.device):
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
                var_sums[name] = value + getattr(forecast, name).sum(dim=0)
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
    per_dim = {"mse": mse} | {name: value / n for name, value in var_sums.items()}  # each [forecast step, state dim]
    per_dim["variance"] = per_dim["aleatoric_var"] + per_dim["epistemic_var"]
    curves = {
        group: {name: values[:, dims].mean(dim=1).tolist() for name, values in per_dim.items()}
        for group, dims in ({"all": slice(None)} | field_slices).items()
    }
    return metrics, curves


@torch.no_grad()
def evaluate_sampled(
    model, state, action, normalize, num_particles: int, field_slices: dict[str, slice]
) -> tuple[dict[str, float], dict[str, dict[str, list[float]]]]:
    """Calibration of the uncertainty propagated over the rollout, from Monte Carlo rollouts (sample_forecast).

    state/action: raw windows [windows, H + F, dim]. Errors are those of the particle mean, the predicted variance is
    the variance over the particles. With calibrated uncertainty, mse ≈ variance at every forecast step (up to a factor
    1 + 1/num_particles, because the particle mean is itself an estimate).
    Coverage: share of the true states within ±1σ/±2σ of the particle mean, per state dim. Gaussian: 68.3% and 95.4%.
    Lower: overconfident, higher: underconfident (or heavy-tailed errors, where a few large errors inflate σ).

    Metrics: coverage_{1sigma,2sigma}[_<field>], mean over windows, forecast steps and dims.
    Curves: {"all" or state field: {"mse", "variance", "aleatoric_var", "epistemic_var", "coverage_1sigma",
    "coverage_2sigma": one value per forecast step}}, mean over windows and dims, in normalized units.
    """
    model.eval()
    sums = defaultdict(float)  # each [forecast step, state dim], summed over windows
    chunk = max(1, MAX_PARTICLE_ROWS // num_particles)
    with fixed_rng(model.device):
        for start in range(0, len(state), chunk):
            x_state, x_action = normalize(
                {"state": state[start : start + chunk], "action": action[start : start + chunk]}
            )
            forecast = model.sample_forecast(x_state, x_action, num_particles)
            err = (forecast.mean - x_state[:, model.history_horizon :]).square()
            sums["mse"] = sums["mse"] + err.sum(dim=0)
            for name in ("variance", "aleatoric_var", "epistemic_var"):
                sums[name] = sums[name] + getattr(forecast, "var" if name == "variance" else name).sum(dim=0)
            for name, z in (("coverage_1sigma", 1), ("coverage_2sigma", 2)):
                sums[name] = sums[name] + (err <= z**2 * forecast.var).sum(dim=0)  # |error| <= z * σ
    model.train()

    per_dim = {name: value / len(state) for name, value in sums.items()}
    groups = {"all": slice(None)} | field_slices
    curves = {
        group: {name: values[:, dims].mean(dim=1).tolist() for name, values in per_dim.items()}
        for group, dims in groups.items()
    }
    metrics = {
        f"coverage_{z}{'' if group == 'all' else f'_{group}'}": per_dim[f"coverage_{z}"][:, dims].mean().item()
        for z in GAUSSIAN_COVERAGE
        for group, dims in groups.items()
    }
    return metrics, curves
