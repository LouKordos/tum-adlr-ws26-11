"""Evaluation metrics of the world model, computed on normalized windows."""

from collections import defaultdict

import torch

from adlr_rwm.models import LOSS_TERMS


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
    Curves: {"all" or state field: {"mse", "aleatoric_var", "epistemic_var": one value per forecast step}}, mean over
    windows and over all state dims or the dims of that field, in normalized units. aleatoric_var + epistemic_var is
    the predicted variance: with calibrated uncertainty, mse ≈ predicted variance.
    """
    model.eval()
    sums, sq_err, n = defaultdict(float), 0.0, 0
    window_err, aleatoric, epistemic = [], [], []
    var_sums = {"aleatoric_var": 0.0, "epistemic_var": 0.0}  # [forecast step, state dim], summed over windows
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
    curves = {
        group: {name: values[:, dims].mean(dim=1).tolist() for name, values in per_dim.items()}
        for group, dims in ({"all": slice(None)} | field_slices).items()
    }
    return metrics, curves
