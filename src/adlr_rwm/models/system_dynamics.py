# Source: rsl_rl_rwm (https://github.com/leggedrobotics/rsl_rl_rwm)
# TODO Currrent issues:
# (1) There is some leftover logic for an RSSM backbone, including KL loss terms,
# which now default to 0. Either remove this or implement an RSSM.
# (2) The "sequence" prediction mode, which is diabled now by default, predicts
# states starting from the second entry of the history horizon, not just the last state.
# [-------H-------][----F----], H+F=T
#      history       forecast
# "single"   :   p(f1 | h_1, h_2, ... h_H)
# "sequence" :   p(h_2, ...., h_H, f_1 | h_1, h_2, ... h_H )
# An additional "sequence loss" is being calculated in this case (currently defaults to 0)
# In theory this should givea denser gradient signal, but it'll increase the model size by a lot.
# We should consider if this is worth integrating (the source repo doesn't)


from typing import NamedTuple

import torch
from omegaconf import DictConfig
from torch import nn

from .architectures import MLPAuxiliaryHead, MLPBase, MLPStateHead, RNNBase

# Names of the losses returned by SystemDynamicsEnsemble.compute_loss, in that order
LOSS_TERMS = ("state", "sequence", "bound", "kl", "extension", "contact", "termination")


class Forecast(NamedTuple):
    """Output of SystemDynamicsEnsemble.forecast(), in normalized state units.

    The ensemble is treated as an equally weighted mixture of the heads' Gaussians, so the total predictive variance
    of each state dim is aleatoric_var + epistemic_var.
    """

    mean: torch.Tensor  # [B, F, state_dim] ensemble mean
    aleatoric: torch.Tensor  # [B, F] predicted std, mean over heads, summed over dims (as returned by forward())
    epistemic: torch.Tensor  # [B, F] std of the head means, summed over dims (as returned by forward())
    aleatoric_var: torch.Tensor  # [B, F, state_dim] predicted variance, mean over heads
    epistemic_var: torch.Tensor  # [B, F, state_dim] variance of the head means


class SampledForecast(NamedTuple):
    """Output of SystemDynamicsEnsemble.sample_forecast(), in normalized state units.

    The variances are over the particles of each window, so they include the uncertainty that accumulates over the
    rollout. var = aleatoric_var + epistemic_var (law of total variance over the heads' particle groups), each an
    unbiased estimate. epistemic_var subtracts the sampling noise of the heads' particle means, which would otherwise
    add about 0.8 * aleatoric_var / (K / 5): as much as the epistemic part itself at K = 30. So for single windows it
    can come out slightly negative; averages are unbiased.
    """

    particles: torch.Tensor  # [B, K, F, state_dim] sampled trajectories
    head: torch.Tensor  # [K] head of each particle, the same for every window
    mean: torch.Tensor  # [B, F, state_dim] mean over the particles
    var: torch.Tensor  # [B, F, state_dim] variance over the particles: the propagated predictive variance
    aleatoric_var: torch.Tensor  # [B, F, state_dim] variance within each head's particles, mean over heads
    epistemic_var: torch.Tensor  # [B, F, state_dim] variance between the heads' particle means


class SystemDynamicsEnsemble(nn.Module):
    def __init__(
        self,
        config: DictConfig,
    ):
        super().__init__()
        self.device = config.device

        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.extension_dim = config.extension_dim
        self.contact_dim = config.contact_dim
        self.termination_dim = config.termination_dim

        self.history_horizon = config.history_horizon
        self.forecast_horizon = config.forecast_horizon

        # main NN configurations
        self.base_cfg = config.base
        self.state_head_cfg = config.state_head
        self.auxiliary_head_cfg = config.auxiliary_head
        self.ensemble_size = config.ensemble_size  # number of state/aux heads

        self._init_networks()

    def _init_networks(self):
        self.state_base = self._create_base()
        self.state_heads = nn.ModuleList(
            [
                MLPStateHead(self.base_output_dim, self.state_dim, self.device, self.state_head_cfg).to(self.device)
                for _ in range(self.ensemble_size)
            ]
        )

        self.auxiliary_base = self._create_base()
        self.auxiliary_heads = nn.ModuleList(
            [
                MLPAuxiliaryHead(
                    self.base_output_dim,
                    self.extension_dim,
                    self.contact_dim,
                    self.termination_dim,
                    self.device,
                    self.auxiliary_head_cfg,
                ).to(self.device)
                for _ in range(self.ensemble_size)
            ]
        )

    def _create_base(self):
        if self.base_cfg.type == "mlp":
            input_dim = self.history_horizon * (self.state_dim + self.action_dim)
            self.base_output_dim = self.base_cfg.network.output_dim
            self.prediction_type = "single"
            return MLPBase(
                input_dim=input_dim,
                output_dim=self.base_output_dim,
                device=self.device,
                architecture_config=self.base_cfg.network,
            )
        elif self.base_cfg.type == "rnn":
            input_dim = self.state_dim + self.action_dim
            self.base_output_dim = self.base_cfg.hidden_size
            self.prediction_type = "single"
            return RNNBase(input_dim=input_dim, device=self.device, architecture_config=self.base_cfg)
        else:
            raise ValueError("Invalid architecture type.")

    def forward(self, x_state_batch: torch.Tensor, x_action_batch: torch.Tensor, model_ids=None):

        # shape checks and safety measures
        x_state_batch = x_state_batch.to(self.device)
        x_action_batch = x_action_batch.to(self.device)

        assert x_state_batch.shape[:2] == x_action_batch.shape[:2], (
            f"Batch size and sequence length must match between x_state_batch {x_state_batch.shape[:2]} "
            f"and x_action_batch {x_action_batch.shape[:2]}."
        )

        if self.base_cfg.type == "mlp":
            assert x_state_batch.shape[1] == self.history_horizon, (
                f"MLP requires full history_horizon ({self.history_horizon}), got T={x_state_batch.shape[1]}."
            )
        elif self.base_cfg.type == "rnn":
            if self.state_base.memory.hidden_states is None:
                # Cold start: requires full history window to warm up memory
                assert x_state_batch.shape[1] == self.history_horizon, (
                    f"RNN cold start requires history_horizon ({self.history_horizon}), got T={x_state_batch.shape[1]}."
                )
            else:
                # Active hidden state: expect single-step autoregressive update
                assert x_state_batch.shape[1] == 1, (
                    f"RNN with active hidden_states expects single-step input (T=1), "
                    f"got T={x_state_batch.shape[1]}. Call model.reset() first if starting a new sequence."
                )

        extensions, contacts, terminations = [], [], []
        state_means, state_stds = self._state_head_outputs(x_state_batch, x_action_batch)

        auxiliary_base_output = self.auxiliary_base(x_state_batch, x_action_batch)
        for head in self.auxiliary_heads:
            extension, contact, termination = head(auxiliary_base_output, x_state_batch)
            if self.prediction_type == "sequence":
                extension = extension[:, -1] if extension is not None else None
                contact = contact[:, -1] if contact is not None else None
                termination = termination[:, -1] if termination is not None else None
            extensions.append(extension.unsqueeze(0) if extension is not None else None)
            contacts.append(contact.unsqueeze(0) if contact is not None else None)
            terminations.append(termination.unsqueeze(0) if termination is not None else None)

        extensions = torch.cat(extensions, dim=0) if self.extension_dim > 0 else None
        contacts = torch.cat(contacts, dim=0) if self.contact_dim > 0 else None
        terminations = torch.cat(terminations, dim=0) if self.termination_dim > 0 else None

        if model_ids is None:
            output_state_means = state_means.mean(dim=0)
            output_extensions = extensions.mean(dim=0) if extensions is not None else None
            output_contacts = contacts.mean(dim=0) if contacts is not None else None
            output_terminations = terminations.mean(dim=0) if terminations is not None else None
        else:
            output_state_means = torch.gather(state_means, 0, model_ids.repeat(1, 1, self.state_dim)).squeeze(0)
            output_extensions = (
                torch.gather(extensions, 0, model_ids.repeat(1, 1, self.extension_dim)).squeeze(0)
                if extensions is not None
                else None
            )
            output_contacts = (
                torch.gather(contacts, 0, model_ids.repeat(1, 1, self.contact_dim)).squeeze(0)
                if contacts is not None
                else None
            )
            output_terminations = (
                torch.gather(terminations, 0, model_ids.repeat(1, 1, self.termination_dim)).squeeze(0)
                if terminations is not None
                else None
            )

        aleatoric_uncertainty = state_stds.mean(dim=0).sum(dim=1)
        epistemic_uncertainty = (
            state_means.std(dim=0).sum(dim=1)
            if self.ensemble_size > 1
            else torch.zeros(output_state_means.shape[0], device=self.device)
        )
        return (
            output_state_means,
            aleatoric_uncertainty,
            epistemic_uncertainty,
            output_extensions,
            output_contacts,
            output_terminations,
        )

    def _state_head_outputs(self, x_state_batch, x_action_batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Predicted mean and std of every state head, [ensemble, B, state_dim] each."""
        state_base_output = self.state_base(x_state_batch, x_action_batch)
        state_means, state_stds = [], []
        for head in self.state_heads:
            state_mean, state_std = head(state_base_output, x_state_batch)
            if self.prediction_type == "sequence":
                state_mean = state_mean[:, -1]
                state_std = state_std[:, -1]
            state_means.append(state_mean)
            state_stds.append(state_std)
        return torch.stack(state_means), torch.stack(state_stds)

    def compute_loss(
        self,
        state_batch,
        action_batch,
        extension_batch,
        contact_batch,
        termination_batch,
        bootstrap=False,
        loss_type="gaussian_nll",
    ):
        state_losses = []
        sequence_losses = []
        bound_losses = []
        kl_losses = []
        extension_losses = []
        contact_losses = []
        termination_losses = []

        for i in range(self.ensemble_size):
            self.reset()  # reset rnn hidden states, otherwise hidden states leak from one head to another!

            if bootstrap:
                ids = torch.randint(0, state_batch.shape[0], (state_batch.shape[0],), device=self.device)
            else:
                ids = torch.arange(0, state_batch.shape[0], device=self.device)

            state_loss, sequence_loss, bound_loss, kl_loss = self.compute_state_loss(
                self.state_heads[i], state_batch[ids], action_batch[ids], loss_type=loss_type
            )

            if self.auxiliary_heads is not None:
                extension_loss, contact_loss, termination_loss = self.compute_auxiliary_loss(
                    self.auxiliary_heads[i],
                    state_batch[ids],
                    action_batch[ids],
                    extension_batch[ids] if extension_batch is not None else None,
                    contact_batch[ids] if contact_batch is not None else None,
                    termination_batch[ids] if termination_batch is not None else None,
                )
            else:
                extension_loss = torch.tensor(0.0, device=self.device)
                contact_loss = torch.tensor(0.0, device=self.device)
                termination_loss = torch.tensor(0.0, device=self.device)

            state_losses.append(state_loss.unsqueeze(0))
            sequence_losses.append(sequence_loss.unsqueeze(0))
            bound_losses.append(bound_loss.unsqueeze(0))
            kl_losses.append(kl_loss.unsqueeze(0))
            extension_losses.append(extension_loss.unsqueeze(0))
            contact_losses.append(contact_loss.unsqueeze(0))
            termination_losses.append(termination_loss.unsqueeze(0))

        state_loss = torch.mean(torch.cat(state_losses, dim=0), dim=0)
        sequence_loss = torch.mean(torch.cat(sequence_losses, dim=0), dim=0)
        bound_loss = torch.mean(torch.cat(bound_losses, dim=0), dim=0)
        kl_loss = torch.mean(torch.cat(kl_losses, dim=0), dim=0)
        extension_loss = torch.mean(torch.cat(extension_losses, dim=0), dim=0)
        contact_loss = torch.mean(torch.cat(contact_losses, dim=0), dim=0)
        termination_loss = torch.mean(torch.cat(termination_losses, dim=0), dim=0)
        return state_loss, sequence_loss, bound_loss, kl_loss, extension_loss, contact_loss, termination_loss

    def compute_state_loss(self, head, state_batch, action_batch, loss_type="gaussian_nll"):
        if loss_type == "gaussian_nll" and not head.output_std:
            raise ValueError("loss_type='gaussian_nll' requires a state head with a logstd output (state_head.logstd).")
        assert (
            self.forecast_horizon + self.history_horizon == state_batch.shape[1]
            and self.forecast_horizon + self.history_horizon == action_batch.shape[1]
        ), (
            f"Training trajectories must be of length "
            f"T = History + Forecast "
            f"(={self.forecast_horizon + self.history_horizon}). "
            f"The input has: T={state_batch.shape[1]}"
        )

        forecast_horizon = self.forecast_horizon
        x_state_batch = state_batch[:, : self.history_horizon]
        state_losses = []
        sequence_losses = []
        bound_losses = []
        kl_losses = []

        for i in range(forecast_horizon):
            if self.prediction_type == "single":
                state_target = state_batch[:, self.history_horizon + i]
            elif self.prediction_type == "sequence":
                state_target = state_batch[:, i + 1 : self.history_horizon + i + 1]
            else:
                raise ValueError("Invalid state prediction type.")

            if self.base_cfg.type in ["rnn", "rssm"] and i > 0:
                x_action_batch = action_batch[:, self.history_horizon + i : self.history_horizon + i + 1]
                if self.prediction_type == "sequence":
                    state_target = state_target[:, [-1]]
            else:
                # rnn starting state is the the whole history sequence for i=0 and only the newest state later.
                # MLP always needs the full history sequence.
                x_action_batch = action_batch[:, i + 1 : self.history_horizon + i + 1]

            state_mean_pred, state_std_pred = head.forward(
                self.state_base.forward(x_state_batch, x_action_batch), x_state_batch
            )

            state_loss, sequence_loss = self.compute_regression_loss(
                state_mean_pred, state_std_pred, state_target, loss_type=loss_type
            )
            bound_loss = self.compute_bound_loss(head) if head.output_std else torch.tensor(0.0, device=self.device)
            kl_loss = self.state_base.kl_loss if self.base_cfg.type == "rssm" else torch.tensor(0.0, device=self.device)

            state_losses.append(state_loss.unsqueeze(0))
            sequence_losses.append(sequence_loss.unsqueeze(0))
            bound_losses.append(bound_loss.unsqueeze(0))
            kl_losses.append(kl_loss.unsqueeze(0))

            if self.prediction_type == "sequence":
                state_mean_pred = state_mean_pred[:, -1]
                state_std_pred = state_std_pred[:, -1]

            if self.base_cfg.type in ["rnn", "rssm"]:
                x_state_batch = (
                    (
                        torch.randn_like(state_mean_pred, device=self.device) * state_std_pred + state_mean_pred
                    ).unsqueeze(1)
                    if head.output_std
                    else state_mean_pred.unsqueeze(1)
                )
            else:
                x_state_batch = torch.cat(
                    [
                        x_state_batch[:, 1:].clone(),
                        (
                            torch.randn_like(state_mean_pred, device=self.device) * state_std_pred + state_mean_pred
                        ).unsqueeze(1)
                        if head.output_std
                        else state_mean_pred.unsqueeze(1),
                    ],
                    dim=1,
                )

        state_loss = torch.mean(torch.cat(state_losses, dim=0), dim=0)
        sequence_loss = torch.mean(torch.cat(sequence_losses, dim=0), dim=0)
        bound_loss = torch.mean(torch.cat(bound_losses, dim=0), dim=0)
        kl_loss = torch.mean(torch.cat(kl_losses, dim=0), dim=0)
        return state_loss, sequence_loss, bound_loss, kl_loss

    def forecast(self, state_batch, action_batch, steps: int | None = None) -> Forecast:
        """Open-loop forecast of `steps` states after the history, feeding back the ensemble-mean prediction.

        Deterministic counterpart of compute_state_loss (same windows and action indexing), used for evaluation.
        Nothing is sampled, so the uncertainties of each step are those of a single step from the fed-back mean: they
        don't accumulate over the horizon.

        state_batch: [B, >= H, state_dim]. Only the first H states (the history) are used.
        action_batch: [B, >= H + steps, action_dim]. Action k is the one that leads to state k (how Isaac Lab records
            them), so action H + i produces forecast step i.
        steps: number of forecast steps, forecast_horizon by default. Longer forecasts are fine, the model just
            wasn't trained on them.
        """
        steps = self.forecast_horizon if steps is None else steps
        assert action_batch.shape[1] >= self.history_horizon + steps, (
            f"{steps} forecast steps need {self.history_horizon + steps} actions, got {action_batch.shape[1]}"
        )
        self.reset()
        x_state_batch = state_batch[:, : self.history_horizon]
        per_step = []
        for i in range(steps):
            if self.base_cfg.type in ["rnn", "rssm"] and i > 0:
                x_action_batch = action_batch[:, self.history_horizon + i : self.history_horizon + i + 1]
            else:
                x_action_batch = action_batch[:, i + 1 : self.history_horizon + i + 1]
            means, stds = self._state_head_outputs(x_state_batch, x_action_batch)  # [ensemble, B, state_dim]
            state_mean = means.mean(dim=0)
            no_epistemic = state_mean.new_zeros(len(state_mean))  # a single head has no disagreement
            per_step.append(
                Forecast(
                    mean=state_mean,
                    aleatoric=stds.mean(dim=0).sum(dim=1),
                    epistemic=means.std(dim=0).sum(dim=1) if self.ensemble_size > 1 else no_epistemic,
                    aleatoric_var=stds.square().mean(dim=0),
                    epistemic_var=means.var(dim=0, correction=0),  # population variance: the heads are the mixture
                )
            )
            if self.base_cfg.type in ["rnn", "rssm"]:
                x_state_batch = state_mean.unsqueeze(1)
            else:
                x_state_batch = torch.cat([x_state_batch[:, 1:], state_mean.unsqueeze(1)], dim=1)
        self.reset()
        return Forecast(*(torch.stack(values, dim=1) for values in zip(*per_step)))

    def sample_forecast(
        self, state_batch, action_batch, num_particles: int, steps: int | None = None
    ) -> SampledForecast:
        """Monte Carlo forecast: num_particles sampled rollouts per window (trajectory sampling, as in PETS).

        Every particle keeps one head for the whole rollout ("TS-infinity"), the same number of particles per head.
        At each step its head predicts mean and std, the next state is drawn from that Gaussian and fed back. The
        spread of the particles is the predictive uncertainty propagated through the rollout, which forecast() can't
        show because it feeds back the mean. Same inputs and action indexing as forecast().
        """
        steps = self.forecast_horizon if steps is None else steps
        B, K, E, H = len(state_batch), num_particles, self.ensemble_size, self.history_horizon
        assert K % E == 0 and K > E, f"num_particles ({K}) must be a multiple of the ensemble size ({E}), above it"
        assert action_batch.shape[1] >= H + steps, f"{steps} forecast steps need {H + steps} actions"
        head = torch.arange(K, device=state_batch.device) % E  # particle k uses head k % E
        rows, row_head = torch.arange(B * K, device=state_batch.device), head.repeat(B)
        self.reset()
        # The first step starts from the recorded history, the same for all particles of a window: computed once per
        # window (for the RNN this includes warming up its memory over the H history steps), then copied K times.
        # Particles of a window are neighbours in the batch: row = window * K + particle.
        means, stds = self._state_head_outputs(state_batch[:, :H], action_batch[:, 1 : H + 1])  # [ensemble, B, ...]
        means, stds = means.repeat_interleave(K, dim=1), stds.repeat_interleave(K, dim=1)  # [ensemble, B*K, ...]
        self.state_base.repeat_hidden_states(K)
        x_state_batch = state_batch[:, :H].repeat_interleave(K, dim=0)  # [B*K, H, state_dim]
        action_batch = action_batch.repeat_interleave(K, dim=0)
        samples = []
        for i in range(steps):
            if i > 0:
                if self.base_cfg.type in ["rnn", "rssm"]:
                    x_action_batch = action_batch[:, H + i : H + i + 1]
                else:
                    x_action_batch = action_batch[:, i + 1 : H + i + 1]
                means, stds = self._state_head_outputs(x_state_batch, x_action_batch)  # [ensemble, B*K, state_dim]
            mean, std = means[row_head, rows], stds[row_head, rows]  # each particle's own head
            sample = mean + std * torch.randn_like(mean)
            samples.append(sample)
            if self.base_cfg.type in ["rnn", "rssm"]:
                x_state_batch = sample.unsqueeze(1)
            else:
                x_state_batch = torch.cat([x_state_batch[:, 1:], sample.unsqueeze(1)], dim=1)
        self.reset()

        particles = torch.stack(samples, dim=1).view(B, K, steps, -1)
        by_head = particles.view(B, K // E, E, steps, -1)  # particle k = j * E + head, K / E particles per head
        aleatoric_var = by_head.var(dim=1).mean(dim=1)  # unbiased variance within each head, mean over heads
        # Variance of the heads' particle means, minus what their sampling noise adds to it in expectation
        epistemic_var = by_head.mean(dim=1).var(dim=1, correction=0) - (E - 1) / E * aleatoric_var / (K // E)
        return SampledForecast(
            particles=particles,
            head=head,
            mean=particles.mean(dim=1),
            var=aleatoric_var + epistemic_var,
            aleatoric_var=aleatoric_var,
            epistemic_var=epistemic_var,
        )

    def compute_auxiliary_loss(
        self, head, state_batch, action_batch, extension_batch, contact_batch, termination_batch
    ):
        # shape checks
        assert all(
            self.forecast_horizon + self.history_horizon == batch.shape[1]
            for batch in (
                state_batch,
                action_batch,
                extension_batch,
                contact_batch,
                termination_batch,
            )
            if batch is not None
        ), (
            f"Training trajectories must be of length "
            f"T = History + Forecast "
            f"(={self.forecast_horizon + self.history_horizon})."
        )

        forecast_horizon = self.forecast_horizon
        x_state_batch = state_batch[:, : self.history_horizon]
        extension_losses = []
        contact_losses = []
        termination_losses = []

        for i in range(forecast_horizon):
            extension_target = extension_batch[:, self.history_horizon + i] if extension_batch is not None else None
            contact_target = contact_batch[:, self.history_horizon + i] if contact_batch is not None else None
            termination_target = (
                termination_batch[:, self.history_horizon + i] if termination_batch is not None else None
            )

            if self.base_cfg.type in ["rnn", "rssm"] and i > 0:
                x_action_batch = action_batch[:, self.history_horizon + i : self.history_horizon + i + 1]
            else:
                x_action_batch = action_batch[:, i + 1 : self.history_horizon + i + 1]

            extension_pred, contact_pred, termination_pred = head.forward(
                self.auxiliary_base.forward(x_state_batch, x_action_batch), x_state_batch
            )

            extension_loss = (
                self.compute_extension_loss(extension_pred, extension_target)
                if self.extension_dim > 0
                else torch.tensor(0.0, device=self.device)
            )
            contact_loss = (
                self.compute_contact_loss(contact_pred, contact_target)
                if self.contact_dim > 0
                else torch.tensor(0.0, device=self.device)
            )
            termination_loss = (
                self.compute_termination_loss(termination_pred, termination_target)
                if self.termination_dim > 0
                else torch.tensor(0.0, device=self.device)
            )

            extension_losses.append(extension_loss.unsqueeze(0))
            contact_losses.append(contact_loss.unsqueeze(0))
            termination_losses.append(termination_loss.unsqueeze(0))

            if self.base_cfg.type in ["rnn", "rssm"]:
                x_state_batch = state_batch[:, self.history_horizon + i : self.history_horizon + i + 1]
            else:
                x_state_batch = torch.cat(
                    [
                        x_state_batch[:, 1:].clone(),
                        state_batch[:, self.history_horizon + i : self.history_horizon + i + 1],
                    ],
                    dim=1,
                )

        extension_loss = torch.mean(torch.cat(extension_losses, dim=0), dim=0)
        contact_loss = torch.mean(torch.cat(contact_losses, dim=0), dim=0)
        termination_loss = torch.mean(torch.cat(termination_losses, dim=0), dim=0)
        return extension_loss, contact_loss, termination_loss

    def compute_regression_loss(self, state_mean_pred, state_std_pred, state_target, loss_type="gaussian_nll"):
        if loss_type == "mse":
            if self.prediction_type == "sequence":
                state_mean_pred_seq, state_mean_pred = state_mean_pred[:, :-1], state_mean_pred[:, -1]
                state_std_pred_seq, state_std_pred = state_std_pred[:, :-1], state_std_pred[:, -1]
                state_target_seq, state_target = state_target[:, :-1], state_target[:, -1]
                state_pred_seq = (
                    torch.randn_like(state_mean_pred_seq, device=self.device) * state_std_pred_seq + state_mean_pred_seq
                )
                state_pred_seq = state_pred_seq.flatten(0, 1)
                state_target_seq = state_target_seq.flatten(0, 1)
                sequence_loss = torch.sum(torch.square(state_pred_seq - state_target_seq), dim=1).mean(dim=0)
            else:
                sequence_loss = torch.tensor(0.0, device=self.device)
            state_pred = torch.randn_like(state_mean_pred, device=self.device) * state_std_pred + state_mean_pred
            state_loss = torch.sum(torch.square(state_pred - state_target), dim=1).mean(dim=0)
            return state_loss, sequence_loss
        elif loss_type == "gaussian_nll":
            if self.prediction_type == "sequence":
                state_mean_pred_seq, state_mean_pred = state_mean_pred[:, :-1], state_mean_pred[:, -1]
                state_std_pred_seq, state_std_pred = state_std_pred[:, :-1], state_std_pred[:, -1]
                state_target_seq, state_target = state_target[:, :-1], state_target[:, -1]
                state_mean_pred_seq = state_mean_pred_seq.flatten(0, 1)
                state_std_pred_seq = state_std_pred_seq.flatten(0, 1)
                state_target_seq = state_target_seq.flatten(0, 1)
                sequence_loss = nn.GaussianNLLLoss()(state_mean_pred_seq, state_target_seq, state_std_pred_seq**2)
            else:
                sequence_loss = torch.tensor(0.0, device=self.device)
            state_loss = nn.GaussianNLLLoss()(state_mean_pred, state_target, state_std_pred**2)
            return state_loss, sequence_loss
        else:
            raise ValueError("Invalid loss type.")

    def compute_bound_loss(self, head):
        return torch.mean(head.state_max_logstd) - torch.mean(head.state_min_logstd)

    def compute_extension_loss(self, extension_pred, extension_target):
        if extension_pred is None or extension_target is None:
            return torch.tensor(0.0, device=self.device)
        if self.prediction_type == "sequence":
            extension_pred = extension_pred[:, -1]
        return nn.MSELoss()(extension_pred, extension_target)

    def compute_contact_loss(self, contact_pred, contact_target):
        if contact_pred is None or contact_target is None:
            return torch.tensor(0.0, device=self.device)
        if self.prediction_type == "sequence":
            contact_pred = contact_pred[:, -1]
        return nn.BCEWithLogitsLoss()(contact_pred, contact_target)

    def compute_termination_loss(self, termination_pred, termination_target):
        if termination_pred is None or termination_target is None:
            return torch.tensor(0.0, device=self.device)
        if self.prediction_type == "sequence":
            termination_pred = termination_pred[:, -1]
        return nn.BCEWithLogitsLoss()(termination_pred, termination_target)

    def reset(self):
        self.state_base.reset()
        if self.auxiliary_base is not None:
            self.auxiliary_base.reset()

    def reset_partial(self, batch_indices):
        self.state_base.reset_partial(batch_indices)
        for head in self.state_heads:
            head.reset_partial(batch_indices)
        if self.auxiliary_base is not None:
            self.auxiliary_base.reset_partial(batch_indices)
            for head in self.auxiliary_heads:
                head.reset_partial(batch_indices)

    def train(self, mode=True):
        super().train(mode)
        return self
