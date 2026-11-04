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



import torch
from omegaconf import DictConfig
from torch import nn

from .architectures import MLPAuxiliaryHead, MLPBase, MLPStateHead, RNNBase


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
                MLPStateHead(
                             self.base_output_dim,
                             self.state_dim,
                             self.device,
                             self.state_head_cfg
                ).to(self.device)
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
                architecture_config=self.base_cfg.network
            )
        elif self.base_cfg.type == "rnn":
            input_dim = self.state_dim + self.action_dim
            self.base_output_dim = self.base_cfg.hidden_size
            self.prediction_type = "single"
            return RNNBase(
                input_dim=input_dim,
                device=self.device,
                architecture_config=self.base_cfg
            )
        else:
            raise ValueError("Invalid architecture type.")

    def forward(
                self,
                x_state_batch: torch.Tensor,
                x_action_batch: torch.Tensor,
                model_ids=None
            ):

        # shape checks and safety measures
        x_state_batch = x_state_batch.to(self.device)
        x_action_batch = x_action_batch.to(self.device)

        assert x_state_batch.shape[:2] == x_action_batch.shape[:2], (
            f"Batch size and sequence length must match between x_state_batch {x_state_batch.shape[:2]} "
            f"and x_action_batch {x_action_batch.shape[:2]}."
        )

        if self.base_cfg.type == "mlp":
            assert x_state_batch.shape[1] == self.history_horizon, (
                f"MLP requires full history_horizon ({self.history_horizon}), "
                f"got T={x_state_batch.shape[1]}."
            )
        elif self.base_cfg.type == "rnn":
            if self.state_base.memory.hidden_states is None:
                # Cold start: requires full history window to warm up memory
                assert x_state_batch.shape[1] == self.history_horizon, (
                    f"RNN cold start requires history_horizon ({self.history_horizon}), "
                    f"got T={x_state_batch.shape[1]}."
                )
            else:
                # Active hidden state: expect single-step autoregressive update
                assert x_state_batch.shape[1] == 1, (
                    f"RNN with active hidden_states expects single-step input (T=1), "
                    f"got T={x_state_batch.shape[1]}. Call model.reset() first if starting a new sequence."
                )


        state_means, state_stds, extensions, contacts, terminations = [], [], [], [], []
        state_base_output = self.state_base(x_state_batch, x_action_batch)

        for head in self.state_heads:
            state_mean, state_std = head(state_base_output, x_state_batch)
            if self.prediction_type == "sequence":
                state_mean = state_mean[:, -1]
                state_std = state_std[:, -1]
            state_means.append(state_mean.unsqueeze(0))
            state_stds.append(state_std.unsqueeze(0))

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

        state_means = torch.cat(state_means, dim=0)
        state_stds = torch.cat(state_stds, dim=0)
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

    def compute_loss(
        self, state_batch, action_batch, extension_batch, contact_batch, termination_batch, bootstrap=False
    ):
        state_losses = []
        sequence_losses = []
        bound_losses = []
        kl_losses = []
        extension_losses = []
        contact_losses = []
        termination_losses = []

        for i in range(self.ensemble_size):
            self.reset() # reset rnn hidden states, otherwise hidden states leak from one head to another!

            if bootstrap:
                ids = torch.randint(0, state_batch.shape[0], (state_batch.shape[0],), device=self.device)
            else:
                ids = torch.arange(0, state_batch.shape[0], device=self.device)

            state_loss, sequence_loss, bound_loss, kl_loss = self.compute_state_loss(
                self.state_heads[i], state_batch[ids], action_batch[ids]
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

    def compute_state_loss(self, head, state_batch, action_batch):
        assert (
            self.forecast_horizon + self.history_horizon == state_batch.shape[1]
            and
            self.forecast_horizon + self.history_horizon == action_batch.shape[1]
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

            state_loss, sequence_loss = self.compute_regression_loss(state_mean_pred, state_std_pred, state_target)
            bound_loss = self.compute_bound_loss(head) if head.output_std else torch.tensor(0.0, device=self.device)
            kl_loss = (
                self.state_base.kl_loss
                if self.base_cfg.type == "rssm"
                else torch.tensor(0.0, device=self.device)
            )

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
            ) if batch is not None
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

    def compute_regression_loss(self, state_mean_pred, state_std_pred, state_target, loss_type="mse"):
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
