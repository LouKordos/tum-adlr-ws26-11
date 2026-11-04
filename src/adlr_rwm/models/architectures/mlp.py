import torch
from omegaconf import DictConfig
from torch import nn


ACTIVATIONS = {
    "relu": nn.ReLU
}


def build_mlp(
    input_dim: int,
    output_dim: int | None,
    config: DictConfig,
    device: str,
) -> nn.Sequential:
    """Build an MLP with ReLU activations and an optional output layer."""

    layers = []
    curr_dim = input_dim

    for hidden_dim in config.hidden_dims:
        layers.extend([
            nn.Linear(curr_dim, hidden_dim),
            ACTIVATIONS[config.activation]()
        ])
        curr_dim = hidden_dim

    if output_dim is not None:
        layers.append(nn.Linear(curr_dim, output_dim))

    return nn.Sequential(*layers).to(device)


class MLPBase(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int | None,
        device: str,
        architecture_config: DictConfig
    ):
        super().__init__()

        self.input_dim = input_dim
        self.device = device

        self.layers = build_mlp(
            input_dim=input_dim,
            output_dim=output_dim,
            config=architecture_config,
            device=device,
        )

    def forward(self, x_state_batch, x_action_batch):
        x = torch.cat([x_state_batch, x_action_batch], dim=-1)
        x = x.flatten(1, 2)
        return self.layers(x)

    def reset(self):
        pass

    def reset_partial(self, batch_indices):
        pass


class MLPStateHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        device: str,
        architecture_config: DictConfig,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.state_dim = state_dim
        self.device = device

        self.state_mean_layers = build_mlp(
            input_dim=input_dim,
            output_dim=state_dim,
            config=architecture_config.mean,
            device=device,
        )

        self.output_std = architecture_config.logstd is not None

        if self.output_std:
            self.state_logstd_layers = build_mlp(
                input_dim=input_dim,
                output_dim=state_dim,
                config=architecture_config.logstd,
                device=device,
            )

            self.state_min_logstd = nn.Parameter(
                torch.ones(1, state_dim, device=device) * -5.0
            )
            self.state_log_delta_logstd = nn.Parameter(
                torch.zeros(1, state_dim, device=device)
            )

    def forward(self, x, x_state_batch):
        if x.dim() == 3:
            sequence_len = x.shape[1]
            x = x.flatten(0, 1)
            x_state_batch = x_state_batch.flatten(0, 1).unsqueeze(1)
        else:
            sequence_len = 0

        state_mean = self.state_mean_layers(x) + x_state_batch[:, -1]

        if self.output_std:
            state_logstd = self.state_logstd_layers(x)

            state_max_logstd = (
                self.state_min_logstd
                + torch.exp(self.state_log_delta_logstd)
            )

            state_logstd = (
                state_max_logstd
                - nn.functional.softplus(state_max_logstd - state_logstd)
            )

            state_logstd = (
                self.state_min_logstd
                + nn.functional.softplus(state_logstd - self.state_min_logstd)
            )
        else:
            state_logstd = torch.full_like(state_mean, -torch.inf)

        if sequence_len > 0:
            state_mean = state_mean.view(-1, sequence_len, self.state_dim)
            state_logstd = state_logstd.view(-1, sequence_len, self.state_dim)

        return state_mean, torch.exp(state_logstd)

    def reset(self):
        pass

    def reset_partial(self, batch_indices):
        pass


class MLPAuxiliaryHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        extension_dim: int,
        contact_dim: int,
        termination_dim: int,
        device: str,
        architecture_config: DictConfig,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.extension_dim = extension_dim
        self.contact_dim = contact_dim
        self.termination_dim = termination_dim
        self.device = device

        if extension_dim > 0:
            assert architecture_config.extension is not None
            self.extension_layers = build_mlp(
                input_dim, extension_dim,
                architecture_config.extension, device,
            )

        if contact_dim > 0:
            assert architecture_config.contact is not None
            self.contact_layers = build_mlp(
                input_dim, contact_dim,
                architecture_config.contact, device,
            )

        if termination_dim > 0:
            assert architecture_config.termination is not None
            self.termination_layers = build_mlp(
                input_dim, termination_dim,
                architecture_config.termination, device,
            )

    def forward(self, x, x_state_batch):
        if x.dim() == 3:
            sequence_len = x.shape[1]
            x = x.flatten(0, 1)
        else:
            sequence_len = 0

        extension_pred = (
            self.extension_layers(x) if self.extension_dim > 0 else None
        )
        contact_logits = (
            self.contact_layers(x) if self.contact_dim > 0 else None
        )
        termination_logits = (
            self.termination_layers(x) if self.termination_dim > 0 else None
        )

        if sequence_len > 0:
            extension_pred = (
                extension_pred.view(-1, sequence_len, self.extension_dim)
                if self.extension_dim > 0 else None
            )
            contact_logits = (
                contact_logits.view(-1, sequence_len, self.contact_dim)
                if self.contact_dim > 0 else None
            )
            termination_logits = (
                termination_logits.view(-1, sequence_len, self.termination_dim)
                if self.termination_dim > 0 else None
            )

        return extension_pred, contact_logits, termination_logits

    def reset(self):
        pass

    def reset_partial(self, batch_indices):
        pass
