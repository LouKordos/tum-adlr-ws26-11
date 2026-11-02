from dataclasses import dataclass

from torch import nn


# Configurations for RNN
@dataclass
class RNNConfig:
    type: str = "gru"
    num_layers: int = 1
    hidden_size: int = 256


# Configurations for MLP
@dataclass
class MLPConfig:
    hidden_dims: tuple[int, ...] = (256, 256)
    activation: type[nn.Module] = nn.ReLU


@dataclass
class MLPBaseConfig:
    network: MLPConfig


@dataclass
class MLPStateHeadConfig:
    mean: MLPConfig
    logstd: MLPConfig | None


@dataclass
class MLPAuxiliaryHeadConfig:
    extension: MLPConfig | None
    contact: MLPConfig | None
    termination: MLPConfig | None
