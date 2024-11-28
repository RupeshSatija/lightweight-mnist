from dataclasses import dataclass
from typing import Any, Dict

import torch


@dataclass
class ModelConfig:
    conv1_channels: int = 8
    conv2_channels: int = 16
    fc_features: int = 32


@dataclass
class TrainingConfig:
    batch_size: int = 128
    epochs: int = 1
    learning_rate: float = 0.003
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def get_config() -> Dict[str, Any]:
    model_config = ModelConfig()
    training_config = TrainingConfig()

    return {**model_config.__dict__, **training_config.__dict__}
