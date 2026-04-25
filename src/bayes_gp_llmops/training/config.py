from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

SchedulerType = Literal["none", "linear", "cosine"]


class TrainConfig(BaseModel):
    """Configuration for baseline model training."""

    learning_rate: float = Field(gt=0.0)
    weight_decay: float = Field(ge=0.0)
    epochs: int = Field(ge=1)
    gradient_clip_norm: float = Field(gt=0.0)
    early_stopping_patience: int = Field(ge=1)
    mixed_precision: bool = Field(default=True)
    checkpoint_dir: Path
    random_seed: int = Field(default=42)
    device_preference: str = Field(default="auto")
    log_frequency: int = Field(default=100, ge=1)
    scheduler: SchedulerType = Field(default="cosine")
    warmup_ratio: float = Field(default=0.05, ge=0.0, lt=1.0)
    max_train_batches_per_epoch: int | None = Field(default=None, ge=1)
    max_validation_batches_per_epoch: int | None = Field(default=None, ge=1)
    max_test_batches_per_epoch: int | None = Field(default=None, ge=1)


def load_train_config(path: Path) -> TrainConfig:
    """Load training configuration from YAML."""

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration file {path} must contain a mapping at the top level.")
    training_section = payload.get("training")
    if not isinstance(training_section, dict):
        raise ValueError(f"Configuration file {path} must include a 'training' mapping.")
    return TrainConfig.model_validate(training_section)


def train_config_to_dict(config: TrainConfig) -> dict[str, Any]:
    """Convert validated training configuration to a JSON-serializable dictionary."""

    return config.model_dump(mode="json")
