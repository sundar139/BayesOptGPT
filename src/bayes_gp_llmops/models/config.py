from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

PoolingStrategy = Literal["masked_mean"]


class ModelConfig(BaseModel):
    """Configuration for the tiny LLaMA-style classifier."""

    vocab_size: int = Field(ge=128)
    max_sequence_length: int = Field(ge=8)
    hidden_size: int = Field(ge=64)
    num_layers: int = Field(ge=1)
    num_attention_heads: int = Field(ge=1)
    feedforward_multiplier: float = Field(ge=1.0)
    dropout: float = Field(ge=0.0, le=1.0)
    num_classes: int = Field(ge=2)
    rope_base: float = Field(default=10000.0, gt=0.0)
    pooling: PoolingStrategy = Field(default="masked_mean")

    @model_validator(mode="after")
    def validate_shape_constraints(self) -> ModelConfig:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads.")
        head_dim = self.hidden_size // self.num_attention_heads
        if head_dim % 2 != 0:
            raise ValueError("Per-head dimension must be even for RoPE.")
        return self


def load_model_config(path: Path) -> ModelConfig:
    """Load model configuration from YAML."""

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration file {path} must contain a mapping at the top level.")
    model_section = payload.get("model")
    if not isinstance(model_section, dict):
        raise ValueError(f"Configuration file {path} must include a 'model' mapping.")
    return ModelConfig.model_validate(model_section)


def model_config_to_dict(config: ModelConfig) -> dict[str, Any]:
    """Convert validated model configuration to a JSON-serializable dictionary."""

    return config.model_dump(mode="json")
