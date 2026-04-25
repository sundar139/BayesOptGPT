from __future__ import annotations

from dataclasses import dataclass

import optuna

from bayes_gp_llmops.models.config import ModelConfig


@dataclass(frozen=True)
class SampledHyperparameters:
    """Sampled hyperparameters for one Optuna trial."""

    learning_rate: float
    weight_decay: float
    dropout: float
    batch_size: int
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    feedforward_multiplier: float
    warmup_ratio: float
    gradient_clip_norm: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "dropout": self.dropout,
            "batch_size": self.batch_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "feedforward_multiplier": self.feedforward_multiplier,
            "warmup_ratio": self.warmup_ratio,
            "gradient_clip_norm": self.gradient_clip_norm,
        }


def sample_hyperparameters(
    *,
    trial: optuna.trial.Trial,
    base_model_config: ModelConfig,
) -> SampledHyperparameters:
    """Sample a constrained and realistic tuning configuration."""

    learning_rate = trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 5e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.3)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 48, 64])

    hidden_size = trial.suggest_categorical("hidden_size", [128, 192, 256, 320])
    valid_heads = _valid_attention_head_choices(hidden_size=hidden_size)
    num_attention_heads = trial.suggest_categorical("num_attention_heads", valid_heads)

    num_layers = trial.suggest_int("num_layers", 2, 6)
    feedforward_multiplier = trial.suggest_categorical("feedforward_multiplier", [2.0, 3.0, 4.0])
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.2)
    gradient_clip_norm = trial.suggest_float("gradient_clip_norm", 0.5, 2.0)

    _validate_model_compatibility(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        base_model_config=base_model_config,
    )

    return SampledHyperparameters(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        dropout=dropout,
        batch_size=batch_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        feedforward_multiplier=feedforward_multiplier,
        warmup_ratio=warmup_ratio,
        gradient_clip_norm=gradient_clip_norm,
    )


def _valid_attention_head_choices(*, hidden_size: int) -> list[int]:
    candidates = [4, 6, 8, 10]
    valid = [
        heads
        for heads in candidates
        if hidden_size % heads == 0 and ((hidden_size // heads) % 2 == 0)
    ]
    if not valid:
        raise ValueError(f"No valid attention head choices for hidden size {hidden_size}.")
    return valid


def _validate_model_compatibility(
    *,
    hidden_size: int,
    num_attention_heads: int,
    base_model_config: ModelConfig,
) -> None:
    if hidden_size < 64:
        raise ValueError("hidden_size must remain >= 64.")
    if hidden_size % num_attention_heads != 0:
        raise ValueError("hidden_size must be divisible by num_attention_heads.")
    if ((hidden_size // num_attention_heads) % 2) != 0:
        raise ValueError("Per-head hidden dimension must be even for RoPE.")
    if base_model_config.max_sequence_length < 8:
        raise ValueError("max_sequence_length must be >= 8.")
