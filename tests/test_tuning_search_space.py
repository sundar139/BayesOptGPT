from __future__ import annotations

from typing import Any, cast

import optuna

from bayes_gp_llmops.models.config import ModelConfig
from bayes_gp_llmops.tuning.search_space import sample_hyperparameters


def test_search_space_samples_valid_head_dimension() -> None:
    trial = optuna.trial.FixedTrial(
        {
            "learning_rate": 1e-3,
            "weight_decay": 1e-2,
            "dropout": 0.1,
            "batch_size": 32,
            "hidden_size": 256,
            "num_attention_heads": 8,
            "num_layers": 4,
            "feedforward_multiplier": 4.0,
            "warmup_ratio": 0.05,
            "gradient_clip_norm": 1.0,
        }
    )
    base_model = ModelConfig(
        vocab_size=16000,
        max_sequence_length=256,
        hidden_size=256,
        num_layers=4,
        num_attention_heads=8,
        feedforward_multiplier=4.0,
        dropout=0.1,
        num_classes=4,
        rope_base=10000.0,
        pooling="masked_mean",
    )

    sampled = sample_hyperparameters(trial=cast(Any, trial), base_model_config=base_model)
    assert sampled.hidden_size % sampled.num_attention_heads == 0
    assert (sampled.hidden_size // sampled.num_attention_heads) % 2 == 0
    assert sampled.learning_rate > 0.0
