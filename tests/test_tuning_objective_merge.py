from __future__ import annotations

from pathlib import Path

from bayes_gp_llmops.data.config import DataLoaderConfig, DataPipelineConfig
from bayes_gp_llmops.models.config import ModelConfig
from bayes_gp_llmops.training.config import TrainConfig
from bayes_gp_llmops.tuning.objective import build_trial_configs
from bayes_gp_llmops.tuning.search_space import SampledHyperparameters


def test_build_trial_configs_applies_sampled_parameters(tmp_path: Path) -> None:
    base_data = DataPipelineConfig(
        dataloader=DataLoaderConfig(batch_size=32, num_workers=0, pin_memory=False)
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
    base_train = TrainConfig(
        learning_rate=3e-4,
        weight_decay=0.01,
        epochs=3,
        gradient_clip_norm=1.0,
        early_stopping_patience=2,
        mixed_precision=False,
        checkpoint_dir=tmp_path / "base-checkpoints",
        random_seed=42,
        device_preference="cpu",
        log_frequency=100,
        scheduler="cosine",
        warmup_ratio=0.05,
        max_train_batches_per_epoch=200,
        max_validation_batches_per_epoch=80,
        max_test_batches_per_epoch=80,
    )
    sampled = SampledHyperparameters(
        learning_rate=1e-3,
        weight_decay=1e-2,
        dropout=0.2,
        batch_size=16,
        hidden_size=192,
        num_layers=3,
        num_attention_heads=6,
        feedforward_multiplier=3.0,
        warmup_ratio=0.1,
        gradient_clip_norm=1.5,
    )

    merged = build_trial_configs(
        base_data_config=base_data,
        base_model_config=base_model,
        base_train_config=base_train,
        sampled=sampled,
        trial_number=7,
        trials_dir=tmp_path / "trials",
    )

    assert merged.data_config.dataloader.batch_size == 16
    assert merged.model_config.hidden_size == 192
    assert merged.model_config.num_attention_heads == 6
    assert merged.model_config.num_layers == 3
    assert merged.train_config.learning_rate == 1e-3
    assert merged.train_config.warmup_ratio == 0.1
    assert merged.train_config.checkpoint_dir == tmp_path / "trials" / "trial_0007" / "checkpoints"
