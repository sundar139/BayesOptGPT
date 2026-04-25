from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from bayes_gp_llmops.models.classifier import TinyLlamaForSequenceClassification
from bayes_gp_llmops.models.config import ModelConfig
from bayes_gp_llmops.training.config import TrainConfig
from bayes_gp_llmops.training.trainer import Trainer, load_checkpoint


class TinyClassificationDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        *,
        size: int,
        sequence_length: int,
        vocab_size: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(7)
        self.input_ids = torch.randint(0, vocab_size, (size, sequence_length), generator=generator)
        self.attention_mask = torch.ones(size, sequence_length, dtype=torch.long)
        self.labels = torch.randint(0, num_classes, (size,), generator=generator)

    def __len__(self) -> int:
        return self.labels.size(0)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "labels": self.labels[index],
        }


def test_trainer_smoke_step_and_checkpoint_round_trip(tmp_path: Path) -> None:
    model_config = ModelConfig(
        vocab_size=128,
        max_sequence_length=16,
        hidden_size=64,
        num_layers=1,
        num_attention_heads=8,
        feedforward_multiplier=2.0,
        dropout=0.0,
        num_classes=4,
        rope_base=10000.0,
        pooling="masked_mean",
    )
    train_config = TrainConfig(
        learning_rate=5e-4,
        weight_decay=0.01,
        epochs=2,
        gradient_clip_norm=1.0,
        early_stopping_patience=2,
        mixed_precision=False,
        checkpoint_dir=tmp_path / "checkpoints",
        random_seed=11,
        device_preference="cpu",
        log_frequency=100,
        scheduler="linear",
        warmup_ratio=0.1,
        max_train_batches_per_epoch=2,
        max_validation_batches_per_epoch=1,
    )
    model = TinyLlamaForSequenceClassification(model_config)

    train_dataset = TinyClassificationDataset(
        size=16,
        sequence_length=10,
        vocab_size=model_config.vocab_size,
        num_classes=model_config.num_classes,
    )
    validation_dataset = TinyClassificationDataset(
        size=8,
        sequence_length=10,
        vocab_size=model_config.vocab_size,
        num_classes=model_config.num_classes,
    )
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
    validation_loader = DataLoader(validation_dataset, batch_size=4, shuffle=False)

    trainer = Trainer(
        model=model,
        config=train_config,
        device=torch.device("cpu"),
        num_classes=model_config.num_classes,
    )
    artifacts = trainer.fit(
        train_loader=train_loader,
        validation_loader=validation_loader,
        resolved_config={
            "data": {"debug_subset_size": 16},
            "model": model_config.model_dump(mode="json"),
            "training": train_config.model_dump(mode="json"),
            "runtime": {"device": "cpu", "debug_mode": True},
        },
    )

    assert artifacts.best_checkpoint_path.exists()
    assert artifacts.latest_checkpoint_path.exists()
    assert artifacts.history_path.exists()
    assert artifacts.resolved_config_path.exists()

    reloaded_model = TinyLlamaForSequenceClassification(model_config)
    checkpoint_payload = load_checkpoint(
        path=artifacts.best_checkpoint_path,
        model=reloaded_model,
        map_location="cpu",
    )
    assert isinstance(checkpoint_payload, dict)
    assert "epoch" in checkpoint_payload
