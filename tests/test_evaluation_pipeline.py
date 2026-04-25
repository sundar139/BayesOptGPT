from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from datasets import Dataset

from bayes_gp_llmops.data.datasets import DatasetSplits
from bayes_gp_llmops.data.tokenizer import train_and_save_tokenizer
from bayes_gp_llmops.evaluation.pipeline import run_evaluation_pipeline
from bayes_gp_llmops.models.classifier import TinyLlamaForSequenceClassification
from bayes_gp_llmops.models.config import ModelConfig


def test_evaluation_pipeline_smoke_with_synthetic_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_config_path = tmp_path / "data.yaml"
    model_config_path = tmp_path / "model.yaml"
    train_config_path = tmp_path / "train.yaml"
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir = tmp_path / "tokenizer"
    output_dir = tmp_path / "evaluation"

    corpus = [
        "World markets rally on positive economic signals",
        "Sports team wins a close championship game",
        "Business earnings surpass analyst expectations",
        "Technology firm launches new AI processor",
    ]
    train_and_save_tokenizer(
        corpus,
        output_dir=tokenizer_dir,
        vocab_size=128,
        min_frequency=1,
        max_sequence_length=16,
        corpus_source="synthetic",
    )

    _write_yaml(
        data_config_path,
        "\n".join(
            [
                "dataset:",
                "  name: ag_news",
                "  config: null",
                "  text_field: text",
                "  label_field: label",
                "  validation_split_ratio: 0.1",
                "  random_seed: 42",
                "  debug_subset_size: null",
                "tokenizer:",
                "  vocab_size: 128",
                "  min_frequency: 1",
                "  max_sequence_length: 16",
                "dataloader:",
                "  batch_size: 2",
                "  num_workers: 0",
                "  pin_memory: false",
                "paths:",
                f"  dataset_cache_dir: {tmp_path / 'hf_cache'}",
                f"  processed_cache_dir: {tmp_path / 'processed'}",
                f"  tokenizer_dir: {tokenizer_dir}",
            ]
        ),
    )
    _write_yaml(
        model_config_path,
        "\n".join(
            [
                "model:",
                "  vocab_size: 128",
                "  max_sequence_length: 16",
                "  hidden_size: 64",
                "  num_layers: 1",
                "  num_attention_heads: 8",
                "  feedforward_multiplier: 2.0",
                "  dropout: 0.0",
                "  num_classes: 4",
                "  rope_base: 10000.0",
                "  pooling: masked_mean",
            ]
        ),
    )
    _write_yaml(
        train_config_path,
        "\n".join(
            [
                "training:",
                "  learning_rate: 0.0003",
                "  weight_decay: 0.01",
                "  epochs: 1",
                "  gradient_clip_norm: 1.0",
                "  early_stopping_patience: 1",
                "  mixed_precision: false",
                f"  checkpoint_dir: {checkpoint_dir}",
                "  random_seed: 7",
                "  device_preference: cpu",
                "  log_frequency: 10",
                "  scheduler: none",
                "  warmup_ratio: 0.0",
                "  max_train_batches_per_epoch: 1",
                "  max_validation_batches_per_epoch: 2",
            ]
        ),
    )

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
    model = TinyLlamaForSequenceClassification(model_config)
    checkpoint_path = checkpoint_dir / "best.ckpt"
    torch.save({"model_state_dict": model.state_dict(), "epoch": 1}, checkpoint_path)

    splits = _synthetic_splits()
    monkeypatch.setattr(
        "bayes_gp_llmops.evaluation.pipeline.load_ag_news_splits",
        lambda _cfg: splits,
    )

    artifacts = run_evaluation_pipeline(
        data_config_path=data_config_path,
        model_config_path=model_config_path,
        train_config_path=train_config_path,
        checkpoint_path=checkpoint_path,
        device_override="cpu",
        output_dir=output_dir,
        enable_temperature_scaling=True,
        debug_mode=False,
        mlflow_enabled=False,
    )

    assert artifacts.metrics_validation_path.exists()
    assert artifacts.metrics_test_path.exists()
    assert artifacts.metrics_validation_calibrated_path is not None
    assert artifacts.metrics_validation_calibrated_path.exists()
    assert artifacts.predictions_validation_path.exists()
    assert artifacts.reliability_diagram_plot_path.exists()

    with artifacts.metrics_test_path.open("r", encoding="utf-8") as handle:
        metrics_payload = json.load(handle)
    assert "brier_score" in metrics_payload
    assert "ece" in metrics_payload
    assert "uncertainty_summary" in metrics_payload

    frame = _read_csv_rows(artifacts.predictions_test_path)
    assert frame
    required_prediction_columns = {
        "split",
        "index",
        "true_label",
        "predicted_label",
        "confidence",
        "entropy",
        "margin",
        "correct",
    }
    assert required_prediction_columns.issubset(set(frame[0].keys()))


def _synthetic_splits() -> DatasetSplits:
    train = Dataset.from_dict(
        {
            "text": [
                "World news sample",
                "Sports update sample",
                "Business report sample",
                "Science and tech sample",
            ],
            "label": [0, 1, 2, 3],
        }
    )
    validation = Dataset.from_dict(
        {
            "text": [
                "International market overview",
                "League season summary",
                "Corporate merger announcement",
                "New robotics breakthrough",
            ],
            "label": [0, 1, 2, 3],
        }
    )
    test = Dataset.from_dict(
        {
            "text": [
                "Global economy update",
                "Playoff match analysis",
                "Quarterly profits increase",
                "Chip manufacturing advances",
            ],
            "label": [0, 1, 2, 3],
        }
    )
    return DatasetSplits(train=train, validation=validation, test=test)


def _write_yaml(path: Path, content: str) -> None:
    path.write_text(content + "\n", encoding="utf-8")


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        headers = handle.readline().strip().split(",")
        for line in handle:
            if not line.strip():
                continue
            values = line.strip().split(",")
            rows.append(dict(zip(headers, values, strict=True)))
    return rows
