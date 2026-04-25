from __future__ import annotations

import json
import logging
from pathlib import Path

from tokenizers import Tokenizer

from bayes_gp_llmops.config import get_settings
from bayes_gp_llmops.data.config import DataPipelineConfig, load_data_config, to_serializable_config
from bayes_gp_llmops.data.datamodule import build_split_dataloaders
from bayes_gp_llmops.data.datasets import load_ag_news_splits, summarize_split_sizes
from bayes_gp_llmops.data.tokenizer import load_tokenizer, tokenizer_artifacts_exist
from bayes_gp_llmops.logging import configure_logging
from bayes_gp_llmops.models.classifier import TinyLlamaForSequenceClassification
from bayes_gp_llmops.models.config import ModelConfig, load_model_config, model_config_to_dict
from bayes_gp_llmops.seed import set_global_seed
from bayes_gp_llmops.tracking.mlflow_utils import (
    flatten_mapping,
    log_artifact_files,
    log_metrics,
    log_parameters,
    start_mlflow_run,
    write_json,
)
from bayes_gp_llmops.training.config import TrainConfig, load_train_config, train_config_to_dict
from bayes_gp_llmops.training.trainer import Trainer, TrainingArtifacts
from bayes_gp_llmops.utils.device import resolve_device

LOGGER = logging.getLogger("bayes_gp_llmops.training.pipeline")


def run_training_pipeline(
    *,
    data_config_path: Path,
    model_config_path: Path,
    train_config_path: Path,
    device_override: str | None,
    debug_mode: bool,
    data_config_override: DataPipelineConfig | None = None,
    model_config_override: ModelConfig | None = None,
    train_config_override: TrainConfig | None = None,
    mlflow_enabled: bool | None = None,
    mlflow_experiment_name: str | None = None,
    mlflow_run_name: str | None = None,
    mlflow_tags: dict[str, str] | None = None,
    mlflow_nested: bool = False,
) -> TrainingArtifacts:
    """Run end-to-end baseline training from configuration files."""

    settings = get_settings()
    configure_logging(settings.log_level)

    data_config = data_config_override or load_data_config(data_config_path)
    model_config = model_config_override or load_model_config(model_config_path)
    train_config = train_config_override or load_train_config(train_config_path)
    active_mlflow = settings.enable_mlflow if mlflow_enabled is None else mlflow_enabled
    experiment_name = mlflow_experiment_name or "bayes-gp-llmops-training"

    if device_override is not None:
        train_config = train_config.model_copy(update={"device_preference": device_override})
    if debug_mode:
        debug_subset_size = data_config.dataset.debug_subset_size
        if debug_subset_size is None:
            debug_subset_size = 1024
        debug_subset_size = min(debug_subset_size, 1024)
        data_config = data_config.model_copy(
            update={
                "dataset": data_config.dataset.model_copy(
                    update={"debug_subset_size": debug_subset_size}
                ),
                "dataloader": data_config.dataloader.model_copy(update={"batch_size": 16}),
            }
        )
        train_config = train_config.model_copy(
            update={
                "epochs": 1,
                "max_train_batches_per_epoch": 20,
                "max_validation_batches_per_epoch": 10,
            }
        )

    device = resolve_device(train_config.device_preference)
    set_global_seed(train_config.random_seed)

    if not tokenizer_artifacts_exist(data_config.paths.tokenizer_dir):
        raise FileNotFoundError(
            "Tokenizer artifacts were not found. Run scripts/download_data.py before training."
        )

    tokenizer = load_tokenizer(
        data_config.paths.tokenizer_dir,
        max_sequence_length=data_config.tokenizer.max_sequence_length,
    )
    _validate_tokenizer_compatibility(
        tokenizer=tokenizer,
        configured_vocab_size=model_config.vocab_size,
    )

    splits = load_ag_news_splits(data_config)
    split_sizes = summarize_split_sizes(splits)
    LOGGER.info("Loaded dataset splits: %s", split_sizes)

    dataloaders = build_split_dataloaders(
        splits,
        tokenizer=tokenizer,
        config=data_config,
        num_classes=model_config.num_classes,
    )
    model = TinyLlamaForSequenceClassification(model_config)
    trainer = Trainer(
        model=model,
        config=train_config,
        device=device,
        num_classes=model_config.num_classes,
    )

    resolved_config = {
        "data": to_serializable_config(data_config),
        "model": model_config_to_dict(model_config),
        "training": train_config_to_dict(train_config),
        "runtime": {
            "device": str(device),
            "debug_mode": debug_mode,
            "split_sizes": split_sizes,
        },
    }
    with start_mlflow_run(
        enabled=active_mlflow,
        experiment_name=experiment_name,
        run_name=mlflow_run_name or "baseline-train",
        nested=mlflow_nested,
        tags=mlflow_tags,
    ):
        log_parameters(flatten_mapping(resolved_config), enabled=active_mlflow)
        artifacts = trainer.fit(
            train_loader=dataloaders.train,
            validation_loader=dataloaders.validation,
            resolved_config=resolved_config,
        )
        summary_metrics = _extract_training_summary_metrics(artifacts.history_path)
        log_metrics(summary_metrics, enabled=active_mlflow)
        best_checkpoint_metadata_path = _write_best_checkpoint_metadata(
            artifacts=artifacts,
            split_sizes=split_sizes,
        )
        log_artifact_files(
            [
                artifacts.history_path,
                artifacts.resolved_config_path,
                best_checkpoint_metadata_path,
            ],
            enabled=active_mlflow,
            artifact_path="training",
        )
    LOGGER.info("best_checkpoint=%s", artifacts.best_checkpoint_path)
    LOGGER.info("latest_checkpoint=%s", artifacts.latest_checkpoint_path)
    LOGGER.info("history_path=%s", artifacts.history_path)
    LOGGER.info("resolved_config_path=%s", artifacts.resolved_config_path)
    return artifacts


def _validate_tokenizer_compatibility(*, tokenizer: Tokenizer, configured_vocab_size: int) -> None:
    trained_vocab_size = tokenizer.get_vocab_size()
    if configured_vocab_size < trained_vocab_size:
        raise ValueError(
            "Configured model vocab size is smaller than tokenizer vocabulary size: "
            f"{configured_vocab_size} < {trained_vocab_size}"
        )


def _extract_training_summary_metrics(history_path: Path) -> dict[str, float]:
    with history_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list) or not payload:
        raise ValueError("Training history must be a non-empty list.")

    best_validation_macro_f1 = 0.0
    final_train_loss = 0.0
    final_validation_loss = 0.0
    final_validation_macro_f1 = 0.0
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Training history entries must be mappings.")
        validation = item.get("validation")
        train = item.get("train")
        if isinstance(validation, dict):
            macro_f1 = validation.get("macro_f1")
            loss = validation.get("loss")
            if isinstance(macro_f1, (int, float)):
                best_validation_macro_f1 = max(best_validation_macro_f1, float(macro_f1))
                final_validation_macro_f1 = float(macro_f1)
            if isinstance(loss, (int, float)):
                final_validation_loss = float(loss)
        if isinstance(train, dict):
            train_loss = train.get("loss")
            if isinstance(train_loss, (int, float)):
                final_train_loss = float(train_loss)
    return {
        "best_validation_macro_f1": best_validation_macro_f1,
        "final_validation_macro_f1": final_validation_macro_f1,
        "final_validation_loss": final_validation_loss,
        "final_train_loss": final_train_loss,
    }


def _write_best_checkpoint_metadata(
    *,
    artifacts: TrainingArtifacts,
    split_sizes: dict[str, int],
) -> Path:
    checkpoint_path = artifacts.best_checkpoint_path
    metadata = {
        "best_checkpoint_path": str(checkpoint_path),
        "latest_checkpoint_path": str(artifacts.latest_checkpoint_path),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "split_sizes": split_sizes,
    }
    metadata_path = checkpoint_path.parent / "best_checkpoint_metadata.json"
    write_json(metadata_path, metadata)
    return metadata_path
