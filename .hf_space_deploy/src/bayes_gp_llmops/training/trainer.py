from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from .callbacks import CheckpointManager, EarlyStopping
from .config import TrainConfig
from .losses import classification_loss
from .metrics import compute_accuracy, compute_macro_f1, logits_to_predictions

LOGGER = logging.getLogger("bayes_gp_llmops.training.trainer")


@dataclass(frozen=True)
class TrainingArtifacts:
    """Paths for persisted training artifacts."""

    best_checkpoint_path: Path
    latest_checkpoint_path: Path
    history_path: Path
    resolved_config_path: Path


class Trainer:
    """Model trainer with checkpointing, scheduling, and early stopping."""

    def __init__(
        self,
        *,
        model: nn.Module,
        config: TrainConfig,
        device: torch.device,
        num_classes: int,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.num_classes = num_classes

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self._scheduler: LambdaLR | None = None
        self._amp_enabled = config.mixed_precision and device.type == "cuda"
        self._scaler = torch.amp.GradScaler(  # type: ignore[attr-defined]
            "cuda",
            enabled=self._amp_enabled,
        )
        self._checkpoint_manager = CheckpointManager(config.checkpoint_dir)
        self._early_stopping = EarlyStopping(patience=config.early_stopping_patience)

    def fit(
        self,
        *,
        train_loader: DataLoader[dict[str, torch.Tensor]],
        validation_loader: DataLoader[dict[str, torch.Tensor]],
        resolved_config: dict[str, Any],
    ) -> TrainingArtifacts:
        train_steps_per_epoch = _resolve_steps_per_epoch(
            len(train_loader),
            self.config.max_train_batches_per_epoch,
        )
        total_training_steps = max(1, train_steps_per_epoch * self.config.epochs)
        self._scheduler = self._build_scheduler(total_training_steps)
        self._checkpoint_manager.write_resolved_config(resolved_config)

        history: list[dict[str, Any]] = []
        for epoch in range(1, self.config.epochs + 1):
            train_metrics = self._run_epoch(
                loader=train_loader,
                training=True,
                max_batches=self.config.max_train_batches_per_epoch,
            )
            validation_metrics = self._run_epoch(
                loader=validation_loader,
                training=False,
                max_batches=self.config.max_validation_batches_per_epoch,
            )

            epoch_record = {
                "epoch": epoch,
                "train": train_metrics,
                "validation": validation_metrics,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
            }
            history.append(epoch_record)
            self._checkpoint_manager.write_history(history)

            checkpoint_state = self._build_checkpoint_state(
                epoch=epoch,
                last_metrics=epoch_record,
            )
            self._checkpoint_manager.save_latest(checkpoint_state)
            is_best = self._checkpoint_manager.save_best(
                checkpoint_state,
                validation_metrics["macro_f1"],
            )

            LOGGER.info(
                (
                    "epoch=%d train_loss=%.4f train_acc=%.4f val_loss=%.4f "
                    "val_acc=%.4f val_macro_f1=%.4f%s"
                ),
                epoch,
                train_metrics["loss"],
                train_metrics["accuracy"],
                validation_metrics["loss"],
                validation_metrics["accuracy"],
                validation_metrics["macro_f1"],
                " best_checkpoint_updated=true" if is_best else "",
            )

            if self._early_stopping.step(validation_metrics["macro_f1"]):
                LOGGER.info("Early stopping triggered at epoch %d.", epoch)
                break

        return TrainingArtifacts(
            best_checkpoint_path=self._checkpoint_manager.paths.best,
            latest_checkpoint_path=self._checkpoint_manager.paths.latest,
            history_path=self._checkpoint_manager.paths.history,
            resolved_config_path=self._checkpoint_manager.paths.resolved_config,
        )

    def _run_epoch(
        self,
        *,
        loader: DataLoader[dict[str, torch.Tensor]],
        training: bool,
        max_batches: int | None,
    ) -> dict[str, float]:
        if training:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_samples = 0
        all_predictions: list[int] = []
        all_labels: list[int] = []

        effective_batches = max_batches if max_batches is not None else len(loader)
        for batch_index, batch in enumerate(loader, start=1):
            if max_batches is not None and batch_index > max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            if training:
                self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(  # type: ignore[attr-defined]
                "cuda",
                enabled=self._amp_enabled,
            ):
                logits = self.model(input_ids, attention_mask=attention_mask)
                loss = classification_loss(logits, labels)

            if training:
                scaled_loss = self._scaler.scale(loss)
                torch.autograd.backward(scaled_loss)
                self._scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_norm,
                )
                self._scaler.step(self.optimizer)
                self._scaler.update()
                if self._scheduler is not None:
                    self._scheduler.step()

                if batch_index % self.config.log_frequency == 0:
                    LOGGER.info(
                        "mode=train batch=%d/%d loss=%.4f",
                        batch_index,
                        effective_batches,
                        float(loss.item()),
                    )

            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            all_predictions.extend(logits_to_predictions(logits.detach().cpu()))
            all_labels.extend(labels.detach().cpu().tolist())

        if total_samples == 0:
            raise ValueError("No samples were processed in the epoch.")

        return {
            "loss": total_loss / total_samples,
            "accuracy": compute_accuracy(all_predictions, all_labels),
            "macro_f1": compute_macro_f1(
                all_predictions,
                all_labels,
                num_classes=self.num_classes,
            ),
        }

    def _build_scheduler(self, total_training_steps: int) -> LambdaLR | None:
        if self.config.scheduler == "none":
            return None
        warmup_steps = int(total_training_steps * self.config.warmup_ratio)

        def lr_lambda(step_index: int) -> float:
            if warmup_steps > 0 and step_index < warmup_steps:
                return float(step_index + 1) / float(warmup_steps)
            progress_denominator = max(1, total_training_steps - warmup_steps)
            progress = float(step_index - warmup_steps) / float(progress_denominator)
            progress = min(max(progress, 0.0), 1.0)
            if self.config.scheduler == "linear":
                return max(0.0, 1.0 - progress)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(0.0, cosine)

        return LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    def _build_checkpoint_state(
        self,
        *,
        epoch: int,
        last_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self._scheduler.state_dict() if self._scheduler else None,
            "metrics": last_metrics,
        }


def load_checkpoint(
    *,
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: LambdaLR | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a model checkpoint and optionally restore optimizer/scheduler states."""

    checkpoint = torch.load(path, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint at {path} must contain a dictionary payload.")
    model_state = checkpoint.get("model_state_dict")
    if not isinstance(model_state, dict):
        raise ValueError("Checkpoint is missing model_state_dict.")
    model.load_state_dict(model_state)

    if optimizer is not None:
        optimizer_state = checkpoint.get("optimizer_state_dict")
        if isinstance(optimizer_state, dict):
            optimizer.load_state_dict(optimizer_state)

    if scheduler is not None:
        scheduler_state = checkpoint.get("scheduler_state_dict")
        if isinstance(scheduler_state, dict):
            scheduler.load_state_dict(scheduler_state)

    return checkpoint


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON payload with deterministic formatting."""

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _resolve_steps_per_epoch(total_batches: int, configured_limit: int | None) -> int:
    if configured_limit is None:
        return total_batches
    return min(total_batches, configured_limit)
